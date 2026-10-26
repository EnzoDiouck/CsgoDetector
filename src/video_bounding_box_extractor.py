import cv2
import os
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time

# ==================== CONFIGURATION GLOBALE ====================
DEFAULT_INTERVAL_SECONDS = 0.5      # Intervalle en secondes entre les captures
DEFAULT_MODEL_PATH = "yolov8n-pose.pt"  # Chemin vers le modèle YOLO pose
DEFAULT_OUTPUT_DIR = "bounding_box"      # Dossier de sortie par défaut
DEFAULT_MIN_KEYPOINTS = 8                # Nombre minimum de keypoints requis
DEFAULT_CONFIDENCE_THRESHOLD = 0.9       # Seuil de confiance pour les keypoints

# Liste des vidéos à traiter - Ajoutez vos vidéos ici
DEFAULT_VIDEO_LIST = [
    "./video/cs_go_ancient.mp4",
    "./video/cs_go_dust.mp4",
    "./video/cs_go_dust2.mp4",
    "./video/cs_go_mirage.mp4",
    "./video/cs_go_nuke.mp4",
    "./video/cs_go_overpass.mp4",
    "./video/cs_go_train.mp4",
    "./video/cs_go_vertigo.mp4",
    "./video/cs_go_jura.mp4",
    "./video/cs_go_1.mp4",
    "./video/cs_go_2.mp4",
    "./video/cs_go_3.mp4",
    "./video/cs_go_4.mp4"
]

# Valeurs par défaut pour les vidéos non configurées individuellement
DEFAULT_START_TIME = 0                  # Temps de début en secondes (0 = début de vidéo)
DEFAULT_END_TIME = None                  # Temps de fin en secondes (None = fin de vidéo)
DEFAULT_DURATION = None                  # Durée de capture en secondes (None = jusqu'à la fin)
# ================================================================


class VideoBoundingBoxExtractor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, output_dir=DEFAULT_OUTPUT_DIR, min_keypoints=DEFAULT_MIN_KEYPOINTS):
        """
        Extracteur de bounding boxes depuis une vidéo avec détection de pose YOLO.
        
        Args:
            model_path (str): Chemin vers le modèle YOLOv8-pose
            output_dir (str): Dossier de sortie pour les bounding boxes
            min_keypoints (int): Nombre minimum de points de pose requis
        """
        self.model = YOLO(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.min_keypoints = min_keypoints
        self.frame_count = 0
        self.all_detections = []  # Liste pour stocker toutes les détections de la vidéo
        self.saved_file_counter = 0  # Compteur pour nommer les fichiers séquentiellement
        
    def count_valid_keypoints(self, keypoints):
        """
        Compte le nombre de points de pose valides (confiance > 0.5).
        
        Args:
            keypoints: Array des keypoints de YOLO [x, y, confidence]
            
        Returns:
            int: Nombre de keypoints valides
        """
        if keypoints is None or len(keypoints) == 0:
            return 0
            
        # Les keypoints sont au format [x, y, confidence]
        # On compte ceux avec une confiance > seuil défini
        valid_count = 0
        for i in range(0, len(keypoints), 3):
            if i + 2 < len(keypoints) and keypoints[i + 2] > DEFAULT_CONFIDENCE_THRESHOLD:
                valid_count += 1
                
        return valid_count
    
    def process_frame(self, frame, frame_number):
        """
        Traite une frame et sauvegarde l'image complète avec bounding boxes et les positions dans un fichier texte.
        Règles de filtrage :
        - Ignore les frames avec une seule personne ayant plus de 15 keypoints
        - Ignore les frames avec plus de 3 personnes
        - Ignore les frames avec une seule personne dont la bounding box fait >85% de la hauteur de l'écran
        
        Args:
            frame: Image à traiter
            frame_number: Numéro de la frame
            
        Returns:
            int: Nombre de personnes détectées et sauvegardées
        """
        # Détection avec YOLO pose
        results = self.model(frame, verbose=False)
        
        saved_count = 0
        frame_detections = []  # Liste pour stocker les positions de cette frame
        
        # Copier l'image pour dessiner les bounding boxes
        frame_with_boxes = frame.copy()
        
        for r in results:
            # Vérifier s'il y a des détections
            if r.boxes is None or len(r.boxes) == 0:
                continue
                
            boxes = r.boxes.xyxy.cpu().numpy()  # Bounding boxes
            
            # Vérifier s'il y a des keypoints
            if hasattr(r, 'keypoints') and r.keypoints is not None:
                keypoints = r.keypoints.data.cpu().numpy()
                
                valid_persons = []  # Liste temporaire pour stocker les personnes valides
                
                for i, (box, kpts) in enumerate(zip(boxes, keypoints)):
                    # Compter les keypoints valides
                    kpts_flat = kpts.flatten()
                    valid_keypoints = self.count_valid_keypoints(kpts_flat)
                    
                    if valid_keypoints >= self.min_keypoints:
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Stocker les informations de cette personne
                        person_info = {
                            'index': i,
                            'box': (x1, y1, x2, y2),
                            'keypoints_count': valid_keypoints
                        }
                        valid_persons.append(person_info)
                
                # Vérifier les règles de filtrage :
                
                # 2. Ignorer si plus de 3 personnes détectées
                if len(valid_persons) > 3:
                    print(f"⚠️ Frame {frame_number}: Ignorée - trop de personnes détectées ({len(valid_persons)} > 3)")
                    return 0
                
                # 3. Ignorer si une seule personne avec une bounding box trop haute (>85% de l'écran)
                if len(valid_persons) == 1:
                    img_height, img_width = frame.shape[:2]
                    x1, y1, x2, y2 = valid_persons[0]['box']
                    bbox_height = y2 - y1
                    
                    # Calculer le pourcentage de hauteur de l'image occupé par la bounding box
                    height_ratio = bbox_height / img_height
                    print(f"Debug: Frame {frame_number}, Height Ratio: {height_ratio:.2f}")
                    
                    # Ignorer si la bounding box occupe plus de 80% de la hauteur de l'image
                    if height_ratio > 0.78:
                        print(f"⚠️ Frame {frame_number}: Ignorée - bounding box trop haute ({height_ratio:.1%} de l'écran)")
                        return 0
                
                # Si on arrive ici, la frame est valide, traiter les personnes
                for person_info in valid_persons:
                    i = person_info['index']
                    x1, y1, x2, y2 = person_info['box']
                    valid_keypoints = person_info['keypoints_count']
                    
                    # Dessiner la bounding box sur l'image
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Ajouter le label avec le nombre de keypoints
                    label = f"Person {i+1} ({valid_keypoints} kpts)"
                    cv2.putText(frame_with_boxes, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Stocker les informations de position
                    detection_info = {
                        'frame_number': frame_number,
                        'person_id': i + 1,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'keypoints_count': valid_keypoints
                    }
                    frame_detections.append(detection_info)
                    
                    # Ajouter à la liste globale des détections
                    self.all_detections.append(detection_info)
                    
                    saved_count += 1
                    
                    print(f"✅ Détecté: Person {i+1} à ({x1},{y1},{x2},{y2}) avec {valid_keypoints} keypoints")
            else:
                # Pas de keypoints détectés, ignorer cette frame
                print(f"❌ Frame {frame_number}: Aucun keypoint détecté")
        
        # Sauvegarder les images si des personnes ont été détectées
        if saved_count > 0:
            # Générer les noms des fichiers avec compteur séquentiel
            image_original_filename = f"{self.saved_file_counter}.jpg"
            image_with_boxes_filename = f"{self.saved_file_counter}_box.jpg"
            
            image_original_filepath = self.output_dir / image_original_filename
            image_with_boxes_filepath = self.output_dir / image_with_boxes_filename
            
            # 1. Sauvegarder l'image originale complète (sans modifications)
            cv2.imwrite(str(image_original_filepath), frame)
            
            # 2. Sauvegarder l'image avec les bounding boxes dessinées
            cv2.imwrite(str(image_with_boxes_filepath), frame_with_boxes)
            
            # 3. Sauvegarder chaque bounding box isolée
            bbox_filenames = []
            for i, detection in enumerate(frame_detections):
                x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
                
                # S'assurer que les coordonnées sont dans les limites de l'image
                img_height, img_width = frame.shape[:2]
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))
                
                # Extraire la région de la bounding box
                bbox_crop = frame[y1:y2, x1:x2]
                
                if bbox_crop.size > 0:  # Vérifier que l'image croppée n'est pas vide
                    bbox_filename = f"{self.saved_file_counter}_{i}.jpg"
                    bbox_filepath = self.output_dir / bbox_filename
                    cv2.imwrite(str(bbox_filepath), bbox_crop)
                    bbox_filenames.append(bbox_filename)
            
            # 4. Sauvegarder les annotations au format YOLO
            yolo_filename = f"{self.saved_file_counter}.txt"
            yolo_filepath = self.output_dir / yolo_filename
            
            # Obtenir les dimensions de l'image
            img_height, img_width = frame.shape[:2]
            
            with open(yolo_filepath, 'w', encoding='utf-8') as f:
                for detection in frame_detections:
                    # Calculer les valeurs normalisées pour le format YOLO
                    x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
                    
                    # Centre de la bounding box
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    
                    # Largeur et hauteur de la bounding box
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    # Normalisation (diviser par les dimensions de l'image)
                    center_x_norm = center_x / img_width
                    center_y_norm = center_y / img_height
                    width_norm = bbox_width / img_width
                    height_norm = bbox_height / img_height
                    
                    # Format YOLO: class_id center_x center_y width height
                    # Classe 0 pour "person"
                    f.write(f"0 {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
            
            # Messages de confirmation
            print(f"💾 Image originale: {image_original_filename}")
            print(f"🎨 Image avec boxes: {image_with_boxes_filename}")
            if bbox_filenames:
                print(f"✂️ Bounding boxes isolées: {', '.join(bbox_filenames)}")
            print(f"📄 Annotations YOLO: {yolo_filename}")
            
            # Incrémenter le compteur pour le prochain fichier
            self.saved_file_counter += 1
                
        return saved_count
    
    def save_global_summary(self, video_path, start_time, end_time, fps):
        """
        Sauvegarde un fichier de résumé global avec toutes les positions détectées dans la vidéo.
        
        Args:
            video_path (str): Chemin de la vidéo traitée
            start_time (float): Temps de début du segment
            end_time (float): Temps de fin du segment
            fps (float): FPS de la vidéo
        """
        if not self.all_detections:
            print("⚠️ Aucune détection à sauvegarder dans le fichier global")
            return
        
        # Nom du fichier de résumé
        video_name = Path(video_path).stem
        summary_filename = f"GLOBAL_SUMMARY_{video_name}.txt"
        summary_filepath = self.output_dir / summary_filename
        
        # Statistiques
        total_detections = len(self.all_detections)
        frames_with_detections = len(set(d['frame_number'] for d in self.all_detections))
        avg_keypoints = sum(d['keypoints_count'] for d in self.all_detections) / total_detections
        
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RÉSUMÉ GLOBAL DES DÉTECTIONS DE POSES\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Vidéo traitée: {video_path}\n")
            f.write(f"Segment analysé: {start_time:.2f}s → {end_time:.2f}s\n")
            f.write(f"FPS de la vidéo: {fps:.2f}\n")
            f.write(f"Keypoints minimum requis: {self.min_keypoints}\n")
            f.write(f"Règles de filtrage: \n")
            f.write(f"  - Frames avec une seule personne ayant >15 keypoints sont ignorées\n")
            f.write(f"  - Frames avec plus de 3 personnes sont ignorées\n")
            f.write(f"  - Frames avec une seule personne dont la bounding box fait >70% de la hauteur de l'écran sont ignorées\n")
            f.write(f"Format des annotations: YOLO (classe 0 = person)\n")
            f.write(f"Dossier de sortie: {self.output_dir}\n\n")
            
            f.write("STATISTIQUES GLOBALES:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total de détections: {total_detections}\n")
            f.write(f"Frames avec détections: {frames_with_detections}\n")
            f.write(f"Keypoints moyens par détection: {avg_keypoints:.1f}\n\n")
            
            f.write("DÉTAIL DE TOUTES LES DÉTECTIONS:\n")
            f.write("-" * 40 + "\n\n")
            
            # Regrouper par frame
            frames = {}
            for detection in self.all_detections:
                frame_num = detection['frame_number']
                if frame_num not in frames:
                    frames[frame_num] = []
                frames[frame_num].append(detection)
            
            # Écrire les détections par frame
            for frame_num in sorted(frames.keys()):
                frame_detections = frames[frame_num]
                time_seconds = frame_num / fps
                
                f.write(f"Frame {frame_num} (temps: {time_seconds:.2f}s)\n")
                f.write(f"  Nombre de personnes: {len(frame_detections)}\n")
                
                for i, detection in enumerate(frame_detections, 1):
                    x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
                    
                    # Calculer les coordonnées YOLO normalisées (approximation pour le résumé)
                    # Note: Les dimensions exactes de l'image ne sont pas stockées dans detection
                    # Ces valeurs sont indicatives
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    f.write(f"  Personne {i}:\n")
                    f.write(f"    Position absolue: ({x1}, {y1}) → ({x2}, {y2})\n")
                    f.write(f"    Centre: ({center_x:.0f}, {center_y:.0f})\n")
                    f.write(f"    Taille: {width}x{height} pixels\n")
                    f.write(f"    Keypoints: {detection['keypoints_count']}\n")
                    f.write(f"    Format YOLO: 0 [centre_x_norm] [centre_y_norm] [width_norm] [height_norm]\n")
                
                f.write("\n")
        
        print(f"📊 Fichier de résumé global créé: {summary_filename}")
        print(f"   Total détections: {total_detections}")
        print(f"   Frames avec détections: {frames_with_detections}")
        
        return summary_filepath
    
    def process_video(self, video_path, interval_seconds=DEFAULT_INTERVAL_SECONDS, 
                     start_time=DEFAULT_START_TIME, end_time=DEFAULT_END_TIME, duration=DEFAULT_DURATION,
                     output_subdir=None):
        """
        Traite une vidéo complète ou un segment.
        
        Args:
            video_path (str): Chemin vers la vidéo
            interval_seconds (float): Intervalle en secondes entre les captures
            start_time (float): Temps de début en secondes (0 = début de vidéo)
            end_time (float): Temps de fin en secondes (None = fin de vidéo)
            duration (float): Durée de capture en secondes (None = jusqu'à la fin)
            output_subdir (str): Sous-dossier pour cette vidéo (None = dossier principal)
        """
        # Déterminer le dossier de sortie pour cette vidéo
        if output_subdir:
            current_output_dir = self.output_dir / output_subdir
            current_output_dir.mkdir(exist_ok=True)
        else:
            current_output_dir = self.output_dir
            
        # Stocker temporairement le dossier de sortie original
        original_output_dir = self.output_dir
        self.output_dir = current_output_dir
        
        # Réinitialiser la liste des détections pour cette vidéo
        self.all_detections = []
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            # Restaurer le dossier de sortie original
            self.output_dir = original_output_dir
            raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
        
        # Obtenir les propriétés de la vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        # Calculer les temps de début et fin
        if duration is not None:
            # Si une durée est spécifiée, calculer le temps de fin
            calculated_end_time = start_time + duration
        elif end_time is not None:
            # Si un temps de fin est spécifié, l'utiliser
            calculated_end_time = end_time
        else:
            # Sinon, aller jusqu'à la fin de la vidéo
            calculated_end_time = video_duration
        
        # S'assurer que les temps sont dans les limites de la vidéo
        start_time = max(0, min(start_time, video_duration))
        calculated_end_time = max(start_time, min(calculated_end_time, video_duration))
        
        # Calculer les frames correspondantes
        start_frame = int(start_time * fps)
        end_frame = int(calculated_end_time * fps)
        segment_duration = calculated_end_time - start_time
        
        print(f"📹 Vidéo: {video_path}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Frames totales: {total_frames}")
        print(f"   Durée totale: {video_duration:.2f} secondes")
        print(f"   📊 SEGMENT À TRAITER:")
        print(f"      Début: {start_time:.2f}s (frame {start_frame})")
        print(f"      Fin: {calculated_end_time:.2f}s (frame {end_frame})")
        print(f"      Durée du segment: {segment_duration:.2f} secondes")
        print(f"   Intervalle de capture: {interval_seconds} secondes")
        print(f"   Keypoints minimum requis: {self.min_keypoints}")
        print(f"   Dossier de sortie: {self.output_dir}")
        print("-" * 50)
        
        # Calculer l'intervalle en frames
        frame_interval = int(fps * interval_seconds)
        
        # Commencer au frame de début spécifié
        current_frame = start_frame
        total_saved = 0
        processed_frames = 0
        
        processing_start_time = time.time()
        
        while current_frame < end_frame:
            # Positionner à la frame désirée
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Traiter la frame
            saved_count = self.process_frame(frame, current_frame)
            total_saved += saved_count
            processed_frames += 1
            
            # Calculer le progrès dans le segment
            segment_progress = ((current_frame - start_frame) / (end_frame - start_frame)) * 100
            current_time = current_frame / fps
            elapsed = time.time() - processing_start_time
            
            # Estimer le temps restant
            if processed_frames > 0:
                frames_remaining = (end_frame - current_frame) // frame_interval
                eta = (elapsed / processed_frames) * frames_remaining
            else:
                eta = 0
            
            print(f"🔄 Progrès: {segment_progress:.1f}% | Temps: {current_time:.1f}s/{calculated_end_time:.1f}s | "
                  f"Frame {current_frame} | Sauvegardées: {saved_count} | Total: {total_saved} | ETA: {eta:.0f}s")
            
            # Passer à la prochaine frame
            current_frame += frame_interval
        
        cap.release()
        
        # Sauvegarder le fichier de résumé global
        self.save_global_summary(video_path, start_time, calculated_end_time, fps)
        
        print("-" * 50)
        print(f"✅ Traitement terminé!")
        print(f"   Segment traité: {start_time:.2f}s → {calculated_end_time:.2f}s")
        print(f"   Frames traitées: {processed_frames}")
        print(f"   Personnes sauvegardées: {total_saved}")
        print(f"   Temps de traitement: {time.time() - processing_start_time:.2f} secondes")
        print(f"   Images sauvegardées dans: {current_output_dir}")
        
        # Restaurer le dossier de sortie original
        self.output_dir = original_output_dir
        
        return total_saved, processed_frames
    
    def process_video_list(self, video_list=None):
        """
        Traite une liste de vidéos avec les mêmes paramètres par défaut.
        
        Args:
            video_list (list): Liste des chemins des vidéos à traiter
            
        Returns:
            dict: Statistiques de traitement pour chaque vidéo
        """
        if video_list is None:
            video_list = DEFAULT_VIDEO_LIST
        
        total_stats = {
            "videos_processed": 0,
            "videos_failed": 0,
            "total_persons_saved": 0,
            "total_frames_processed": 0,
            "processing_times": []
        }
        
        results = {}
        
        print("🎬 TRAITEMENT DE PLUSIEURS VIDÉOS")
        print(f"📋 Nombre de vidéos à traiter: {len(video_list)}")
        print(f"🤖 Modèle: {DEFAULT_MODEL_PATH}")
        print(f"📁 Dossier de sortie principal: {self.output_dir}")
        print("=" * 60)
        
        for i, video_path in enumerate(video_list, 1):
            print(f"\n🎥 [{i}/{len(video_list)}] Traitement de: {video_path}")
            
            # Vérifier que la vidéo existe
            if not os.path.exists(video_path):
                print(f"❌ Erreur: La vidéo '{video_path}' n'existe pas")
                results[video_path] = {"status": "failed", "error": "File not found"}
                total_stats["videos_failed"] += 1
                continue
            
            # Utiliser les paramètres par défaut pour toutes les vidéos
            start_time = DEFAULT_START_TIME
            end_time = DEFAULT_END_TIME
            duration = DEFAULT_DURATION
            interval_seconds = DEFAULT_INTERVAL_SECONDS
            output_subdir = None  # Toutes les vidéos dans le même dossier
            
            # Validation des paramètres
            if end_time is not None and duration is not None:
                print(f"❌ Erreur pour {video_path}: end_time et duration ne peuvent pas être définis simultanément")
                results[video_path] = {"status": "failed", "error": "Invalid configuration"}
                total_stats["videos_failed"] += 1
                continue
                
            if start_time < 0:
                print(f"❌ Erreur pour {video_path}: start_time doit être >= 0")
                results[video_path] = {"status": "failed", "error": "Invalid start_time"}
                total_stats["videos_failed"] += 1
                continue
                
            if end_time is not None and end_time <= start_time:
                print(f"❌ Erreur pour {video_path}: end_time doit être supérieur à start_time")
                results[video_path] = {"status": "failed", "error": "Invalid end_time"}
                total_stats["videos_failed"] += 1
                continue
                
            if duration is not None and duration <= 0:
                print(f"❌ Erreur pour {video_path}: duration doit être > 0")
                results[video_path] = {"status": "failed", "error": "Invalid duration"}
                total_stats["videos_failed"] += 1
                continue
            
            try:
                video_start_time = time.time()
                
                # Traiter la vidéo
                persons_saved, frames_processed = self.process_video(
                    video_path=video_path,
                    interval_seconds=interval_seconds,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    output_subdir=output_subdir
                )
                
                processing_time = time.time() - video_start_time
                
                # Enregistrer les résultats
                results[video_path] = {
                    "status": "success",
                    "persons_saved": persons_saved,
                    "frames_processed": frames_processed,
                    "processing_time": processing_time
                }
                
                # Mettre à jour les statistiques globales
                total_stats["videos_processed"] += 1
                total_stats["total_persons_saved"] += persons_saved
                total_stats["total_frames_processed"] += frames_processed
                total_stats["processing_times"].append(processing_time)
                
                print(f"✅ Vidéo {i}/{len(video_list)} terminée avec succès!")
                
            except Exception as e:
                print(f"❌ Erreur lors du traitement de {video_path}: {e}")
                results[video_path] = {"status": "failed", "error": str(e)}
                total_stats["videos_failed"] += 1
        
        # Afficher les statistiques finales
        print("\n" + "=" * 60)
        print("📊 STATISTIQUES FINALES")
        print(f"✅ Vidéos traitées avec succès: {total_stats['videos_processed']}")
        print(f"❌ Vidéos échouées: {total_stats['videos_failed']}")
        print(f"👥 Total de personnes sauvegardées: {total_stats['total_persons_saved']}")
        print(f"🖼️ Total de frames traitées: {total_stats['total_frames_processed']}")
        
        if total_stats["processing_times"]:
            avg_time = sum(total_stats["processing_times"]) / len(total_stats["processing_times"])
            total_time = sum(total_stats["processing_times"])
            print(f"⏱️ Temps de traitement total: {total_time:.2f} secondes")
            print(f"⏱️ Temps moyen par vidéo: {avg_time:.2f} secondes")
        
        print("\n📋 DÉTAIL PAR VIDÉO:")
        for video_path, result in results.items():
            status_icon = "✅" if result["status"] == "success" else "❌"
            print(f"{status_icon} {video_path}")
            if result["status"] == "success":
                print(f"    👥 Personnes: {result['persons_saved']}")
                print(f"    🖼️ Frames: {result['frames_processed']}")
                print(f"    ⏱️ Temps: {result['processing_time']:.2f}s")
            else:
                print(f"    ❌ Erreur: {result['error']}")
        
        return results, total_stats


def main():
    """
    Fonction principale qui traite une liste de vidéos.
    Modifiez DEFAULT_VIDEO_LIST en haut du fichier pour ajouter vos vidéos.
    Toutes les vidéos utilisent les mêmes paramètres par défaut.
    """
    
    # Vérifier que la liste de vidéos n'est pas vide
    if not DEFAULT_VIDEO_LIST:
        print("❌ Erreur: DEFAULT_VIDEO_LIST est vide")
        print("💡 Ajoutez des vidéos à DEFAULT_VIDEO_LIST dans les configurations globales")
        return
    
    # Vérifier que le modèle existe
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"❌ Erreur: Le modèle '{DEFAULT_MODEL_PATH}' n'existe pas")
        print(f"💡 Modifiez DEFAULT_MODEL_PATH dans les configurations globales")
        return
    
    # Validation des configurations par défaut
    if DEFAULT_END_TIME is not None and DEFAULT_DURATION is not None:
        print("❌ Erreur: Vous ne pouvez pas spécifier à la fois DEFAULT_END_TIME et DEFAULT_DURATION")
        return
    
    if DEFAULT_START_TIME < 0:
        print("❌ Erreur: DEFAULT_START_TIME doit être >= 0")
        return
        
    if DEFAULT_END_TIME is not None and DEFAULT_END_TIME <= DEFAULT_START_TIME:
        print("❌ Erreur: DEFAULT_END_TIME doit être supérieur à DEFAULT_START_TIME")
        return
        
    if DEFAULT_DURATION is not None and DEFAULT_DURATION <= 0:
        print("❌ Erreur: DEFAULT_DURATION doit être > 0")
        return
    
    try:
        print("� Démarrage du traitement multi-vidéos:")
        print(f"   📋 Nombre de vidéos: {len(DEFAULT_VIDEO_LIST)}")
        print(f"   🤖 Modèle: {DEFAULT_MODEL_PATH}")
        print(f"   📁 Sortie: {DEFAULT_OUTPUT_DIR}")
        print(f"   ⏱️ Intervalle par défaut: {DEFAULT_INTERVAL_SECONDS}s")
        print(f"   🎯 Keypoints min: {DEFAULT_MIN_KEYPOINTS}")
        print(f"   ⏰ Début par défaut: {DEFAULT_START_TIME}s")
        if DEFAULT_DURATION is not None:
            print(f"   ⏳ Durée par défaut: {DEFAULT_DURATION}s")
        elif DEFAULT_END_TIME is not None:
            print(f"   🏁 Fin par défaut: {DEFAULT_END_TIME}s")
        else:
            print(f"   🏁 Fin par défaut: jusqu'à la fin de chaque vidéo")
        print("\n📋 Vidéos à traiter:")
        for i, video in enumerate(DEFAULT_VIDEO_LIST, 1):
            status = "✅" if os.path.exists(video) else "❌ (fichier introuvable)"
            print(f"   {i}. {video} {status}")
        print("=" * 60)
        
        # Créer l'extracteur avec les configurations globales
        extractor = VideoBoundingBoxExtractor()
        
        # Traiter toutes les vidéos
        results, total_stats = extractor.process_video_list()
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
