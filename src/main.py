from ultralytics import YOLO
import mss
import numpy as np
import cv2
import time
import keyboard
import win32gui
import win32con

MODEL_PATH = "runs/detect/train3/weights/last.pt"
CONF = 0.40
IOU = 0.45
FRAME_SKIP = 1  # ✅ Maximum réactivité
LINE_WIDTH = 2
FONT_SIZE = 0.5

def find_csgo_window():
    """Trouve la fenêtre CS:GO et retourne ses coordonnées"""
    def enum_windows_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if any(keyword in window_title.lower() for keyword in ['counter-strike', 'cs2', 'csgo']):
                windows.append((hwnd, window_title))
        return True
    
    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    
    if windows:
        hwnd, title = windows[0]
        rect = win32gui.GetWindowRect(hwnd)
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top
        
        print(f"[INFO] Found CS:GO window: '{title}' at ({left}, {top}) size {width}x{height}")
        
        return {
            'top': top,
            'left': left,
            'width': width,
            'height': height
        }
    else:
        print("[WARNING] CS:GO window not found, using full screen")
        return None

def get_capture_region():
    """Obtient la région de capture optimale"""
    # Essayer de trouver CS:GO d'abord
    csgo_region = find_csgo_window()
    
    if csgo_region:
        # Ajuster si la fenêtre est trop grande
        if csgo_region['width'] > 1920 or csgo_region['height'] > 1080:
            print("[INFO] CS:GO window is large, using reduced capture area")
            # Capturer une zone centrale
            center_x = csgo_region['left'] + csgo_region['width'] // 2
            center_y = csgo_region['top'] + csgo_region['height'] // 2
            
            capture_width = min(1280, csgo_region['width'])
            capture_height = min(720, csgo_region['height'])
            
            return {
                'top': center_y - capture_height // 2,
                'left': center_x - capture_width // 2,
                'width': capture_width,
                'height': capture_height
            }
        else:
            return csgo_region
    else:
        # Fallback: capturer le centre de l'écran principal
        sct_temp = mss.mss()
        monitor = sct_temp.monitors[1]
        
        capture_width = 1280
        capture_height = 720
        center_x = monitor['width'] // 2
        center_y = monitor['height'] // 2
        
        return {
            'top': center_y - capture_height // 2,
            'left': center_x - capture_width // 2,
            'width': capture_width,
            'height': capture_height
        }

# ✅ Fonction pour déterminer la résolution optimale
def get_optimal_resolution():
    """Détermine la résolution optimale basée sur la région de capture"""
    test_region = get_capture_region()
    
    # Si CS:GO est déjà petit, on garde sa taille
    if test_region['width'] <= 800 and test_region['height'] <= 600:
        return test_region['width'], test_region['height']
    
    # Sinon, on redimensionne intelligemment
    aspect_ratio = test_region['width'] / test_region['height']
    
    if aspect_ratio > 1.7:  # Format ultra-wide
        return 960, 540
    elif aspect_ratio > 1.5:  # Format 16:9 classique
        return 854, 480
    else:  # Format 4:3 ou carré
        return 640, 480

# ✅ Paramètres d'optimisation dynamiques
DISPLAY_WIDTH, DISPLAY_HEIGHT = get_optimal_resolution()
TARGET_FPS = 60  # ✅ FPS cible plus élevé maintenant

# ✅ Couleurs vives pour chaque classe (simplifié à 2 types)
CLASS_COLORS = {
    'CT': (0, 255, 0),        # Vert vif (BGR format)
    'T': (0, 165, 255),       # Orange vif (BGR format)
}

def calculate_iou(box1, box2):
    """Calcule l'IoU (Intersection over Union) entre deux boîtes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculer l'intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def filter_overlapping_detections(detections, iou_threshold=0.3):
    """Filtre les détections qui se chevauchent, gardant celle avec la confiance la plus élevée"""
    if len(detections) <= 1:
        return detections
    
    # Trier par confiance décroissante
    detections.sort(key=lambda x: x['conf'], reverse=True)
    
    filtered = []
    for detection in detections:
        # Vérifier si cette détection chevauche avec une détection déjà filtrée
        overlap = False
        for filtered_detection in filtered:
            iou = calculate_iou(detection['box'], filtered_detection['box'])
            if iou > iou_threshold:
                overlap = True
                break
        
        if not overlap:
            filtered.append(detection)
    
    return filtered

def draw_custom_boxes(img, results):
    """Dessine des boîtes avec des couleurs personnalisées selon la classe et gère les overlaps"""
    if not results or len(results) == 0:
        return img
    
    # Copier l'image pour ne pas modifier l'originale
    annotated = img.copy()
    
    # Récupérer les détections
    boxes = results[0].boxes
    if boxes is None:
        return annotated
    
    # Récupérer les noms de classes du modèle
    class_names = results[0].names
    
    # Préparer les détections avec simplification des classes
    detections = []
    for box in boxes:
        # Coordonnées de la boîte
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        
        # Confiance
        conf = float(box.conf[0].cpu().numpy())
        
        # Classe
        class_id = int(box.cls[0].cpu().numpy())
        original_class_name = class_names[class_id].lower()
        
        # Simplifier les classes : regrouper head et body
        if 'anti' in original_class_name:
            simplified_class = 'CT'
        elif 'terro' in original_class_name:
            simplified_class = 'T'
        else:
            continue  # Ignorer les autres classes
        
        detections.append({
            'box': (x1, y1, x2, y2),
            'conf': conf,
            'class': simplified_class
        })
    
    # Filtrer les détections qui se chevauchent
    filtered_detections = filter_overlapping_detections(detections)
    
    # Dessiner les détections filtrées
    for detection in filtered_detections:
        x1, y1, x2, y2 = detection['box']
        conf = detection['conf']
        class_name = detection['class']
        
        # Choisir la couleur selon la classe
        color = CLASS_COLORS.get(class_name, (255, 255, 255))  # Blanc par défaut
        
        # Dessiner le rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, LINE_WIDTH)
        
        # Préparer le texte
        label = f"{class_name} {conf:.2f}"
        
        # Calculer la taille du texte
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, 1
        )
        
        # Dessiner le fond du texte
        cv2.rectangle(
            annotated,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Dessiner le texte en blanc
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            (255, 255, 255),  # Texte blanc
            1
        )
    
    return annotated

def draw_color_legend(img):
    """Ajoute une légende des couleurs dans le coin de l'image"""
    legend_height = 120
    legend_width = 200
    
    # Position de la légende (coin supérieur droit)
    start_x = img.shape[1] - legend_width - 10
    start_y = 10
    
    # Fond semi-transparent pour la légende
    overlay = img.copy()
    cv2.rectangle(overlay, (start_x, start_y), (start_x + legend_width, start_y + legend_height), (0, 0, 0), -1)
    img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    # Titre de la légende
    cv2.putText(img, "Classes:", (start_x + 5, start_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Dessiner chaque classe avec sa couleur
    y_offset = 35
    for class_name, color in CLASS_COLORS.items():
        y_pos = start_y + y_offset
        
        # Rectangle coloré
        cv2.rectangle(img, (start_x + 5, y_pos - 8), (start_x + 20, y_pos + 2), color, -1)
        
        # Nom de la classe
        cv2.putText(img, class_name, (start_x + 25, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
    
    return img

def stream_screen():
    """Stream screen and run YOLO every N frames without creating new windows."""
    model = YOLO(MODEL_PATH)
    
    # ✅ Optimisations du modèle
    model.fuse()  # Fuser les couches pour plus de rapidité
    
    sct = mss.mss()
    
    # ✅ Obtenir la région de capture optimisée (CS:GO ou zone réduite)
    capture_region = get_capture_region()
    
    print(f"[INFO] Capturing region: {capture_region['width']}x{capture_region['height']} at ({capture_region['left']}, {capture_region['top']})")
    print(f"[INFO] Display resolution: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    
    window_name = "YOLO Stream - CS:GO Optimized"

    # ✅ Create a single window manually once
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    frame_id = 0
    last_frame = None
    frame_time_target = 1.0 / TARGET_FPS  # ✅ Temps cible par frame

    while True:
        start_time = time.perf_counter()
        
        # ✅ Vérifier si la fenêtre existe encore
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Window closed by user, exiting.")
                break
        except cv2.error:
            print("[INFO] Window closed or destroyed, exiting.")
            break
        
        # ✅ Capture d'écran plus rapide avec région réduite si possible
        # ✅ Capture d'écran optimisée sur la région CS:GO
        img = np.array(sct.grab(capture_region))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # ✅ Redimensionner pour l'affichage si nécessaire (optimisé)
        if img.shape[1] != DISPLAY_WIDTH or img.shape[0] != DISPLAY_HEIGHT:
            img_resized = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
        else:
            img_resized = img

        # Run YOLO only every N frames
        if frame_id % FRAME_SKIP == 0:
            # ✅ Utiliser l'image redimensionnée pour YOLO (plus rapide)
            results = model(img_resized, conf=CONF, iou=IOU, verbose=False, imgsz=416)  # ✅ Taille réduite pour plus de FPS
            try:
                # ✅ Utiliser notre fonction de dessin personnalisée
                annotated = draw_custom_boxes(img_resized, results)
            except (TypeError, AttributeError) as e:
                print(f"[WARNING] Custom drawing failed: {e}, using default")
                annotated = results[0].plot(line_width=LINE_WIDTH, font_size=FONT_SIZE)
            last_frame = annotated
        
        # ✅ Toujours afficher quelque chose
        display_frame = last_frame if last_frame is not None else img_resized
        
        # ✅ Ajouter la légende des couleurs
        display_frame = draw_color_legend(display_frame)

        # ✅ Affichage optimisé
        try:
            cv2.imshow(window_name, display_frame)
        except cv2.error:
            print("[INFO] Cannot display frame, window may be closed.")
            break

        frame_id += 1
        
        # ✅ Calcul FPS plus efficace
        frame_time = time.perf_counter() - start_time
        fps = 1 / frame_time if frame_time > 0 else 0
        
        # ✅ Mettre à jour le titre pour un meilleur feedback FPS
        if frame_id % 5 == 0:  # Toutes les 5 frames pour voir les changements FPS
            try:
                fps_text = f"YOLO CS:GO Stream ({fps:.1f} FPS) — Press NUM1 or ESC to exit"
                cv2.setWindowTitle(window_name, fps_text)
            except cv2.error:
                pass

        # ✅ Actualiser la région de capture périodiquement (au cas où CS:GO bouge)
        if frame_id % 300 == 0:  # Toutes les 300 frames (~10 secondes)
            new_region = get_capture_region()
            if new_region != capture_region:
                capture_region = new_region
                print(f"[INFO] Updated capture region: {capture_region['width']}x{capture_region['height']}")

        # ✅ Gestion des touches optimisée
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("[INFO] ESC pressed, exiting.")
            break
            
        # ✅ Vérifier numpad moins souvent pour éviter le lag
        if frame_id % 5 == 0 and keyboard.is_pressed("num 1"):
            print("[INFO] Numpad 1 pressed, exiting.")
            break
            
        # ✅ Limitation de FPS pour éviter la surcharge CPU
        elapsed = time.perf_counter() - start_time
        if elapsed < frame_time_target:
            time.sleep(frame_time_target - elapsed)

    cv2.destroyAllWindows()
    # ✅ Forcer la fermeture de toutes les fenêtres OpenCV
    for i in range(4):  # Répéter plusieurs fois pour s'assurer
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    print("[INFO] Stream stopped cleanly.")

if __name__ == "__main__":
    stream_screen()
