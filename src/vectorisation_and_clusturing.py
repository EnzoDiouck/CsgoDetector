import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import glob
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel

class ImageVectorizer:
    def __init__(self):
        """
        Initialise le vectoriseur avec facebook/dinov2-small pré-entraîné
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilisation du device: {self.device}")

        # Charger facebook/dinov2-small pré-entraîné
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.model = AutoModel.from_pretrained("facebook/dinov2-small")
        self.model.eval()
        self.model.to(self.device)
    
    def extract_features(self, image_path):
        """
        Extrait les features d'une image avec facebook/dinov2-small
        """
        try:
            # Charger et prétraiter l'image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # Extraire les features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Utiliser le token CLS (premier token) comme représentation globale
                image_features = outputs.last_hidden_state[:, 0, :]

            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Erreur lors de l'extraction des features pour {image_path}: {e}")
            return None
    
    def vectorize_images(self, image_folder, max_images=None):
        """
        Vectorise les images au format {number}_{index}.jpg d'un dossier, avec une limite optionnelle.
        """
        # Chercher uniquement les images au format number_index.jpg (ex: 0_0.jpg, 1_2.jpg)
        import re
        all_images = glob.glob(os.path.join(image_folder, "*.jpg"))
        image_paths = []
        for img_path in all_images:
            filename = os.path.basename(img_path)
            # Vérifier si le nom correspond au pattern number_number.jpg
            if re.match(r'^\d+_\d+\.jpg$', filename):
                image_paths.append(img_path)
        
        print(f"Images trouvées au format number_index.jpg: {len(image_paths)}")
        
        # Si max_images est défini, limiter le nombre d'images
        if max_images is not None:
            image_paths = image_paths[:max_images]
        
        features_list = []
        valid_paths = []
        
        print(f"Traitement de {len(image_paths)} images...")
        
        for i, img_path in enumerate(image_paths):
            if i % 50 == 0:
                print(f"Progression: {i}/{len(image_paths)}")
            
            features = self.extract_features(img_path)
            if features is not None:
                features_list.append(features)
                valid_paths.append(img_path)
        
        print(f"Vectorisation terminée: {len(valid_paths)} images traitées")
        return np.array(features_list), valid_paths

class ImageClusterer:
    def __init__(self):
        """
        Initialise le système de clustering
        """
        pass
    
    def get_true_labels(self, image_paths):
        """
        Détermine les vrais labels basés sur les fichiers d'annotation
        """
        true_labels = []
        image_names = []
        base_numbers = []
        
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            image_names.append(img_name)
            
            # Extraire le numéro de base de l'image (ex: pour 0_1.jpg -> 0)
            base_num = img_name.split('_')[0]
            base_numbers.append(base_num)
            
            # Chercher le fichier d'annotation correspondant
            annotation_file = f"./bounding_box/{base_num}.txt"
            
            # Déterminer le label basé sur l'existence du fichier
            if os.path.exists(annotation_file):
                true_labels.append(f"Image_{base_num}")
            else:
                true_labels.append("Inconnu")
        
        return true_labels, image_names, base_numbers
    
    def normalize_features(self, features):
        """
        Normalise les features pour améliorer le clustering
        """
        # Aplatir les caractéristiques si elles sont multidimensionnelles
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Standardisation (moyenne=0, écart-type=1)
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        return normalized_features, scaler
    
    def evaluate_clustering(self, features, labels):
        """
        Évalue la qualité du clustering
        """
        try:
            silhouette_avg = silhouette_score(features, labels)
            calinski_score = calinski_harabasz_score(features, labels)
            return silhouette_avg, calinski_score
        except:
            return -1, -1
    
    def find_optimal_clusters(self, features, max_clusters=8):
        """
        Trouve le nombre optimal de clusters
        """
        silhouette_scores = []
        calinski_scores = []
        k_range = range(2, min(max_clusters + 1, len(features) // 2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            sil_score, cal_score = self.evaluate_clustering(features, labels)
            silhouette_scores.append(sil_score)
            calinski_scores.append(cal_score)
        
        # Choisir le k avec le meilleur score silhouette
        if silhouette_scores:
            optimal_k = k_range[np.argmax(silhouette_scores)]
            return optimal_k
        return 2
    
    def perform_clustering(self, features, n_clusters=None, algorithm='auto'):
        """
        Effectue le clustering avec plusieurs algorithmes et optimisation automatique
        """
        # Normaliser les features
        normalized_features, scaler = self.normalize_features(features)
        
        # Déterminer le nombre optimal de clusters si non spécifié
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(normalized_features)
            print(f"Nombre optimal de clusters détecté: {n_clusters}")
        
        best_score = -1
        best_labels = None
        best_model = None
        best_algorithm = None
        
        algorithms_to_try = ['kmeans', 'gaussian_mixture', 'agglomerative'] if algorithm == 'auto' else [algorithm]
        
        for alg in algorithms_to_try:
            try:
                if alg == 'kmeans':
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
                    labels = model.fit_predict(normalized_features)
                    
                elif alg == 'gaussian_mixture':
                    model = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
                    model.fit(normalized_features)
                    labels = model.predict(normalized_features)
                    
                elif alg == 'agglomerative':
                    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                    labels = model.fit_predict(normalized_features)
                    
                elif alg == 'dbscan':
                    # Estimer eps automatiquement
                    neighbors = NearestNeighbors(n_neighbors=4)
                    neighbors_fit = neighbors.fit(normalized_features)
                    distances, indices = neighbors_fit.kneighbors(normalized_features)
                    distances = np.sort(distances[:, 3], axis=0)
                    eps = distances[int(len(distances) * 0.9)]  # 90ème percentile
                    
                    model = DBSCAN(eps=eps, min_samples=4)
                    labels = model.fit_predict(normalized_features)
                    
                    # Vérifier si DBSCAN a trouvé des clusters valides
                    unique_labels = np.unique(labels)
                    if len(unique_labels) < 2 or -1 in unique_labels:
                        continue
                
                # Évaluer la qualité du clustering
                sil_score, cal_score = self.evaluate_clustering(normalized_features, labels)
                
                print(f"Algorithme: {alg}, Silhouette: {sil_score:.3f}, Calinski-Harabasz: {cal_score:.1f}")
                
                if sil_score > best_score:
                    best_score = sil_score
                    best_labels = labels
                    best_model = model
                    best_algorithm = alg
                    
            except Exception as e:
                print(f"Erreur avec l'algorithme {alg}: {e}")
                continue
        
        if best_labels is None:
            # Fallback vers K-means simple
            print("Fallback vers K-means standard")
            model = KMeans(n_clusters=n_clusters, random_state=42)
            best_labels = model.fit_predict(normalized_features)
            best_model = model
            best_algorithm = 'kmeans_fallback'
        
        print(f"\nMeilleur algorithme: {best_algorithm} (Score Silhouette: {best_score:.3f})")
        print(f"Clustering terminé avec {len(np.unique(best_labels))} clusters")
        
        return best_labels, best_model, scaler, normalized_features
    
    def reduce_dimensions(self, features, method='tsne', n_components=2):
        """
        Réduit les dimensions des features pour la visualisation.
        """
        # Aplatir les caractéristiques si elles sont en 4D
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)

        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Méthode de réduction de dimensions inconnue: {method}")

        coords_2d = reducer.fit_transform(features)
        return coords_2d
    
    def create_interactive_plot(self, coords_2d, cluster_labels, true_labels, image_names):
        """
        Crée un graphique interactif avec Plotly
        """
        # Créer un DataFrame pour faciliter la manipulation
        df = pd.DataFrame({
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'cluster': cluster_labels,
            'true_label': true_labels,
            'image_name': image_names
        })
        
        # Créer le graphique interactif
        fig = go.Figure()
        
        # Ajouter les points pour chaque cluster
        colors = ['red', 'blue', 'green', 'orange']
        for cluster_id in np.unique(cluster_labels):
            cluster_data = df[df['cluster'] == cluster_id]
            
            fig.add_trace(go.Scatter(
                x=cluster_data['x'],
                y=cluster_data['y'],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    size=8,
                    color=colors[cluster_id % len(colors)],
                    opacity=0.7
                ),
                text=cluster_data['image_name'],
                customdata=cluster_data['true_label'],
                hovertemplate='<b>Image:</b> %{text}<br>' +
                              '<b>Vrai label:</b> %{customdata}<br>' +
                              '<b>Cluster:</b> %{fullData.name}<br>' +
                              '<extra></extra>'
            ))
        
        fig.update_layout(
            title="Clustering des Images - Anti-terroristes vs Terroristes",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            hovermode='closest',
            width=1000,
            height=700
        )
        
        return fig, df
    
    def analyze_clustering_results(self, cluster_labels, true_labels):
        """
        Analyse les résultats du clustering
        """
        print("\n=== Analyse des résultats de clustering ===")
        
        # Créer un DataFrame pour l'analyse
        df = pd.DataFrame({
            'cluster': cluster_labels,
            'true_label': true_labels
        })
        
        # Compter les éléments par cluster et vrai label
        crosstab = pd.crosstab(df['cluster'], df['true_label'])
        print("\nTableau de contingence (Cluster vs Vrai Label):")
        print(crosstab)
        
        # Calculer la pureté de chaque cluster
        print("\nAnalyse par cluster:")
        for cluster_id in np.unique(cluster_labels):
            cluster_data = df[df['cluster'] == cluster_id]
            total_cluster = len(cluster_data)
            
            print(f"\nCluster {cluster_id} ({total_cluster} images):")
            label_counts = cluster_data['true_label'].value_counts()
            for label, count in label_counts.items():
                percentage = (count / total_cluster) * 100
                print(f"  - {label}: {count} ({percentage:.1f}%)")
        
        return crosstab
    
    def update_txt_files(self, image_paths, cluster_labels, base_numbers):
        """
        Met à jour les fichiers .txt existants en remplaçant le label (premier chiffre) 
        par le résultat du clustering.
        
        Args:
            image_paths (list): Liste des chemins des images.
            cluster_labels (list): Liste des labels de cluster pour chaque image.
            base_numbers (list): Liste des numéros de base des images.
        """
        # Grouper les résultats par numéro de base
        base_to_cluster = {}
        for base_num, cluster_label in zip(base_numbers, cluster_labels):
            if base_num not in base_to_cluster:
                base_to_cluster[base_num] = []
            base_to_cluster[base_num].append(cluster_label)
        
        # Pour chaque fichier txt, déterminer le label majoritaire
        for base_num, cluster_list in base_to_cluster.items():
            txt_file_path = f"./bounding_box/{base_num}.txt"
            
            if os.path.exists(txt_file_path):
                # Déterminer le label majoritaire pour cette image
                from collections import Counter
                most_common_cluster = Counter(cluster_list).most_common(1)[0][0]
                
                # Lire le contenu existant
                with open(txt_file_path, 'r') as f:
                    lines = f.readlines()
                
                # Modifier les labels (premier chiffre de chaque ligne)
                updated_lines = []
                for line in lines:
                    if line.strip():  # Ignorer les lignes vides
                        parts = line.strip().split()
                        if len(parts) >= 5:  # Format YOLO: label x y w h
                            # Remplacer le label par le résultat du clustering
                            parts[0] = str(most_common_cluster)
                            updated_lines.append(' '.join(parts) + '\n')
                        else:
                            updated_lines.append(line)  # Garder la ligne telle quelle si format incorrect
                
                # Écrire le contenu mis à jour
                with open(txt_file_path, 'w') as f:
                    f.writelines(updated_lines)
                
                print(f"Fichier {txt_file_path} mis à jour avec le label {most_common_cluster}")
            else:
                print(f"Fichier {txt_file_path} non trouvé")
        
        print(f"Mise à jour terminée pour {len(base_to_cluster)} fichiers .txt")

def main():
    """
    Fonction principale
    """
    print("=== Démarrage de la vectorisation et du clustering ===")
    
    # Chemins
    bounding_box_folder = "./bounding_box"
    
    # Vérifier que le dossier existe
    if not os.path.exists(bounding_box_folder):
        print(f"Erreur: Le dossier {bounding_box_folder} n'existe pas!")
        return
    
    # 1. Vectorisation des images
    print("\n1. Vectorisation des images avec ResNet-50...")
    vectorizer = ImageVectorizer()
    features, image_paths = vectorizer.vectorize_images(bounding_box_folder, max_images=None)
    
    if len(features) == 0:
        print("Aucune image n'a pu être traitée!")
        return
    
    print(f"Features extraites: {features.shape}")
    
    # 2. Clustering
    print("\n2. Clustering des images...")
    clusterer = ImageClusterer()
    
    # Obtenir les vrais labels et numéros de base
    true_labels, image_names, base_numbers = clusterer.get_true_labels(image_paths)
    
    # Effectuer le clustering avec optimisation automatique
    cluster_labels, best_model, scaler, normalized_features = clusterer.perform_clustering(features, n_clusters=None, algorithm='auto')
    
    # 3. Réduction de dimensions pour la visualisation
    print("\n3. Préparation de la visualisation...")
    coords_2d = clusterer.reduce_dimensions(normalized_features, method='tsne')
    
    # 4. Créer la visualisation interactive
    print("\n4. Création de la visualisation interactive...")
    fig, df = clusterer.create_interactive_plot(coords_2d, cluster_labels, true_labels, image_names)
    
    # 5. Analyser les résultats
    crosstab = clusterer.analyze_clustering_results(cluster_labels, true_labels)
    
    # 6. Afficher le graphique
    print("\n5. Affichage du graphique interactif...")
    fig.show()
    
    # Sauvegarder le graphique
    output_path = os.path.join(os.getcwd(), "clustering_results.html")
    
    # Vérifier et créer le dossier si nécessaire avant d'écrire le fichier HTML
    output_folder = os.path.dirname(output_path)
    os.makedirs(output_folder, exist_ok=True)

    # Écrire le fichier HTML
    fig.write_html(output_path)
    print(f"\nGraphique sauvegardé: {output_path}")
    
    # Sauvegarder les résultats dans un CSV
    results_path = "./clustering_results.csv"
    df.to_csv(results_path, index=False)
    print(f"Résultats sauvegardés: {results_path}")
    
    # Afficher les résultats du clustering avant la mise à jour
    print("\n6. Résultats du clustering:")
    print("=" * 50)
    
    # Analyser la distribution des clusters
    from collections import Counter, defaultdict
    cluster_count = Counter(cluster_labels)
    cluster_images = defaultdict(list)
    
    for i, (img_path, cluster_label, base_num) in enumerate(zip(image_paths, cluster_labels, base_numbers)):
        img_name = os.path.basename(img_path)
        cluster_images[cluster_label].append((img_name, base_num))
    
    for cluster_id in sorted(cluster_count.keys()):
        count = cluster_count[cluster_id]
        print(f"\nCluster {cluster_id}: {count} images")
        print(f"Exemples d'images:")
        
        # Afficher quelques exemples
        examples = cluster_images[cluster_id][:5]
        for img_name, base_num in examples:
            print(f"  - {img_name} (base: {base_num})")
        
        if len(cluster_images[cluster_id]) > 5:
            print(f"  ... et {len(cluster_images[cluster_id]) - 5} autres")
    
    print("\n" + "=" * 50)
    print("Ces labels vont remplacer les labels existants dans les fichiers .txt")
    
    # Demander confirmation
    response = input("\nVoulez-vous procéder à la mise à jour des fichiers .txt ? (o/n): ")
    
    if response.lower() in ['o', 'oui', 'y', 'yes']:
        # Mettre à jour les fichiers .txt existants avec les résultats du clustering
        print("\n7. Mise à jour des fichiers .txt avec les résultats du clustering...")
        clusterer.update_txt_files(image_paths, cluster_labels, base_numbers)
    else:
        print("\nMise à jour annulée. Les fichiers .txt n'ont pas été modifiés.")
    
    print("\n=== Analyse terminée ===")
    print("Les fichiers .txt ont été mis à jour avec les nouveaux labels de clustering!")
    print("Passez la souris sur les points pour voir les noms des images!")

if __name__ == "__main__":
    main()