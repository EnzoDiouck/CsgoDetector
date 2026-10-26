#!/usr/bin/env python3
"""
Programme pour organiser les images et labels YOLO en structure train/valid/test
avec une répartition de 70% train, 20% valid, 10% test.

Usage:
    python create_and_split_ds.py --source_dir path/to/source --output_dir path/to/output
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple
import yaml

# Variables globales pour les dossiers d'images, de labels et de sortie
IMAGE_FOLDER = "./bounding_box"  # Dossier contenant les images numérotées (0.jpg, 1.jpg, etc.)
LABEL_FOLDER = "./bounding_box"  # Dossier contenant les labels numérotés (0.txt, 1.txt, etc.)
OUTPUT_FOLDER = "./output"  # Dossier de sortie pour la structure train/valid/test


def create_directory_structure(output_dir: str) -> None:
    """Crée la structure de dossiers train/valid/test avec images et labels."""
    base_path = Path(output_dir)
    
    # Créer les dossiers principaux
    for split in ['train', 'valid', 'test']:
        for subfolder in ['images', 'labels']:
            folder_path = base_path / split / subfolder
            folder_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Structure de dossiers créée dans: {output_dir}")


def get_image_label_pairs(source_dir: str) -> List[Tuple[str, str]]:
    """
    Trouve tous les pairs image-label avec des noms numériques (0.jpg/0.txt, 1.jpg/1.txt, etc.).
    Supporte les formats d'images: .jpg, .jpeg, .png, .bmp
    """
    # Utiliser les variables globales si définies
    image_folder_path = Path(IMAGE_FOLDER) if IMAGE_FOLDER else Path(source_dir)
    label_folder_path = Path(LABEL_FOLDER) if LABEL_FOLDER else Path(source_dir)

    pairs = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    # Chercher d'abord tous les fichiers txt avec des noms numériques
    for label_file in label_folder_path.glob('*.txt'):
        # Vérifier que le nom du fichier est purement numérique
        stem = label_file.stem
        if stem.isdigit():
            # Chercher l'image correspondante avec différentes extensions
            for ext in image_extensions:
                image_file = image_folder_path / f"{stem}{ext}"
                if image_file.exists():
                    pairs.append((str(image_file), str(label_file)))
                    break
            else:
                print(f"Attention: Pas d'image trouvée pour {label_file}")

    return pairs


def split_dataset(pairs: List[Tuple[str, str]], train_ratio: float = 0.7, 
                  valid_ratio: float = 0.2, test_ratio: float = 0.1) -> Tuple[List, List, List]:
    """
    Divise les pairs image-label selon les ratios spécifiés.
    """
    # Vérifier que les ratios sont corrects
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Les ratios doivent sommer à 1.0, obtenu: {total_ratio}")
    
    # Mélanger les pairs pour randomiser
    random.shuffle(pairs)
    
    total_pairs = len(pairs)
    train_count = int(total_pairs * train_ratio)
    valid_count = int(total_pairs * valid_ratio)
    
    # Diviser les données
    train_pairs = pairs[:train_count]
    valid_pairs = pairs[train_count:train_count + valid_count]
    test_pairs = pairs[train_count + valid_count:]
    
    print(f"Répartition des données:")
    print(f"  Train: {len(train_pairs)} images ({len(train_pairs)/total_pairs*100:.1f}%)")
    print(f"  Valid: {len(valid_pairs)} images ({len(valid_pairs)/total_pairs*100:.1f}%)")
    print(f"  Test:  {len(test_pairs)} images ({len(test_pairs)/total_pairs*100:.1f}%)")
    
    return train_pairs, valid_pairs, test_pairs


def copy_files(pairs: List[Tuple[str, str]], output_dir: str, split_name: str) -> None:
    """
    Copie les fichiers image et label vers le dossier de destination.
    """
    output_path = Path(output_dir)
    images_dir = output_path / split_name / 'images'
    labels_dir = output_path / split_name / 'labels'
    
    for image_path, label_path in pairs:
        image_file = Path(image_path)
        label_file = Path(label_path)
        
        # Copier l'image
        dest_image = images_dir / image_file.name
        shutil.copy2(image_path, dest_image)
        
        # Copier le label
        dest_label = labels_dir / label_file.name
        shutil.copy2(label_path, dest_label)
    
    print(f"Copié {len(pairs)} files vers {split_name}/")


def create_data_yaml(output_dir: str, class_names: List[str] = None) -> None:
    """
    Crée le fichier data.yaml pour YOLO.
    """
    if class_names is None:
        class_names = ['anti', 'terro']  # Classes par défaut
    
    output_path = Path(output_dir)
    
    data_config = {
        'names': class_names,
        'nc': len(class_names),
        'test': 'test\\images',
        'train': 'train\\images',
        'val': 'valid\\images'
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Fichier data.yaml créé: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Organise les images et labels en structure YOLO")
    parser.add_argument('--source_dir', '-s', required=False, 
                        help='Dossier source contenant les images et labels numériques',
                        default=IMAGE_FOLDER)
    parser.add_argument('--output_dir', '-o', required=False,
                        help='Dossier de sortie pour la structure train/valid/test',
                        default=OUTPUT_FOLDER)
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Pourcentage pour train (défaut: 0.7)')
    parser.add_argument('--valid_ratio', type=float, default=0.2,
                        help='Pourcentage pour valid (défaut: 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Pourcentage pour test (défaut: 0.1)')
    parser.add_argument('--classes', nargs='+', default=['terro', 'anti'],
                        help='Noms des classes (défaut: terro anti)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Graine pour la randomisation (défaut: 42)')

    args = parser.parse_args()

    # Fixer la graine pour la reproductibilité
    random.seed(args.seed)

    print(f"=== Organisation du dataset YOLO ===")
    print(f"Source: {args.source_dir}")
    print(f"Sortie: {args.output_dir}")
    print(f"Ratios: Train={args.train_ratio}, Valid={args.valid_ratio}, Test={args.test_ratio}")
    print()

    # Vérifier que le dossier source existe
    if not os.path.exists(args.source_dir):
        print(f"Erreur: Le dossier source '{args.source_dir}' n'existe pas!")
        return

    # Créer la structure de dossiers
    create_directory_structure(args.output_dir)

    # Trouver les pairs image-label
    print("Recherche des images et labels...")
    pairs = get_image_label_pairs(args.source_dir)

    if not pairs:
        print("Aucune paire image-label trouvée!")
        return

    print(f"Trouvé {len(pairs)} paires image-label")

    # Diviser le dataset
    train_pairs, valid_pairs, test_pairs = split_dataset(
        pairs, args.train_ratio, args.valid_ratio, args.test_ratio
    )

    # Copier les fichiers
    print("\nCopie des fichiers...")
    copy_files(train_pairs, args.output_dir, 'train')
    copy_files(valid_pairs, args.output_dir, 'valid')
    copy_files(test_pairs, args.output_dir, 'test')

    # Créer le fichier data.yaml
    create_data_yaml(args.output_dir, args.classes)

    print("\n=== Terminé! ===")
    print(f"Dataset organisé dans: {args.output_dir}")


if __name__ == "__main__":
    main()
