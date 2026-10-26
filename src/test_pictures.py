from ultralytics import YOLO
import cv2

# Charger ton modèle YOLO entraîné
model = YOLO("runs/detect/train_v2/weights/best.pt")

# Lancer la détection
results = model("./bounding_box/9_original.jpg", conf=0.10, iou=0.45)

# Récupérer l'image annotée et la personnaliser
annotated = results[0].plot(line_width=1, font_size=0.4)  # 👈 paramètre clé ici !

# Afficher l'image
cv2.namedWindow("Résultat YOLOv8", cv2.WINDOW_NORMAL)
cv2.imshow("Résultat YOLOv8", annotated)

# Attendre fermeture manuelle (ESC ou croix)
while True:
    if cv2.waitKey(1) & 0xFF == 27:  # ESC pour fermer
        break

cv2.destroyAllWindows()