from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple


def filter_pose_persons(results, min_keypoints: int = 6, kp_conf_threshold: float = 0.3) -> List[Tuple[int, int, int, int, float]]:
    """
    Filter YOLOv8-pose detections to keep only full humans with at least `min_keypoints` visible.

    :param results: YOLO pose results
    :param min_keypoints: Minimum number of valid keypoints required to keep detection
    :param kp_conf_threshold: Minimum confidence for each keypoint
    :return: List of filtered bounding boxes (x1, y1, x2, y2, conf)
    """
    filtered_boxes = []

    for box, keypoints in zip(results.boxes, results.keypoints.data):
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # Keep only "person" class (cls == 0)
        if cls != 0:
            continue

        # Count valid keypoints
        keypoints_np = keypoints.cpu().numpy().reshape(-1, 3)  # (x, y, conf)
        valid_kps = np.sum(keypoints_np[:, 2] > kp_conf_threshold)

        if valid_kps >= min_keypoints:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            filtered_boxes.append((x1, y1, x2, y2, conf))

    return filtered_boxes


def run_yolo_pose_filtered(image_path: str) -> None:
    """
    Run YOLOv8-pose on an image and show only detections with enough keypoints.

    :param image_path: Path to the image
    """
    model = YOLO("yolov8n-pose.pt")
    results = model(image_path, conf=0.25, iou=0.45)

    # Load image manually for custom annotation
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    boxes = filter_pose_persons(results[0], min_keypoints=6, kp_conf_threshold=0.3)

    # Draw filtered bounding boxes
    for (x1, y1, x2, y2, conf) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"person {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Display the annotated image
    cv2.namedWindow("YOLOv8 Pose — Filtered", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLOv8 Pose — Filtered", image)

    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC pour fermer
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_yolo_pose_filtered("test4.png")
