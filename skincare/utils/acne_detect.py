from ultralytics import YOLO
import cv2

class SkinIssueDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, face_img):
        results = self.model.predict(face_img, imgsz=640, conf=0.3, verbose=False)
        detections = results[0].boxes

        issues = []
        for det in detections:
            cls_id = int(det.cls)
            conf = float(det.conf)
            label = self.model.names[cls_id]
            issues.append((label, conf))

        return issues
