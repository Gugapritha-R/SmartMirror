import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, min_face_size_ratio=0.3, brightness_threshold=50):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=min_detection_confidence)
        self.min_face_size_ratio = min_face_size_ratio
        self.brightness_threshold = brightness_threshold

    def detect_face(self, frame):
        ih, iw, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        if results.detections:
            for det in results.detections:
                bboxC = det.location_data.relative_bounding_box
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w_box = int(bboxC.width * iw)
                h_box = int(bboxC.height * ih)

                # Clamp bounding box within image bounds
                x = max(0, x)
                y = max(0, y)
                x2 = min(iw, x + w_box)
                y2 = min(ih, y + h_box)

                face = frame[y:y2, x:x2]

                # Check if face is large enough
                if w_box < self.min_face_size_ratio * iw or h_box < self.min_face_size_ratio * ih:
                    print("⚠️ Face too small. Ask user to come closer.")
                    return None, None

                # Lighting check
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                brightness = gray_face.mean()
                if brightness < self.brightness_threshold:
                    print("💡 Low lighting. Ask user to increase lighting.")
                    return None, None

                return face, (x, y, w_box, h_box)

        return None, None
