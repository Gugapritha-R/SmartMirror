import time
import cv2
from collections import Counter
from utils.face_detection import FaceDetector
from utils.skintype_detect import SkinTypeClassifier
from utils.acne_detect import SkinIssueDetector

# Load models
yolo_detector = SkinIssueDetector(r'C:\Users\Admin\Desktop\smart mirror\models\acne_detection\runs\content\runs\detect\train\weights\best.pt')
skin_classifier = SkinTypeClassifier(r'C:\Users\Admin\Desktop\smart mirror\models\skin_classification_model.keras')
cap = cv2.VideoCapture(0)
detector = FaceDetector()

# Prediction stabilization
prediction_buffer = []
buffer_size = 10
start_time = None
finalized = False

# Quality check buffer
clear_face_frames = 0
required_clear_frames = 15  # wait for 15 good frames (~1 sec+)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face, bbox = detector.detect_face(frame)

    if face is not None and not finalized:
        clear_face_frames += 1

        
        if clear_face_frames >= required_clear_frames:
            skin_type, conf = skin_classifier.predict(face)

            # Only accept high-confidence predictions
            if conf > 0.55:
                prediction_buffer.append(skin_type)
                if len(prediction_buffer) > buffer_size:
                    prediction_buffer.pop(0)

                most_common = Counter(prediction_buffer).most_common(1)[0]
                if most_common[1] >= (buffer_size * 0.7):
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time > 2:  # stable for 2 seconds
                        stable_skin_type = most_common[0]
                        issues = yolo_detector.detect(face)
                        finalized = True
                        print("✅ Final Results:")
                        print(f"Skin Type: {stable_skin_type}")
                        print(f"Skin Issues: {issues}")
            else:
                print("⚠️ Low confidence, skipping this frame.")

    else:
        clear_face_frames = 0
        start_time = None
        prediction_buffer.clear()

    # Visual feedback
    if face is not None and bbox:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if finalized:
        cv2.putText(frame, f"Skin Type: {stable_skin_type}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for i, issue in enumerate(issues):
            cv2.putText(frame, f"Issue: {issue}", (30, 60 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

    cv2.imshow("Skincare Assistant", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('r'):
        finalized = False
        prediction_buffer.clear()
        start_time = None
        clear_face_frames = 0

cap.release()
cv2.destroyAllWindows()
