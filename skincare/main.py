from utils.face_detection import FaceDetector
import cv2
from utils.skintype_detect import SkinTypeClassifier
from utils.acne_detect import SkinIssueDetector

yolo_detector = SkinIssueDetector(r'C:\Users\Admin\Desktop\smart mirror\models\acne_detection\runs\content\runs\detect\train\weights\best.pt')

skin_classifier = SkinTypeClassifier(r'C:\Users\Admin\Desktop\smart mirror\models\skin_classification_model.keras')


cap = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face, bbox = detector.detect_face(frame)

    if face is not None:
       skin_type, conf = skin_classifier.predict(face)
       issues = yolo_detector.detect(face)
       print(f"Skin Type: {skin_type} ({conf*100:.2f}%)")
       print(f"Issues Detected: {issues}")
      

    cv2.imshow("Skincare Assistant", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
