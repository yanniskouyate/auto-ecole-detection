import cv2
from ultralytics import YOLO

# 1. Charger le modèle YOLOv11 (il se téléchargera tout seul au premier lancement)
model = YOLO("yolo11n.pt") 

# 2. Ouvrir la vidéo de la dashcam
cap = cv2.VideoCapture("auto_ecole_test.mov")
# 3. Traiter la vidéo frame par frame
while True:     
    ret, frame = cap.read()     
    if not ret:         
        break  # Fin de la vidéo

    # 4. Appliquer le modèle YOLOv11 sur la frame
    results = model(frame)

    # 5. Afficher les résultats (bounding boxes, labels, etc.)
    annotated_frame = results[0].plot()  # Dessiner les annotations sur la frame

    # 6. Afficher la frame annotée
    cv2.imshow("YOLOv11 Detection", annotated_frame)

    # 7. Quitter si l'utilisateur appuie sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 8. Libérer les ressources
cap.release()
cv2.destroyAllWindows() 