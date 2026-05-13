import cv2
from ultralytics import YOLO
import csv

print("Chargement du modèle...")
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture("auto_ecole_test.mov")

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

# Création (ou ouverture) du fichier CSV
with open('rapport_conduite.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'Objet', 'Confiance', 'Position_X'])

    frame_count = 0
    print("Début de l'analyse. Appuie sur 'q' pour quitter.")
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("Fin de la vidéo.")
            break
        
        frame_count += 1
        
        # On passe l'image au modèle (sans stream=True pour éviter le bug du générateur sur une seule frame)
        results = model(frame)

        # On parcourt les résultats
        for r in results:
            # 1. On crée l'image avec les boîtes dessinées dessus directement ici
            annotated_frame = r.plot()
            
            # 2. On extrait les données pour notre fichier CSV
            for box in r.boxes:
                cls = int(box.cls[0])
                label = r.names[cls]
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]

                # ---------------------------------------------------------
                # DÉBUT DE TES RÈGLES D'AUTO-ÉCOLE
                # ---------------------------------------------------------

                # RÈGLE 1 : Le Panneau Stop
                if label == "stop sign" and conf > 0.8:
                    writer.writerow([frame_count, "Signalisation", label, conf, coords[0]])
                    print(f"🛑 Panneau STOP détecté à l'image {frame_count}")

                # RÈGLE 2 : Le Vélo (Distance de sécurité)
                elif label == "bicycle" and conf > 0.60:
                    # On met une confiance un peu plus basse (60%) car les vélos sont 
                    # parfois plus durs à détecter de loin.
                    writer.writerow([frame_count, "Usager Vulnérable", label, conf, coords[0]])
                    print(f"🚲 Vélo repéré à l'image {frame_count} - Attention à l'écart !")

                # RÈGLE 3 : Le Piéton
                elif label == "person" and conf > 0.70:
                    writer.writerow([frame_count, "Usager Vulnérable", label, conf, coords[0]])
                    print(f"🚶 Piéton repéré à l'image {frame_count}")
                
                # ---------------------------------------------------------
                # FIN DE TES RÈGLES
                # ---------------------------------------------------------

            # 3. On affiche l'image annotée (le bug de la ligne 36 est corrigé)
            cv2.imshow("Analyse", annotated_frame)
            
        # Touche 'q' pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

cap.release()
cv2.destroyAllWindows()
print("Analyse terminée. Vérifie le fichier rapport_conduite.csv !")