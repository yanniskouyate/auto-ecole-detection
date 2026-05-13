import cv2
from ultralytics import YOLO
import os

# 1. Charger le modèle
model = YOLO("yolo11n.pt") 

# 2. Lancer la prédiction et SAUVEGARDER le résultat
# - save=True : crée une vidéo avec les boîtes dessinées
# - project='runs' : définit le dossier de sortie
# - name='test_docker' : nom du sous-dossier
results = model.predict(source="auto_ecole_test.mov", save=True, project="runs", name="test_docker", exist_ok=True)

print("--- VÉRIFICATION TERMINÉE ---")
print(f"La vidéo annotée a été enregistrée dans le dossier : runs/test_docker/")