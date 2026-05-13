# Utilise une image Python optimisée pour l'IA
FROM python:3.10-slim

# Installer les dépendances système nécessaires pour OpenCV (traitement vidéo)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail dans le conteneur
WORKDIR /app

# Copier uniquement le fichier des dépendances pour optimiser le cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# On ne copie pas le code ici pour le mode "développement" (on utilisera un volume)
# Mais on prépare la commande de lancement
CMD ["python", "main.py"]