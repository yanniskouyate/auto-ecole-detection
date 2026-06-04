# 🚗 Auto-École Vision : Système d'Analyse Intelligente de Conduite

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11n-green.svg)](https://github.com/ultralytics/ultralytics)
[![Docker](https://img.shields.io/badge/Docker-Compatible-blue.svg)](https://www.docker.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)](https://opencv.org/)

**Auto-École Vision** est un outil d'intelligence artificielle basé sur **YOLOv11** et **OpenCV** conçu pour analyser les séquences vidéo de dashcam lors des séances de conduite. Son but est d'assister les moniteurs d'auto-école en détectant automatiquement les événements clés de la route, les usagers vulnérables et le respect de la signalisation routière afin de générer un rapport de conduite objectif.

---

## 🎯 Objectifs du Projet

L'évaluation d'un élève conducteur repose souvent sur la vigilance constante du moniteur. Ce projet propose un outil complémentaire pour débriefer une leçon de conduite à l'aide de la vidéo :
- **Sécurité routière** : Détection en temps réel des usagers vulnérables (piétons, cyclistes).
- **Respect du code** : Identification de la signalisation critique (panneaux STOP).
- **Rapport automatisé** : Génération d'un fichier d'analyse temporelle (`.csv`) pour identifier les moments forts de la leçon.

---

## 🏗️ Architecture du Projet

Le dépôt est structuré comme suit :

```bash
├── README.md               # Le document que vous lisez actuellement
├── Dockerfile              # Configuration pour conteneuriser l'environnement de production/dev
├── requirements.txt        # Liste des dépendances Python requises
├── main.py                 # Script automatisé (batch) idéal pour Docker (sans affichage direct)
├── detect_auto.py          # Script de démonstration de détection en temps réel (GUI OpenCV)
├── analyse_video.py        # Script métier principal (filtrage, alertes et génération de rapport)
├── auto_ecole_test.mov     # Vidéo source de test (dashcam)
├── yolo11n.pt              # Poids pré-entraînés du modèle YOLOv11 nano (téléchargés automatiquement)
├── rapport_conduite.csv    # Rapport d'analyse produit par le script principal
└── runs/                   # Dossier contenant les enregistrements vidéos annotés par YOLO
    └── test_docker/        # Sortie générée par main.py
```

---

## 🛠️ Installation et Configuration

### Méthode 1 : Installation Locale (Recommandé pour le développement avec GUI)

#### 1. Prérequis
- Python 3.10 (ou supérieur) installé.
- Une webcam ou une vidéo de test (`auto_ecole_test.mov` fournie dans le répertoire).

#### 2. Configuration de l'environnement virtuel
Il est fortement recommandé d'isoler vos dépendances dans un environnement virtuel :

```bash
# Création de l'environnement virtuel
python3 -m venv auto_ecole_env

# Activation de l'environnement virtuel
# Sur macOS/Linux :
source auto_ecole_env/bin/activate
# Sur Windows :
auto_ecole_env\Scripts\activate
```

#### 3. Installation des dépendances
Installez les packages listés dans `requirements.txt` :

```bash
pip install -r requirements.txt
```

> **Note de développement** : Les scripts `detect_auto.py` et `analyse_video.py` utilisent un affichage graphique (`cv2.imshow()`). Assurez-vous d'utiliser la version standard d'OpenCV si vous souhaitez l'interface visuelle en local : `pip install opencv-python`. Si vous tournez uniquement en mode sans affichage ou Docker, `opencv-python-headless` (déjà présent dans le requirements.txt) est suffisant.

---

## 🚀 Utilisation des Scripts

Le projet propose trois niveaux d'utilisation :

### 1. Inférence standard automatisée (`main.py`)
Ce script traite la vidéo en tâche de fond et exporte une copie annotée complète dans le dossier `runs/test_docker/`. Idéal pour un traitement non-interactif et pour s'intégrer dans Docker.

```bash
python main.py
```

### 2. Visualisation en temps réel (`detect_auto.py`)
Ce script ouvre la vidéo `auto_ecole_test.mov` et affiche l'ensemble des détections du modèle YOLOv11 en temps réel à l'écran.

```bash
python detect_auto.py
```
* **Contrôle** : Appuyez sur la touche **`q`** pour fermer la fenêtre de rendu.

### 3. Analyse Métier Auto-École & Génération de Rapport (`analyse_video.py`)
Il s'agit du cœur applicatif. Ce script applique des filtres stricts basés sur les règles de conduite et enregistre les anomalies ou points d'attention dans un fichier CSV.

```bash
python analyse_video.py
```

---

## 🧠 Logique métier & Règles d'analyse

Dans `analyse_video.py`, nous n'affichons pas simplement toutes les détections (voitures, arbres, etc.). Nous ciblons les éléments critiques d'apprentissage avec des seuils de confiance rigoureux :

| Règle | Élément Ciblé | Seuil de Confiance | Catégorie Rapport | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Règle 1** | 🛑 Panneau STOP (`stop sign`) | **> 80%** | Signalisation | Alerte de priorité. Permet au moniteur de valider que l'élève a bien marqué l'arrêt. |
| **Règle 2** | 🚲 Vélo (`bicycle`) | **> 60%** | Usager Vulnérable | Vigilance sur la distance latérale de sécurité (1m en ville / 1.5m hors agglomération). Seuil abaissé car les vélos de loin sont complexes à détecter. |
| **Règle 3** | 🚶 Piéton (`person`) | **> 70%** | Usager Vulnérable | Gestion des passages piétons et évitement de comportement dangereux. |

---

## 📊 Structure du Rapport de Conduite (`rapport_conduite.csv`)

Le script `analyse_video.py` génère un fichier structuré facilitant le débriefing d'une séance. Voici un exemple de sortie générée :

```csv
Frame,Categorie,Objet,Confiance,Position_X
6,Usager Vulnérable,person,0.704838752746582,259.82208251953125
7,Usager Vulnérable,person,0.7003364562988281,259.936279296875
84,Signalisation,stop sign,0.8108956813812256,1226.40771484375
207,Usager Vulnérable,person,0.7017578482627869,83.88787841796875
```

### Signification des colonnes :
- **Frame** : Numéro de l'image de la vidéo permettant de retrouver l'instant exact de l'action.
- **Categorie** : Type d'événement filtré (ex: *Signalisation*, *Usager Vulnérable*).
- **Objet** : Classe d'objet détectée par YOLO (ex: *person*, *stop sign*).
- **Confiance** : Score de confiance de la détection (entre 0 et 1).
- **Position_X** : Coordonnée horizontale (pixel) du coin gauche de la boîte de détection. Utile pour savoir si l'obstacle se situe à gauche, au centre ou à droite de la trajectoire.

---

## 🐳 Conteneurisation avec Docker

Une configuration Docker est prête pour exécuter le traitement de manière isolée et portable.

### 1. Construire l'image Docker
```bash
docker build -t auto-ecole-vision .
```

### 2. Exécuter l'analyse
Puisque le traitement vidéo génère des fichiers de sortie, nous utilisons un **volume** pour monter notre dossier local dans le conteneur Docker. Cela permet de récupérer la vidéo annotée finale ainsi que les rapports.

```bash
docker run -it --rm \
  -v "$(pwd)":/app \
  auto-ecole-vision
```

---

## 🚀 Évolutions futures du PoC (Roadmap)

Pour passer d'un prototype de recherche (Proof of Concept) à un produit prêt à l'emploi en auto-école :

1. **Détection de franchissement de ligne** : Implémenter des algorithmes de traitement d'image de type *Lane Detection* (détection de lignes au sol) pour alerter en cas de chevauchement d'une ligne blanche continue.
2. **Estimation des distances (Monoculaire)** : Utiliser la taille relative des boîtes de détection des voitures devant l'élève pour estimer si la distance de sécurité réglementaire (règle des 2 secondes) est respectée.
3. **Couplage Vitesse & Signalisation** : Croiser la vitesse actuelle du véhicule (via flux GPS ou odomètre OBD2) avec la détection des panneaux de limitation de vitesse (30, 50, 80 km/h) pour remonter les excès de vitesse.
4. **Interface Web de Débriefing (Streamlit/FastAPI)** : Créer un tableau de bord convivial permettant au moniteur de charger la vidéo, visualiser les alertes sur une ligne temporelle interactive et cliquer sur une alerte pour sauter directement à la bonne frame vidéo.

---

## 📝 Auteurs & Licence

Développé dans le cadre de l'automatisation et de l'aide pédagogique à la conduite.
* **Licence** : Propriétaire / Usage interne.
