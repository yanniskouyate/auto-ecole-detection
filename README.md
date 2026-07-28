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
├── README.md
├── docs/formalisme_mathematique.md   # Formalisme (homographie, Kalman, TTC)
├── config/homography_default.json    # Calibration IPM image ↔ sol (mètres)
├── src/auto_ecole_math/              # Package mathématique Master
│   ├── geometry/                     # Homographie / IPM
│   ├── tracking/                     # Filtre de Kalman + multi-object tracking
│   ├── kinematics/                   # TTC + distance de freinage
│   ├── uncertainty/                  # EWMA / Beta-Bernoulli + stats de séance
│   ├── database/                     # Persistance SQLAlchemy (historique séances)
│   ├── simulation/                   # Génération de séquences annotées
│   └── pipeline.py                   # Orchestration détection → rapport
├── tests/                            # Tests unitaires (pytest)
├── scripts/demo_homography.py        # Démo pixels → mètres
├── scripts/init_db.py                # Création des tables
├── scripts/migrate_csv_to_db.py      # Import d'un rapport CSV existant
├── scripts/generate_dataset.py       # Génération de vidéos annotées
├── scripts/validate_geometry.py      # Erreur du pipeline vs vérité terrain
├── analyse_video.py                  # CLI métier (appelle le pipeline)
├── main.py / detect_auto.py          # Scripts YOLO historiques
├── requirements.txt / pyproject.toml
├── auto_ecole.db                     # Base SQLite (générée, hors dépôt)
└── rapport_conduite.csv / rapport_stats.json
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

### 3. Analyse Mathématique & Rapport (`analyse_video.py`)
Cœur applicatif : YOLO → homographie (mètres) → Kalman → TTC / freinage → CSV + stats JSON.

```bash
PYTHONPATH=src python analyse_video.py
# Sans fenêtre OpenCV :
PYTHONPATH=src python analyse_video.py --no-display
```

Démo homographie seule :

```bash
PYTHONPATH=src python scripts/demo_homography.py --no-display --max-frames 60
```

### 4. Interface Streamlit (rapport interactif)

```bash
# Recommandé (utilise forcément le Python du .venv, pas Anaconda) :
./scripts/run_streamlit.sh

# Équivalent manuel :
source .venv/bin/activate
PYTHONPATH=src python -m streamlit run streamlit_app.py
```

> ⚠️ Si tu lances juste `streamlit run ...` hors venv, macOS peut utiliser
> `/opt/anaconda3/bin/streamlit` **sans OpenCV** → erreur `import cv2`.

Tests unitaires :

```bash
PYTHONPATH=src pytest tests/ -v
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

Le pipeline génère un CSV enrichi (métriques physiques) :

| Colonne | Description |
| :--- | :--- |
| `Frame`, `track_id` | Temps + identité Kalman de l'objet |
| `Categorie`, `Objet`, `Confiance`, `Confiance_lissee` | Classe métier + score YOLO / EWMA |
| `X_m`, `Y_m`, `vx_mps`, `vy_mps`, `speed_mps` | État filtré dans le plan sol |
| `d_m`, `lateral_m`, `ttc_s` | Distance, écart latéral, Time-To-Collision |
| `d_frein_m`, `margin_m` | Distance de freinage théorique et marge |

Un résumé statistique de séance est écrit dans `rapport_stats.json`
(moyenne / variance des distances, distribution des TTC, etc.).

Voir le formalisme détaillé : [`docs/formalisme_mathematique.md`](docs/formalisme_mathematique.md).

---

## 🗄️ Base de données (historique des séances)

Le CSV et le JSON restent générés à chaque analyse, mais chaque séance est
**aussi** enregistrée en base. C'est ce qui permet de comparer plusieurs
séances, de suivre la progression d'un élève et de retrouver une analyse
plusieurs mois après.

### Schéma

```
students ──────┐
               ├──< sessions >──┬──< detections   (1 ligne / objet / frame)
instructors ───┤                └─── statistics   (agrégats, 1–1)
               │
calibrations ──┘
```

| Table | Rôle |
| :--- | :--- |
| `students` / `instructors` | Élèves et moniteurs |
| `sessions` | Une vidéo analysée : date, modèle YOLO, fps, durée, paramètres de freinage |
| `detections` | État filtré de chaque objet suivi, frame par frame (colonnes du CSV) |
| `statistics` | Résumé de séance + compteurs `n_critical_ttc` / `n_negative_margin` |
| `calibrations` | Homographie utilisée, dédupliquée par empreinte SHA-256 (traçabilité) |

### Initialisation

```bash
PYTHONPATH=src python scripts/init_db.py --demo
```

Cela crée `auto_ecole.db` (SQLite) à la racine. Pour passer à PostgreSQL,
aucune modification de code n'est nécessaire :

```bash
export AUTO_ECOLE_DB_URL="postgresql+psycopg://user:pwd@localhost/auto_ecole"
```

### Analyser en rattachant la séance à un élève

```bash
PYTHONPATH=src python analyse_video.py --no-display \
  --student "Alice Dupont" --instructor "M. Martin" \
  --label "Séance 3 — créneau"
```

L'élève et le moniteur sont créés s'ils n'existent pas encore.
Pour désactiver complètement la persistance : `--no-db`.

### Importer les rapports produits avant la mise en place de la base

```bash
PYTHONPATH=src python scripts/migrate_csv_to_db.py \
  --student "Alice Dupont" --label "Séance historique"
```

### Interroger l'historique

```python
from auto_ecole_math.database.db import session_scope
from auto_ecole_math.database.queries import (
    critical_events, list_sessions, student_progress, vulnerable_users_close,
)

with session_scope() as db:
    for s in list_sessions(db, student_id=1):
        print(s.id, s.label, s.date)

    # Piétons / cyclistes détectés à moins de 5 m
    for d in vulnerable_users_close(db, session_id=1, max_distance_m=5.0):
        print(f"frame {d.frame} — {d.class_name} à {d.distance_m:.1f} m")

    # TTC < 3 s ou marge de freinage négative
    print(len(critical_events(db, session_id=1)), "événements à risque")

    # Évolution de l'élève séance après séance
    for row in student_progress(db, student_id=1):
        print(row["date"], row["distance_mean"], row["n_critical_ttc"])
```

L'onglet **« Historique des séances »** de l'interface Streamlit expose ces
mêmes données sans écrire une ligne de SQL.

---

## 🎬 Génération de séquences annotées

Le projet embarque un générateur de dashcam synthétique. Il produit des vidéos
de situations de conduite — dont des scénarios d'accident — accompagnées d'une
**vérité terrain exacte** : les positions annotées sont celles qui ont servi au
rendu, pas une estimation.

Aucun simulateur externe n'est requis (CARLA ne dispose d'aucune build macOS et
exige un GPU NVIDIA) : le rendu s'appuie sur OpenCV et sur le modèle de caméra
du projet.

### Scénarios disponibles

| Scénario | Erreur annotée | Contact |
| :--- | :--- | :--- |
| `conduite_saine` | — (référence négative) | non |
| `distance_securite` | Intervalle trop court, réaction tardive | oui |
| `pieton_non_anticipe` | Piéton traversant, freinage trop tardif | oui |
| `rabattement_non_controle` | Rabattement sur voie occupée | oui |
| `ecart_lateral_cycliste` | Dépassement à moins d'un mètre | non |
| `refus_priorite` | Véhicule transversal, aucun ralentissement | oui |

### Générer

```bash
PYTHONPATH=src python scripts/generate_dataset.py --variations 10 --yolo
```

Chaque séquence produit trois fichiers dans `outputs/dataset/` :

- `<nom>.mp4` — la vidéo ;
- `<nom>_truth.csv` — vérité terrain (colonnes du rapport d'analyse + boîte
  englobante en pixels + `frames_to_collision`) ;
- `<nom>_meta.json` — erreur commise, frame de collision, calibration employée.

`--yolo` ajoute les labels au format YOLO dans `outputs/dataset/labels/`.
Chaque `(scénario, variation)` est déterministe : le jeu est reproductible.

### Mesurer l'erreur du pipeline

C'est le principal intérêt du générateur : disposer enfin d'une référence pour
quantifier la chaîne géométrique, sans que les erreurs de détection de YOLO ne
viennent masquer celles de l'homographie.

```bash
PYTHONPATH=src python scripts/validate_geometry.py
```

Le script rejoue chaque séquence à travers homographie → Kalman → TTC et
compare à la vérité. Trois enseignements ressortent des mesures actuelles :

1. **Biais systématique de distance.** La boîte englobante touche le sol à son
   bord *avant*, pas en son centre : le pipeline sous-estime la distance de la
   demi-longueur de l'objet (≈ 2,25 m pour une voiture, 0,85 m pour un vélo).
   Correction faite, l'erreur résiduelle tombe à quelques millimètres.
2. **Le TTC est inexploitable en début de piste.** Tant que la vitesse du
   filtre de Kalman n'a pas convergé (~15 frames, soit 0,5 s), l'erreur de TTC
   dépasse la trentaine de secondes. Il ne faut pas déclencher d'alerte sur une
   piste non établie.
3. **Le modèle de Kalman est à vitesse constante.** Il décroche pendant un
   freinage appuyé — précisément le cas d'urgence. Un modèle à accélération
   constante serait plus adapté.

### Calibration

⚠️ `config/homography_default.json` **n'est pas physiquement réalisable**. Son
échelle latérale décroît d'un facteur 1,69 entre 4 m et 20 m, alors qu'une
caméra sténopé impose un facteur 5 (l'échelle varie en 1/Y). La seule pose
compatible serait une caméra inclinée à ~86° vers le sol. Concrètement, le
point annoncé à 20 m est en réalité vers 6,8 m : **toutes les distances
au-delà de quelques mètres sont surestimées**.

Le générateur utilise donc par défaut `config/homography_pinhole.json`, dérivée
de paramètres physiques explicites (focale, hauteur de caméra, inclinaison) et
cohérente par construction :

```bash
PYTHONPATH=src python -c "
from auto_ecole_math.simulation.camera import PinholeCamera
PinholeCamera(horizontal_fov_deg=60, height_m=1.25, pitch_deg=4).save_calibration('config/homography_pinhole.json')"
```

Vérifier une calibration quelconque :

```python
from auto_ecole_math.geometry.homography import HomographyEstimator
from auto_ecole_math.simulation.camera import check_pinhole_consistency

h = HomographyEstimator.from_json("config/homography_default.json")
print(check_pinhole_consistency(h))   # -> consistent: False
```

Cette calibration synthétique reste une **valeur de repli** : pour un usage en
production il faut mesurer la vraie caméra sur des repères au sol de longueur
connue.

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
