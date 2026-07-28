# Formalisme Mathématique — Auto-École Vision

Ce document formalise les modèles utilisés dans `src/auto_ecole_math/`.

## 1. Géométrie projective et homographie

Hypothèses : plan sol plat, caméra fixe (pas de pitch dynamique), objets
posés sur le sol (point de contact ≈ milieu du bord bas de la bbox).

Une homographie \(H \in \mathbb{R}^{3\times 3}\) relie un pixel homogène
\(\mathbf{x}=(u,v,1)^\top\) à un point sol \(\mathbf{X}=(X,Y,1)^\top\) :

\[
\mathbf{X} \sim H\mathbf{x},
\qquad
X=\frac{(H\mathbf{x})_1}{(H\mathbf{x})_3},\quad
Y=\frac{(H\mathbf{x})_2}{(H\mathbf{x})_3}.
\]

Estimation : DLT + RANSAC (`cv2.findHomography`) à partir de \(n\ge 4\)
correspondances image↔sol stockées dans `config/homography_default.json`.

Distance ego–objet :

\[
d=\|(X,Y)-(X_{\mathrm{ego}},Y_{\mathrm{ego}})\|_2.
\]

**Limite** : la calibration par défaut est une IPM approximative pour
1280×832. Pour une métrologie fiable, recalibrer avec des marquages de
longueur connue.

## 2. Filtre de Kalman (vitesse constante)

État dans le plan sol :

\[
\mathbf{z}_k=[X,\,Y,\,\dot X,\,\dot Y]^\top.
\]

Dynamique :

\[
\mathbf{z}_{k|k-1}=F(\Delta t)\,\mathbf{z}_{k-1}+\mathbf{w}_k,
\qquad
F(\Delta t)=\begin{bmatrix}I_2 & \Delta t\,I_2\\ 0 & I_2\end{bmatrix}.
\]

Observation de position : \(\mathbf{y}_k=[X,Y]^\top\).
Bruit de process : modèle d'accélération en bruit blanc discret
(Bar-Shalom). Association multi-objets : algorithme hongrois sur distance
de Mahalanobis, avec coasting pendant les occlusions (`max_age`).

## 3. Time-To-Collision et freinage

Distance et taux de fermeture relatifs :

\[
\mathbf{r}=\mathbf{p}_{\mathrm{obj}}-\mathbf{p}_{\mathrm{ego}},
\quad
d=\|\mathbf{r}\|_2,
\quad
\dot d=\frac{\mathbf{r}\cdot\mathbf{v}_{\mathrm{rel}}}{d}.
\]

\[
\mathrm{TTC}=\begin{cases}
d/(-\dot d) & \text{si }\dot d<0,\\
+\infty & \text{sinon.}
\end{cases}
\]

Distance de freinage (décélération constante \(a\), réaction \(t_r\)) :

\[
d_{\mathrm{frein}}(v)=v\,t_r+\frac{v^2}{2a}.
\]

Sans odométrie, la vitesse ego est nulle dans le repère sol (TTC relatif
pur). Fournir `--ego-speed` si une mesure OBD/GPS est disponible.

## 4. Incertitude et statistiques de séance

Lissage exponentiel des confiances YOLO :

\[
s_t=\alpha c_t+(1-\alpha)s_{t-1}.
\]

Modèle Beta–Bernoulli : postérieure \(\mathrm{Beta}(\alpha_0+s,\beta_0+f)\)
sur la présence d'un objet.

Agrégats de séance : \(\bar d\), \(\widehat{\mathrm{Var}}(d)\), percentiles
et histogramme empirique des TTC.

## 5. Pipeline de données

```
Vidéo → YOLO → filtrage classes/seuils
     → pied de bbox → Homographie (m)
     → Kalman multi-objets
     → TTC / freinage / confiance lissée
     → rapport_conduite.csv + rapport_stats.json
```
