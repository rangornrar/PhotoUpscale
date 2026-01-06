# PhotoUp — Batch Upscaling GUI (Real-ESRGAN & SeedVR2)

PhotoUp est une application Python avec interface graphique (Tkinter) permettant l’**upscaling batch de photos**
avec **deux moteurs distincts**, au choix selon le besoin :

- ⚡ **Real-ESRGAN** — rapide, robuste, idéal pour le traitement en masse
- 🎬 **SeedVR2** — modèle de diffusion avancé, plus lent mais qualité maximale

---

## 1. Fonctionnalités

- Upscaling **x2 / x3 / x4 / x5**
- Traitement batch par dossier
- Conservation optionnelle des sous-dossiers
- Deux moteurs sélectionnables :
  - Real-ESRGAN (ncnn Vulkan)
  - SeedVR2 (diffusion)
- Post-traitement OpenCV :
  - micro-détails
  - netteté contrôlée
- Protection visage (MediaPipe)
- Gestion VRAM (cap de résolution, offload CPU)
- Journal d’exécution détaillé

---

## 2. Arborescence attendue

```
photo upscale/
│
├─ photoUp.py
├─ README.md
├─ realesrgan-ncnn-vulkan.exe
├─ models/
│   ├─ realesrgan-x4plus.param
│   ├─ realesrgan-x4plus.bin
│   └─ ...
│
├─ venv/                 ← venv principal (GUI)
│
└─ seedvr2/
    ├─ inference_cli.py
    ├─ src/
    ├─ requirements.txt
    └─ venv/             ← venv SeedVR2 (séparé)
```

---

## 3. Prérequis matériels et système

- Windows 10 / 11 (64 bits)
- GPU NVIDIA (testé : **RTX 3080 – 10 Go VRAM**)
- Drivers NVIDIA à jour
- Environ **50 Go d’espace disque libre**
- Accès Internet (téléchargement modèles)

---

## 4. Installation de Python

### Version obligatoire
👉 **Python 3.11.x (64-bit)**

Les versions 3.12+ et 3.14 ne sont pas compatibles (MediaPipe, SeedVR2).

Téléchargement officiel :
https://www.python.org/downloads/release/python-3119/

Pendant l’installation :
- cocher **Add Python to PATH**
- installer pour tous les utilisateurs (recommandé)

---

## 5. Installation du venv principal (GUI)

Dans le dossier du projet :

```bat
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install opencv-python mediapipe pillow numpy
```

Lancer l’application :

```bat
python photoUp.py
```

---

## 6. Installation de Real-ESRGAN

### 6.1 Télécharger le binaire
Télécharger **realesrgan-ncnn-vulkan** :
https://github.com/xinntao/Real-ESRGAN/releases

Placer :
- `realesrgan-ncnn-vulkan.exe`
- le dossier `models/`

à la racine du projet.

### 6.2 Modèles recommandés
- `realesrgan-x4plus` → **photo (recommandé)**
- `realesrgan-x4plus-anime` → anime / illustration

---

## 7. Installation de SeedVR2

### 7.1 Cloner le dépôt dans ./seedvr2
Depuis le dossier projet :

```bat
git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git seedvr2
```

(ou téléchargement ZIP GitHub)

---

### 7.2 Créer le venv SeedVR2

```bat
py -3.11 -m venv seedvr2\venv
seedvr2\venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

---

### 7.3 Installer PyTorch (CUDA)

Pour RTX 3080 (CUDA 12.1) :

```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Vérification :

```bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

### 7.4 Installer les dépendances SeedVR2

```bat
cd seedvr2
pip install -r requirements.txt
```

---

### 7.5 Test SeedVR2 en CLI

```bat
python inference_cli.py image.png --output out --resolution 2048 --seed 42
```

Si une image est générée → SeedVR2 est opérationnel.

---

## 8. Lancer l’application

```bat
venv\Scripts\activate
python photoUp.py
```

---

## 9. Choix du moteur

### Real-ESRGAN
- Rapide
- Stable
- Parfait pour le batch
- x5 = x4 + resize Lanczos (automatique)

### SeedVR2
- Qualité maximale
- Lent
- Très gourmand en VRAM
- Résolution cible calculée à partir de la largeur d’origine

---

## 10. Paramétrage recommandé (RTX 3080 – 10 Go)

### SeedVR2
- Facteur : x2 ou x3
- Max resolution : **3072**
- Extra args :
```
--dit_offload_device cpu --blocks_to_swap 16
```

### Real-ESRGAN
- Facteur : x4
- Modèle : `realesrgan-x4plus`

---

## 11. Micro-détails & visages

Réglages sûrs (photo) :

- Force micro-détails : 0.25 – 0.35
- sigma_s : 20 – 30
- sigma_r : 0.15 – 0.22
- Unsharp amount : ≤ 0.6
- Protection visage : **ON**
- Padding visage : 0.25 – 0.35

---

## 12. Dépannage

### Erreur MediaPipe
→ vérifier Python 3.11
→ vérifier venv actif

### Erreur Unicode / cp1252
Avant de lancer :

```bat
set PYTHONUTF8=1
chcp 65001
```

### Erreur OOM (SeedVR2)
- réduire `max_resolution`
- augmenter `blocks_to_swap`
- préférer Real-ESRGAN

---

## 13. Philosophie d’usage

- **Real-ESRGAN** : production, batch, vitesse
- **SeedVR2** : images premium, restauration, rendu cinéma

---

## 14. Licence & responsabilité

- Real-ESRGAN et SeedVR2 : licences respectives
- Ce script est fourni tel quel, sans garantie
- Usage professionnel sous votre responsabilité

---


