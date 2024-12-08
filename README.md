# TP2 - "Deep Dive" - Architecture de réseaux de neurones avancées

Ce projet contient l'implémentation de trois réseaux de neurones convolutifs pour l'entraînement et l'évaluation sur le dataset **CIFAR-10**. Il inclut une interface interactive permettant de choisir le modèle à entraîner.

---

## 📂 Contenu du fichier `TP2.py`

Le fichier `TP2.py` inclut les éléments suivants :

### 1. **Chargement et Préparation des Données**
- Utilisation du dataset **CIFAR-10**.
- Transformation des données pour normaliser et augmenter leur diversité.

### 2. **Modèles Implémentés**
Le script inclut trois types de modèles CNN :
- **SimpleCNN** :
    Un modèle de base avec des connexions résiduelles simples.
  
- **BottleneckNet** :
    Exploite des blocs **Bottleneck**, qui :
    1. Réduisent les dimensions d'entrée via une convolution 1x1.
    2. Maintiennent les dimensions avec une convolution standard.
    3. Restaurent les dimensions initiales avec une convolution 1x1.

- **InvertedBottleneckNet** :
    S'appuie sur des blocs **Inverted Bottleneck**, qui :
    1. Étendent les dimensions avec une convolution 1x1.
    2. Effectuent une convolution **depthwise** (par canal).
    3. Comprimant les dimensions avec une autre convolution 1x1.

Chaque modèle peut être sélectionné et entraîné via une interface interactive.

### 3. **Entraînement et Validation**
- Choisissez le modèle à entraîner.
- Définissez le nombre d'époques d'entraînement.
- Le script calcule les précisions sur les ensembles d'entraînement et de test après chaque époque.
- Optimisation basée sur :
    - **Adam** pour la descente de gradient.
    - Un planificateur d'apprentissage OneCycleLR.

---

## 🚀 Comment exécuter le projet

1. **Installation des dépendances**
    Assurez-vous que les bibliothèques nécessaires sont installées :
    ```bash
    pip install torch torchvision matplotlib tqdm
    ```

2. **Exécution du script**
    Lancez le fichier `TP2.py` avec Python :
    ```bash
    python TP2.py
    ```

3. **Suivez les instructions interactives :**
    - Choisissez l'un des modèles à entraîner.
    - Indiquez le nombre d'époques d'entraînement.

---

## 📊 Résultats attendus

Les performances des modèles peuvent varier selon les paramètres choisis. Voici un aperçu général des comportements :
- **SimpleCNN** :
    - Plus rapide à entraîner mais moins précis (~60% sur les données de test avec 50 epochs).
- **BottleneckNet** et **InvertedBottleneckNet** :
    - Prennent environ **10 fois plus de temps** que SimpleCNN à cause de leur complexité accrue.
    - Offrent une meilleure précision, surtout sur les données de test (~78% pour BottleneckNet et ~85% pour InvertedBottleneckNet avec 50 epochs).

---

## 🛠️ Configuration recommandée

- **GPU** :
    Ce script est conçu pour fonctionner sur un GPU (si disponible). Il s'adapte automatiquement à la configuration matérielle.
- **Frameworks nécessaires** :
    - PyTorch
    - TorchVision
    - Matplotlib
    - TQDM