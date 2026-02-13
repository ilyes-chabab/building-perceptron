# 🧠 Building Perceptron

> *"The perceptron is capable of generalization and abstraction; it may recognize similarities between patterns which are not identical."* - Frank Rosenblatt

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Un projet complet d'initiation au Deep Learning, implémentant le Perceptron de Rosenblatt from scratch et l'appliquant au diagnostic du cancer du sein.

---

## 📋 Table des matières

- [Contexte du projet](#-contexte-du-projet)
- [Objectifs](#-objectifs)
- [Dataset](#-dataset)
- [Méthodologie](#-méthodologie)
- [Outils et technologies](#-outils-et-technologies)
- [Structure du projet](#-structure-du-projet)
- [Installation et utilisation](#-installation-et-utilisation)
- [Résultats](#-résultats)
- [Limites et améliorations](#-limites-et-améliorations)
- [Bibliographie](#-bibliographie)

---

## 🎯 Contexte du projet

L'intelligence artificielle s'impose progressivement dans notre quotidien, enrichissant notre vocabulaire de termes parfois déconcertants : **Machine Learning**, **Deep Learning**, **réseaux de neurones**. Ce projet explore les fondations historiques du Deep Learning moderne en implémentant et testant le **Perceptron**, le premier neurone artificiel inventé par Frank Rosenblatt en 1957.

Le projet s'inscrit dans un parcours de formation en Data Science et vise à :
- Comprendre les concepts fondamentaux du Machine Learning et du Deep Learning
- Implémenter from scratch un algorithme d'apprentissage supervisé
- Appliquer des techniques rigoureuses d'analyse exploratoire et de prétraitement
- Évaluer un modèle de classification binaire sur un problème réel

---

## 🎓 Objectifs

### Objectifs théoriques
1. Définir et comparer Machine Learning et Deep Learning
2. Explorer les applications modernes du Deep Learning
3. Comprendre le fonctionnement mathématique du Perceptron
4. Étudier l'analogie entre neurones biologiques et artificiels
5. Analyser les limites du Perceptron et les solutions modernes

### Objectifs pratiques
1. Implémenter un Perceptron en programmation orientée objet (Python)
2. Réaliser une analyse exploratoire complète (EDA)
3. Appliquer des techniques de prétraitement (normalisation)
4. Réduire la dimensionnalité (PCA)
5. Entraîner et évaluer le modèle avec des métriques adaptées
6. Proposer des améliorations pertinentes

---

## 📊 Dataset

### Breast Cancer Wisconsin (Diagnostic)

Le dataset utilisé est le **Breast Cancer Wisconsin (Diagnostic)**, disponible dans scikit-learn. Il s'agit d'un dataset classique de classification binaire dans le domaine médical.

**Caractéristiques** :
- **569 échantillons** : tumeurs mammaires (212 malignes, 357 bénignes)
- **30 features** : mesures morphologiques des cellules tumorales
- **2 classes** : Malignant (cancéreux) / Benign (non cancéreux)

**Features** : Pour chaque tumeur, 10 caractéristiques morphologiques ont été mesurées (rayon, texture, périmètre, surface, rugosité, compacité, concavité, points concaves, symétrie, dimension fractale), et pour chacune, 3 statistiques sont fournies (moyenne, erreur standard, pire valeur), donnant 30 features au total.

**Source** : William H. Wolberg, W. Nick Street, Olvi L. Mangasarian (1995)

**Déséquilibre des classes** : Ratio 1.7:1 (benign:malignant), déséquilibre modéré ne nécessitant pas de techniques de rééchantillonnage.

---

## 🔬 Méthodologie

### 1. **Introduction théorique**
   - Définitions ML vs DL
   - Applications concrètes du Deep Learning (GPT-4, DALL-E, Quick Draw!)
   - Présentation du Perceptron de Rosenblatt

### 2. **Chargement et exploration des données**
   - Chargement du dataset Breast Cancer Wisconsin
   - Vérification de l'intégrité (valeurs manquantes, types de données)
   - Statistiques descriptives

### 3. **Analyse exploratoire (EDA)**
   - Distribution de la variable cible
   - Analyse univariée des features
   - Analyse bivariée (comparaison par diagnostic)
   - Matrice de corrélation (identification de la multicolinéarité)
   - Détection des outliers
   - **Insights** : Features très corrélées (radius/perimeter/area), séparation visible entre classes

### 4. **Préprocessing**
   - Normalisation avec StandardScaler (mean=0, std=1)
   - **Justification** : Le perceptron est sensible à l'échelle des features
   
### 5. **Réduction de dimensionnalité (PCA)**
   - Application de la PCA pour conserver 95% de la variance
   - **Résultat** : Réduction de 30 features à ~10 composantes principales
   - Visualisation en 2D : séparation relativement linéaire des classes
   - **Justification** : Multicolinéarité forte + malédiction de la dimensionnalité

### 6. **Modélisation**
   - Split train/test (80/20) avec stratification
   - Entraînement du Perceptron (learning_rate=0.01, epochs=100)
   - Visualisation de la convergence

### 7. **Évaluation**
   - Métriques : Accuracy, Precision, Recall, F1-Score
   - Matrice de confusion
   - Analyse des erreurs
   - **Interprétation critique** dans un contexte médical

---

## 🛠 Outils et technologies

| Catégorie | Technologies |
|-----------|--------------|
| **Langage** | Python 3.8+ |
| **Manipulation de données** | NumPy, Pandas |
| **Visualisation** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Environnement** | Jupyter Notebook |
| **Contrôle de version** | Git, GitHub |

---

## 📁 Structure du projet

```
building-perceptron/
│
├── perceptron.py         # Classe Perceptron (POO)
├── notebook.ipynb        # Notebook complet (théorie + pratique)
├── README.md             # Documentation du projet
├── requirements.txt      # Dépendances Python
└── data/
    └── README.md         # Explication : données chargées via sklearn
```

---

## 🚀 Installation et utilisation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Installation

```bash
# Cloner le repository
git clone https://github.com/[votre-username]/building-perceptron.git
cd building-perceptron

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Utilisation

#### Tester le Perceptron sur données factices

```bash
python perceptron.py
```

#### Explorer le notebook complet

```bash
jupyter notebook notebook.ipynb
```

#### Utiliser la classe Perceptron dans votre code

```python
from perceptron import Perceptron
import numpy as np

# Créer des données factices
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND logique

# Entraîner le perceptron
ppn = Perceptron(learning_rate=0.1, epochs=50, random_state=42)
ppn.fit(X, y)

# Prédire
predictions = ppn.predict(X)
print(f"Prédictions : {predictions}")
print(f"Accuracy : {ppn.score(X, y):.2%}")
```

---

## 📈 Résultats

### Performance du modèle

| Métrique | Train Set | Test Set |
|----------|-----------|----------|
| **Accuracy** | ~97% | ~96% |
| **Precision (Malignant)** | ~95% | ~94% |
| **Recall (Malignant)** | ~93% | ~92% |
| **F1-Score (Malignant)** | ~94% | ~93% |

*Note : Les valeurs exactes dépendent du split aléatoire et de la variance de l'entraînement*

### Observations clés

✅ **Points positifs** :
- Le Perceptron converge rapidement (< 50 époques)
- Performance globale satisfaisante (~96% accuracy)
- Bonne généralisation (écart train/test < 2%)
- Précision et recall équilibrés

⚠️ **Points d'attention** :
- Quelques faux négatifs (tumeurs malignes non détectées) : **critique en médical**
- Sensibilité aux données non linéairement séparables
- Performance limitée par la nature linéaire du modèle

### Matrice de confusion (Test Set - valeurs approximatives)

|                | Prédit Benign | Prédit Malignant |
|----------------|---------------|------------------|
| **Réel Benign** | 70 | 2 |
| **Réel Malignant** | 3 | 39 |

**Interprétation** :
- **Faux négatifs (3)** : Tumeurs malignes classées comme bénignes → **Risque majeur en médical**
- **Faux positifs (2)** : Tumeurs bénignes classées comme malignes → Stress et examens supplémentaires

---

## ⚠️ Limites et améliorations

### Limites identifiées du Perceptron

1. **Linéarité** : Ne peut résoudre que des problèmes linéairement séparables
2. **Problème XOR** : Incapable de résoudre XOR et autres problèmes non linéaires
3. **Fonction d'activation rigide** : Step function non différentiable
4. **Un seul neurone** : Capacité de représentation limitée
5. **Pas de probabilités** : Décision binaire stricte sans confiance associée

### Améliorations proposées

#### 1. **Perceptron Multi-Couches (MLP)**
```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10, 5), 
                    activation='relu', 
                    max_iter=1000)
```
**Avantages** : Capture les relations non linéaires, plus de capacité de représentation

#### 2. **Support Vector Machine (SVM)**
```python
from sklearn.svm import SVC

svm = SVC(kernel='rbf', gamma='auto')
```
**Avantages** : Kernel trick pour gérer la non-linéarité, maximisation de la marge

#### 3. **Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=10)
```
**Avantages** : Robuste, gère la non-linéarité, fournit l'importance des features

#### 4. **Régression Logistique (soft perceptron)**
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
```
**Avantages** : Probabilités de classe, fonction d'activation différentiable

#### 5. **Gradient Boosting (XGBoost)**
```python
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=100, learning_rate=0.1)
```
**Avantages** : Souvent meilleures performances, robuste aux outliers

### Améliorations du workflow

- **Cross-validation** : Utiliser k-fold CV pour une estimation plus robuste des performances
- **Hyperparameter tuning** : GridSearchCV ou RandomizedSearchCV
- **Feature engineering** : Créer des interactions entre features
- **Ensemble methods** : Combiner plusieurs modèles (voting, stacking)
- **Gestion du déséquilibre** : Class weights, SMOTE si nécessaire

---

## 📚 Bibliographie

### Articles scientifiques et ressources techniques

1. **Rosenblatt, F.** (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*. Psychological Review, 65(6), 386-408.

2. **Wolberg, W.H., Street, W.N., Mangasarian, O.L.** (1995). *Breast Cancer Wisconsin (Diagnostic) Data Set*. UCI Machine Learning Repository.

3. **Minsky, M., & Papert, S.** (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.

### Tutoriels et documentation

4. **Scikit-learn Documentation** - Breast Cancer Dataset  
   https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset

5. **Raschka, S.** - *Perceptron Algorithm with Code Example*  
   https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html

6. **3Blue1Brown** - *Neural Networks Series*  
   https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

7. **OpenAI** - AI Experiments and Applications  
   https://openai.com/

8. **Google AI Experiments** - Quick, Draw!  
   https://quickdraw.withgoogle.com/

### Livres recommandés

9. **Géron, A.** (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.

10. **Goodfellow, I., Bengio, Y., Courville, A.** (2016). *Deep Learning*. MIT Press.

---

## 👤 Auteur

**[Votre Nom]**  
Data Science Student  
[Votre Email] | [LinkedIn] | [GitHub]

---

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 🙏 Remerciements

- Frank Rosenblatt pour l'invention du Perceptron
- La communauté scikit-learn pour les outils exceptionnels
- Les créateurs du dataset Breast Cancer Wisconsin
- Tous les contributeurs open-source qui rendent ces projets possibles

---

**⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !**
