# 📊 Données du projet

## Source des données

Les données utilisées dans ce projet proviennent de la bibliothèque **scikit-learn** et sont chargées directement via l'API.

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
```

## Pourquoi ce dossier est vide ?

Ce dossier `data/` n'est pas destiné à stocker les fichiers de données mais plutôt à expliquer leur provenance et leur utilisation.

**Avantages de charger les données via scikit-learn** :
- ✅ Données toujours à jour
- ✅ Pas besoin de télécharger manuellement
- ✅ Version standardisée et propre
- ✅ Documentation intégrée accessible via `data.DESCR`
- ✅ Reproductibilité garantie

## Dataset : Breast Cancer Wisconsin (Diagnostic)

### Informations générales
- **Nom** : Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source** : UCI Machine Learning Repository
- **Créateurs** : Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
- **Année** : 1995
- **Taille** : 569 échantillons
- **Features** : 30 caractéristiques numériques
- **Classes** : 2 (Malignant, Benign)

### Description

Le dataset contient des mesures de caractéristiques de noyaux cellulaires présents dans des images numériques de biopsies mammaires (Fine Needle Aspirate - FNA).

**10 features de base calculées pour chaque noyau cellulaire** :
1. radius - rayon
2. texture - écart-type des niveaux de gris
3. perimeter - périmètre
4. area - surface
5. smoothness - variation locale du rayon
6. compactness - périmètre² / surface - 1.0
7. concavity - sévérité des portions concaves
8. concave points - nombre de portions concaves
9. symmetry - symétrie
10. fractal dimension - "approximation de la côte" - 1

**Pour chaque feature, 3 valeurs** :
- **mean** : moyenne
- **se** (standard error) : erreur standard
- **worst** : moyenne des 3 plus grandes valeurs

**Total** : 10 × 3 = 30 features

### Variable cible

- **0** : Malignant (malin / cancéreux)
- **1** : Benign (bénin / non cancéreux)

### Distribution
- Benign : 357 échantillons (62.7%)
- Malignant : 212 échantillons (37.3%)

### Références

**Citation** :
```
Wolberg, W.H., Street, W.N., and Mangasarian, O.L. (1995).
Breast Cancer Wisconsin (Diagnostic) Data Set.
UCI Machine Learning Repository.
```

**Publication** :
```
W.N. Street, W.H. Wolberg and O.L. Mangasarian.
Nuclear feature extraction for breast tumor diagnosis.
IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology,
volume 1905, pages 861-870, San Jose, CA, 1993.
```

## Utilisation dans le projet

Le dataset est chargé au début du notebook et converti en DataFrame pandas pour faciliter l'analyse :

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Chargement
data = load_breast_cancer()

# Conversion en DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Informations
print(data.DESCR)
```

## Licence

Les données sont publiques et disponibles pour un usage éducatif et de recherche.
