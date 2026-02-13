# Building Perceptron -- Réponses aux Questions 1 à 7
python -m streamlit run data_report_app.py

## 1. Qu'est-ce qu'un Perceptron ? Quel est le lien avec un neurone biologique ?

### Définition

Le **Perceptron** est le premier modèle de neurone artificiel, inventé
par **Frank Rosenblatt en 1957**.\
C'est un algorithme de **classification binaire supervisée** capable de
séparer deux classes à l'aide d'une frontière linéaire.

### Lien avec un neurone biologique

  Neurone biologique                  Perceptron
  ----------------------------------- ---------------------------
  Dendrites (reçoivent les signaux)   Entrées (x₁, x₂, ..., xₙ)
  Synapses (pondèrent le signal)      Poids (w₁, w₂, ..., wₙ)
  Corps cellulaire                    Somme pondérée
  Potentiel d'activation              Fonction d'activation
  Axone (sortie)                      Sortie (y)

Le perceptron est une simplification mathématique du fonctionnement d'un
neurone biologique.

------------------------------------------------------------------------

## 2. Fonction mathématique du Perceptron et son usage

### Formule

y = f( Σ (wᵢ xᵢ) + b )

### Définition des termes

-   xᵢ : variables d'entrée (features)
-   wᵢ : poids associés aux entrées
-   b : biais (intercept)
-   Σ (wᵢ xᵢ) : somme pondérée
-   f : fonction d'activation
-   y : sortie (classe prédite)

### Usage

Le perceptron est utilisé pour : - La classification binaire - Les
problèmes linéairement séparables - L'introduction aux réseaux de
neurones

------------------------------------------------------------------------

## 3. Règles d'apprentissage du Perceptron

### Règle de mise à jour des poids

Si erreur :

wᵢ ← wᵢ + η (y_true − y_pred) xᵢ\
b ← b + η (y_true − y_pred)

### Définitions

-   η : taux d'apprentissage (learning rate)
-   y_true : vraie classe
-   y_pred : prédiction du modèle

Les poids sont ajustés uniquement si la prédiction est incorrecte.

------------------------------------------------------------------------

## 4. Fonction d'activation utilisée

Le perceptron classique utilise la **fonction seuil (fonction de
Heaviside)** :

-   1 si z ≥ 0\
-   0 sinon

La sortie peut aussi être codée en {-1, +1}.

------------------------------------------------------------------------

## 5. Processus d'entraînement du Perceptron

1.  Initialisation aléatoire des poids\
2.  Pour chaque observation :
    -   Calcul de la somme pondérée
    -   Application de la fonction d'activation
    -   Comparaison avec la vraie classe
    -   Mise à jour des poids si erreur
3.  Répétition sur plusieurs epochs\
4.  Arrêt lorsque :
    -   plus d'erreur
    -   ou nombre maximal d'itérations atteint

Le perceptron converge uniquement si les données sont linéairement
séparables.

------------------------------------------------------------------------

## 6. Limites du Perceptron

-   Ne résout pas les problèmes non linéaires (exemple : XOR)
-   Classification uniquement binaire (version de base)
-   Frontière de décision strictement linéaire
-   Sensible au choix du taux d'apprentissage

------------------------------------------------------------------------

## 7. Développement d'un Perceptron en Python (POO)

Pour développer un perceptron en programmation orientée objet :

-   Création d'une classe `Perceptron`
-   Méthodes principales :
    -   `__init__()` : initialisation des poids
    -   `fit()` : entraînement
    -   `predict()` : prédiction
-   Génération de données factices avec `numpy`
-   Évaluation via accuracy

Ce développement permet de comprendre : - Le fonctionnement interne d'un
modèle linéaire - L'impact des poids et du biais - Le mécanisme
d'apprentissage supervisé
