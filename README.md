# Building Perceptron 

# Machine learning et deep learning

1) Définition
🔹 Machine Learning (ML)

Le Machine Learning est un domaine de l’intelligence artificielle où l’on entraîne un algorithme à apprendre des relations à partir de données plutôt que de programmer explicitement des règles.

Au lieu de dire :

“si pixel rouge + rond → pomme”

on donne beaucoup d’exemples de pommes et de non-pommes, et l’algorithme apprend lui-même les règles.

Idée clé :
➡️ L’humain conçoit les caractéristiques importantes (features), la machine apprend les paramètres.

Exemples d’algorithmes ML :

régression linéaire/logistique

k-nearest neighbors

arbres de décision / random forest

SVM

clustering (k-means)

🔹 Deep Learning (DL)

Le Deep Learning est un sous-domaine du Machine Learning basé sur les réseaux de neurones profonds (deep neural networks).

Ici la machine apprend directement à partir des données brutes sans que l’humain définisse les caractéristiques.

Exemple :

ML classique : on extrait contours, couleurs, textures d’une image

DL : on donne les pixels → le réseau apprend lui-même les contours

Idée clé :
➡️ La machine apprend les caractéristiques ET la décision.

Architecture typique :

CNN (images)

RNN / LSTM (séquences)

Transformers (texte, audio, vision)

2) Quand utiliser l’un plutôt que l’autre ?
Utiliser le Machine Learning classique quand :

petit dataset

données tabulaires (Excel, base clients, scores)

besoin d’explicabilité (banque, santé, assurance)

ressources matérielles limitées

problème simple de classification/prédiction

📌 Exemple :

prédire si un client va résilier un abonnement

Utiliser le Deep Learning quand :

beaucoup de données

données complexes (image, audio, texte)

reconnaissance de motifs très difficiles

performance maximale recherchée

GPU disponible

📌 Exemple :

reconnaître des visages ou comprendre du langage naturel

3) Applications du Deep Learning (3 exemples)
🧠 1. Vision par ordinateur (Computer Vision)

Le DL permet aux machines de voir et comprendre les images.

Applications :

détection de tumeurs en radiologie

voitures autonomes

reconnaissance faciale

tri automatique d’objets industriels

👉 Les CNN analysent automatiquement les formes, textures et objets.

🗣️ 2. Traitement du langage naturel (NLP)

Les modèles de type Transformer (comme GPT) comprennent et génèrent du texte.

Applications :

assistants conversationnels

traduction automatique

résumé de documents

génération de code

👉 La machine apprend la grammaire et le sens sans règles écrites.

🎵 3. Génération de contenu (IA générative)

Le Deep Learning peut créer du contenu nouveau.

Applications :

génération d’images

musique artificielle

voix synthétique réaliste

vidéo générée par IA

👉 Le modèle apprend la distribution des données et crée de nouveaux exemples plausibles.

#  Réponses aux Questions 1 à 7

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
