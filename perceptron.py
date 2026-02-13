"""
Perceptron de Rosenblatt - Implémentation en Python
====================================================

Ce module implémente le perceptron, premier neurone artificiel inventé par 
Frank Rosenblatt en 1957. Le perceptron est un classificateur linéaire binaire
qui apprend à séparer deux classes en ajustant itérativement ses poids.

Author: [Votre Nom]
Date: 2025
"""

import numpy as np


class Perceptron:
    """
    Implémentation du Perceptron de Rosenblatt.
    
    Le perceptron est un classificateur binaire qui utilise une fonction 
    d'activation à seuil (step function) pour prédire la classe d'une entrée.
    
    Paramètres
    ----------
    learning_rate : float, default=0.01
        Taux d'apprentissage (eta) contrôlant l'ampleur des mises à jour 
        des poids à chaque itération. Un taux trop élevé peut empêcher la 
        convergence, tandis qu'un taux trop faible ralentit l'apprentissage.
        
    epochs : int, default=100
        Nombre de passages complets sur l'ensemble d'entraînement.
        
    random_state : int, default=None
        Graine pour la génération aléatoire des poids initiaux, 
        permettant la reproductibilité des résultats.
    
    Attributs
    ---------
    weights_ : array, shape = [n_features + 1]
        Poids après ajustement (le premier élément est le biais).
        
    errors_ : list
        Nombre d'erreurs de classification à chaque époque.
        
    Méthode mathématique
    --------------------
    Le perceptron calcule une somme pondérée :
        z = w0 + w1*x1 + w2*x2 + ... + wn*xn = w^T * x + b
    
    Puis applique une fonction d'activation à seuil :
        ŷ = 1 si z ≥ 0
        ŷ = 0 si z < 0
    
    Règle d'apprentissage
    ---------------------
    Les poids sont mis à jour selon :
        w_i := w_i + η * (y - ŷ) * x_i
    où :
        - η (eta) est le taux d'apprentissage
        - y est la vraie étiquette
        - ŷ est la prédiction
        - x_i est la valeur de la feature i
    """
    
    def __init__(self, learning_rate=0.01, epochs=100, random_state=None):
        """
        Initialise le perceptron avec ses hyperparamètres.
        
        Parameters
        ----------
        learning_rate : float
            Taux d'apprentissage
        epochs : int
            Nombre d'époques d'entraînement
        random_state : int
            Graine aléatoire pour reproductibilité
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.weights_ = None
        self.errors_ = []
    
    def _activation_function(self, z):
        """
        Fonction d'activation à seuil (step function).
        
        Cette fonction est la fonction d'activation classique du perceptron.
        Elle retourne 1 si l'entrée est positive ou nulle, 0 sinon.
        
        Parameters
        ----------
        z : float or array-like
            Somme pondérée (w^T * x + b)
        
        Returns
        -------
        int or array
            1 si z ≥ 0, sinon 0
        """
        return np.where(z >= 0.0, 1, 0)
    
    def _net_input(self, X):
        """
        Calcule la somme pondérée (entrée nette).
        
        Formule : z = w0 + w1*x1 + w2*x2 + ... + wn*xn
        Équivalent vectoriel : z = X · w[1:] + w[0]
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Données d'entrée
        
        Returns
        -------
        array, shape = [n_samples]
            Sommes pondérées pour chaque échantillon
        """
        return np.dot(X, self.weights_[1:]) + self.weights_[0]
    
    def fit(self, X, y):
        """
        Entraîne le perceptron sur les données d'entraînement.
        
        Processus d'entraînement :
        1. Initialisation aléatoire des poids
        2. Pour chaque époque :
            a. Pour chaque exemple d'entraînement :
                - Calculer la prédiction ŷ
                - Calculer l'erreur (y - ŷ)
                - Mettre à jour les poids selon la règle d'apprentissage
            b. Enregistrer le nombre total d'erreurs
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Données d'entraînement (features)
        
        y : array-like, shape = [n_samples]
            Étiquettes cibles (0 ou 1)
        
        Returns
        -------
        self : object
            Retourne l'instance elle-même
        """
        # Initialisation du générateur aléatoire
        rgen = np.random.RandomState(self.random_state)
        
        # Initialisation des poids avec de petites valeurs aléatoires
        # Taille : 1 (biais) + nombre de features
        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        
        # Liste pour stocker le nombre d'erreurs par époque
        self.errors_ = []
        
        # Boucle d'entraînement sur le nombre d'époques
        for epoch in range(self.epochs):
            errors = 0
            
            # Parcours de chaque exemple d'entraînement
            for xi, target in zip(X, y):
                # Calcul de la mise à jour : η * (y - ŷ)
                # Si prédiction correcte : update = 0
                # Si prédiction incorrecte : update ≠ 0
                update = self.learning_rate * (target - self.predict(xi))
                
                # Mise à jour des poids pour les features
                # Δw = η * (y - ŷ) * x
                self.weights_[1:] += update * xi
                
                # Mise à jour du biais
                # Δw0 = η * (y - ŷ) * 1
                self.weights_[0] += update
                
                # Comptage des erreurs (update != 0 signifie erreur)
                errors += int(update != 0.0)
            
            # Enregistrement du nombre d'erreurs pour cette époque
            self.errors_.append(errors)
        
        return self
    
    def predict(self, X):
        """
        Prédit la classe pour les données en entrée.
        
        Processus de prédiction :
        1. Calculer la somme pondérée z = w^T * x + b
        2. Appliquer la fonction d'activation
        3. Retourner la classe prédite (0 ou 1)
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Données à prédire
        
        Returns
        -------
        array, shape = [n_samples]
            Classe prédite pour chaque échantillon (0 ou 1)
        """
        # Calcul de l'entrée nette puis application de la fonction d'activation
        return self._activation_function(self._net_input(X))
    
    def score(self, X, y):
        """
        Calcule la précision (accuracy) du modèle.
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Données de test
        
        y : array-like, shape = [n_samples]
            Vraies étiquettes
        
        Returns
        -------
        float
            Proportion de prédictions correctes (entre 0 et 1)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ============================================================================
# EXEMPLE D'UTILISATION SUR DONNÉES FACTICES
# ============================================================================

if __name__ == "__main__":
    """
    Test du perceptron sur des données générées aléatoirement.
    Ce code ne s'exécute que si le script est lancé directement.
    """
    
    print("=" * 70)
    print("TEST DU PERCEPTRON SUR DONNÉES FACTICES")
    print("=" * 70)
    
    # Génération de données factices linéairement séparables
    np.random.seed(42)
    
    # Classe 0 : points autour de (-2, -2)
    X_class0 = np.random.randn(50, 2) + np.array([-2, -2])
    y_class0 = np.zeros(50)
    
    # Classe 1 : points autour de (2, 2)
    X_class1 = np.random.randn(50, 2) + np.array([2, 2])
    y_class1 = np.ones(50)
    
    # Combinaison des données
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([y_class0, y_class1])
    
    # Mélange des données
    shuffle_idx = np.random.permutation(100)
    X, y = X[shuffle_idx], y[shuffle_idx]
    
    print(f"\nDonnées générées : {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"Distribution des classes : {np.bincount(y.astype(int))}")
    
    # Création et entraînement du perceptron
    ppn = Perceptron(learning_rate=0.1, epochs=50, random_state=42)
    ppn.fit(X, y)
    
    # Évaluation
    accuracy = ppn.score(X, y)
    print(f"\nPrécision sur les données d'entraînement : {accuracy:.2%}")
    print(f"Poids finaux : {ppn.weights_}")
    print(f"Nombre d'erreurs à la dernière époque : {ppn.errors_[-1]}")
    
    print("\n" + "=" * 70)
    print("Test terminé avec succès !")
    print("=" * 70)
