class Perceptron:

    def __init__(self, n_features, learningRate=0.1, threshold=0.5):
        self.weights = [0.0] * n_features
        self.learningRate = learningRate
        self.threshold = threshold

    def step(self, z):
        if z >= self.threshold :
            return 1
        else :
            return 0 

    def predict_one(self, X):
        z = sum(x*w for x, w in zip(X, self.weights))
        return self.step(z)

    def predict(self, X_set):
        return [self.predict_one(X) for X in X_set]

    def train(self, X_train, y_train, epochs=10):
        for e in range(epochs):
            for X, y_true in zip(X_train, y_train):
                y_pred = self.predict_one(X) 
                error = y_true - y_pred

                # mise à jour des poids
                for i in range(len(self.weights)):
                    self.weights[i] += self.learningRate * error * X[i]




