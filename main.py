from perceptron.Perceptron import Perceptron
import pandas as pd  
from sklearn.model_selection import train_test_split

df = pd.read_csv("bcw_data.csv")
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

features = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]
X = df[features]
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# création du modèle
model = Perceptron(n_features=X_train.shape[1])

# apprentissage
model.train(X_train, y_train, epochs=20)

# test
y_pred = model.predict(X_test)  # ici on passe tout le dataset
print(y_pred[:10])

# accuracy
accuracy = sum(yp == yt for yp, yt in zip(y_pred, y_test)) / len(y_test)
print("Accuracy:", accuracy)
