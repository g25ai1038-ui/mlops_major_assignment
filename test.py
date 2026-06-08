import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

print("Loading dataset for testing...")
faces = fetch_olivetti_faces()
X, y = faces.data, faces.target

_, X_test, _, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print("Loading the saved model (savedmodel.pth)...")
model = joblib.load("savedmodel.pth")

accuracy = model.score(X_test, y_test)
print("====================================")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("====================================")
