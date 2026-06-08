import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

print("Loading Olivetti faces dataset...")
# 1. Olivetti faces dataset लोड करना
faces = fetch_olivetti_faces()
X, y = faces.data, faces.target

# 2. डेटा को 70% trainset और 30% testset में स्प्लिट करना
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
print(f"Training data shape: {X_train.shape}")

# 3. scikit-learn Decision Tree Classifier मॉडल ट्रेन करना
print("Training Decision Tree Classifier...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. joblib का उपयोग करके मॉडल को savedmodel.pth नाम से सेव करना
joblib.dump(model, "savedmodel.pth")
print("Model successfully saved to savedmodel.pth")
