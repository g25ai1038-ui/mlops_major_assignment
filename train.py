import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def train_model():
    print("Loading Olivetti faces dataset...")
    faces = fetch_olivetti_faces()
    X, y = faces.data, faces.target

    # 70% Trainset और 30% Testset में स्प्लिट
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"Training data shape: {X_train.shape}")
    print("Training Decision Tree Classifier...")

    # Decision Tree Classifier मॉडल ट्रेन करना
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # मॉडल को savedmodel.pth नाम से सेव करना
    model_path = "savedmodel.pth"
    joblib.dump(clf, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_model()