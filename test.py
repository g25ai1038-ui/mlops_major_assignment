import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_model():
    print("Loading dataset for testing...")
    faces = fetch_olivetti_faces()
    X, y = faces.data, faces.target

    # सेम स्प्लिट (70-30) और random_state=42 रखना है ताकि टेस्ट सेट वही रहे
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Loading the saved model (savedmodel.pth)...")
    # savedmodel.pth को लोड करना
    clf = joblib.load("savedmodel.pth")

    # प्रेडिक्शन करना
    predictions = clf.predict(X_test)
    
    # एक्यूरेसी कैलकुलेट करना
    acc = accuracy_score(y_test, predictions)

    print("\n" + "="*30)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("="*30 + "\n")

if __name__ == "__main__":
    test_model()