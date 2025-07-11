import pickle
from sklearn.svm import SVC
import numpy as np

# Load the embeddings and labels
with open("models/embeddings.pkl", "rb") as f:
    embeddings, labels = pickle.load(f)

# Train an SVM classifier
model = SVC(kernel='linear', probability=True)
model.fit(embeddings, labels)

# Save the trained classifier
with open("models/classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Classifier trained and saved to models/classifier.pkl")
