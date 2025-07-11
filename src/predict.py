import cv2
import pickle
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

# Load the trained classifier
with open("models/classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Load image and preprocess
image_path = "C:/Users/pinal/Downloads/face_recognition_project/test/pitt_test.jpg"  # replace with your test image path
img = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
img_tensor = transform(img).unsqueeze(0)

# Load FaceNet model
facenet = InceptionResnetV1(pretrained="vggface2").eval()

# Generate embedding
with torch.no_grad():
    embedding = facenet(img_tensor).numpy()

# Predict
prediction = model.predict(embedding)
probability = model.predict_proba(embedding)

name = prediction[0].replace("_", " ").title()


print(f"✅ Predicted: {prediction[0]}")
print(f"🔢 Confidence: {np.max(probability) * 100:.2f}%")
