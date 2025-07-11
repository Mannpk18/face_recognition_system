import os
import cv2
import numpy as np
import torch
import pickle
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import pickle

# Load pre-trained FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding(img_path):
    try:
        print(f"[READING] {img_path}")
        img = Image.open(img_path).convert('RGB')
        img = img.resize((160, 160))

        img_np = np.asarray(img) / 255.0  # Normalize
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC → CHW

        if img_np.shape != (3, 160, 160):
            print(f"[SKIPPED] Bad shape: {img_np.shape}")
            return None

        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            embedding = model(img_tensor)

        if embedding.shape[-1] != 512:
            print(f"[FAIL] Embedding shape wrong → {embedding.shape}")
            return None

        print(f"[OK] {img_path}")
        return embedding.squeeze(0).numpy()

    except Exception as e:
        print(f"[EXCEPTION] {img_path} → {e}")
        return None


def process_directory(aligned_dir):
    embeddings = []
    labels = []

    print(f"\n[CHECK] aligned_dir = {aligned_dir}")
    print(f"[LISTING] folders in aligned_dir:")
    for person in os.listdir(aligned_dir):
        print(" └──", person)

    for person in os.listdir(aligned_dir):
        person_path = os.path.join(aligned_dir, person)
        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            embedding = get_embedding(img_path)
            if embedding is not None:
                print(f"[OK] {img_path}")
                embeddings.append(embedding)
                labels.append(person)

    print(f"\n✅ Total embeddings: {len(embeddings)}")
    return np.vstack(embeddings), np.array(labels)

if __name__ == "__main__":
    aligned_dir = "C:/Users/pinal/Downloads/face_recognition_project/aligned"
    embeddings, labels = process_directory(aligned_dir)

    os.makedirs("../models", exist_ok=True)
    with open("models/embeddings.pkl", "wb") as f:
        pickle.dump((embeddings, labels), f)


    print("✅ Embeddings saved to models/embeddings.pkl")
