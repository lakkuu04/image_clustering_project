import os
import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import shutil

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Extract image embeddings
def extract_embeddings(image_paths):
    embeddings = []
    valid_paths = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        emb = emb.cpu().numpy()[0]
        embeddings.append(emb)
        valid_paths.append(path)
    return np.array(embeddings), valid_paths

# Cluster using KMeans
def cluster_images(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans.labels_

# Organize clustered images into folders
def organize_images(image_paths, labels):
    for label in set(labels):
        os.makedirs(f'cluster_{label}', exist_ok=True)
    for path, label in zip(image_paths, labels):
        name = Path(path).name
        shutil.copy(path, f'cluster_{label}/{name}')

# Main
def main(image_folder, num_clusters):
    image_paths = [str(p) for p in Path(image_folder).rglob("*.jpg")]
    if not image_paths:
        print("❌ No images found.")
        return
    embeddings, valid_paths = extract_embeddings(image_paths)
    labels = cluster_images(embeddings, num_clusters)
    organize_images(valid_paths, labels)
    print(f"✅ Done! Images clustered into {num_clusters} folders.")

# Run
if __name__ == "__main__":
    main(r"D:\photos haiiii", num_clusters=4)
