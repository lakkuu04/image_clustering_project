# 🧠 AI Image Clustering using CLIP + KMeans

This project uses OpenAI's CLIP model combined with KMeans clustering to group visually and semantically similar images together. It extracts image embeddings using CLIP and clusters them into user-defined folders based on visual similarity.

---

## 📌 Features

- 🔍 Extracts semantic features using `CLIP (ViT-B/32)`
- 🧩 Clusters images with `KMeans`
- 📂 Automatically sorts and saves images into labeled folders (`cluster_0`, `cluster_1`, etc.)
- ✅ Supports GPU (if available) for faster processing

---

## 🚀 Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/lakkuu04/image_clustering_project.git
   cd image_clustering_project
