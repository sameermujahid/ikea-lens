import os
import torch
import open_clip
import chromadb
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

# ====================================
# CONFIG
# ====================================

DATASET_PATH = "dataset"
UPLOAD_FOLDER = "uploads"
PORT = int(os.environ.get("PORT", 10000))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = "cpu"  # Render has no GPU

# ====================================
# FLASK INIT
# ====================================

app = Flask(__name__, template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = None
preprocess = None
collection = None

# ====================================
# SAFE MODEL LOADER (LAZY LOAD)
# ====================================

def load_model():
    global model, preprocess

    if model is not None:
        return

    print("🔄 Loading OpenCLIP RN50...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "RN50",
        pretrained="openai"
    )

    model.to(device)
    model.eval()

    print("✅ Model Loaded")

# ====================================
# SAFE CHROMA INIT (LAZY LOAD)
# ====================================

def init_chroma():
    global collection

    if collection is not None:
        return

    print("🔄 Initializing ChromaDB...")

    client = chromadb.PersistentClient(path="chroma_db")

    try:
        collection = client.get_collection("image_embeddings")
        print("✅ Existing collection loaded")
    except:
        collection = client.create_collection("image_embeddings")
        print("🆕 New collection created")

# ====================================
# EMBEDDING FUNCTION
# ====================================

def generate_embedding(image):
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()

# ====================================
# ROUTES
# ====================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/status")
def status():
    return jsonify({
        "model_loaded": model is not None,
        "collection_ready": collection is not None
    })

@app.route("/search", methods=["POST"])
def search():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Lazy load model + DB
    load_model()
    init_chroma()

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    image = Image.open(filepath).convert("RGB")
    query_embedding = generate_embedding(image)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=20,
        include=["metadatas", "distances"]
    )

    response_results = []

    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            similarity_score = max(0, (1 - distance) * 100)

            response_results.append({
                "category": metadata["category"],
                "image_name": metadata["image_name"],
                "image_url": f"/dataset_image/{metadata['category']}/{metadata['image_name']}",
                "similarity_score": round(similarity_score, 1)
            })

    return jsonify({"results": response_results})

@app.route("/dataset_image/<category>/<image_name>")
def serve_dataset_image(category, image_name):
    image_path = os.path.join(DATASET_PATH, category, image_name)
    if os.path.exists(image_path):
        return send_file(image_path)
    return "Image not found", 404

# ====================================
# RUN
# ====================================

if __name__ == "__main__":
    print("🚀 Starting IKEA Lens (Render Safe Mode)")
    app.run(host="0.0.0.0", port=PORT)
