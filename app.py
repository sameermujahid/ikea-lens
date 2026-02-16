import os
import time
import threading
import torch
import open_clip
import chromadb
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor

# ====================================
# CONFIG
# ====================================

DATASET_PATH = "dataset"
UPLOAD_FOLDER = "uploads"
PORT = int(os.environ.get("PORT", 10000))  # Render default

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ====================================
# FLASK INIT
# ====================================

app = Flask(__name__, template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = None
preprocess = None
collection = None

initialization_status = "Initializing..."
initialization_progress = 0
initialization_start_time = time.time()

# ====================================
# LOAD MODEL
# ====================================

def load_model():
    global model, preprocess

    print("🔄 Loading OpenCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"
    )
    model.to(device)
    model.eval()
    print("✅ OpenCLIP Loaded")

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
# INITIALIZATION (BACKGROUND SAFE)
# ====================================

def initialize_visual_search():
    global collection, initialization_status, initialization_progress

    try:
        print("🚀 Initializing Visual Search System...")

        initialization_status = "Loading model..."
        initialization_progress = 10

        load_model()

        initialization_status = "Connecting to database..."
        initialization_progress = 20

        client = chromadb.PersistentClient(path="chroma_db")

        try:
            collection = client.get_collection("image_embeddings")
            print("✅ Existing collection loaded")
        except:
            collection = client.create_collection("image_embeddings")
            print("🆕 New collection created")

        existing_ids = set(collection.get()["ids"])

        initialization_status = "Scanning dataset..."
        initialization_progress = 30

        all_images = []

        if not os.path.exists(DATASET_PATH):
            print("⚠ Dataset folder not found")
            initialization_status = "Dataset not found"
            return

        for category in os.listdir(DATASET_PATH):
            category_path = os.path.join(DATASET_PATH, category)
            if not os.path.isdir(category_path):
                continue

            for image_name in os.listdir(category_path):
                image_id = f"{category}_{image_name}"

                if image_id not in existing_ids:
                    image_path = os.path.join(category_path, image_name)
                    all_images.append((image_id, image_path, category, image_name))

        if not all_images:
            print("✅ No new images to embed")
            initialization_status = "Ready"
            initialization_progress = 100
            return

        print(f"🆕 New images to process: {len(all_images)}")

        initialization_status = "Embedding images..."
        initialization_progress = 50

        def process_image(data):
            image_id, image_path, category, image_name = data
            try:
                image = Image.open(image_path).convert("RGB")
                embedding = generate_embedding(image)
                return {
                    "id": image_id,
                    "embedding": embedding.tolist(),
                    "metadata": {
                        "category": category,
                        "image_name": image_name
                    }
                }
            except Exception as e:
                print(f"❌ Error: {image_path} → {e}")
                return None

        results = []

        with ThreadPoolExecutor(max_workers=4):  # reduced workers for Render free tier
            for result in map(process_image, all_images):
                if result:
                    results.append(result)

        if results:
            collection.add(
                embeddings=[r["embedding"] for r in results],
                ids=[r["id"] for r in results],
                metadatas=[r["metadata"] for r in results]
            )
            print(f"✅ Added {len(results)} images")

        initialization_status = "Ready"
        initialization_progress = 100

    except Exception as e:
        print("🔥 Initialization crashed:", e)
        initialization_status = "Initialization failed"

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
        "collection_ready": collection is not None,
        "initialization_status": initialization_status,
        "initialization_progress": initialization_progress,
        "can_search": collection is not None
    })

@app.route("/search", methods=["POST"])
def search():

    if collection is None or model is None:
        return jsonify({"error": "System not ready"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

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
    # Start heavy ML initialization in background
    threading.Thread(target=initialize_visual_search, daemon=True).start()

    # Start Flask immediately (critical for Render)
    app.run(host="0.0.0.0", port=PORT)
