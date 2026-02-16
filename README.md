# Visual Furniture Search System

A production-ready image similarity search system built with Flask, OpenCLIP, and ChromaDB.

This application allows users to upload an image of furniture and retrieve visually similar items from a pre-indexed dataset using vector embeddings and semantic search.

---

## Features

* Image-based furniture search
* OpenCLIP (ViT-B-32) embeddings
* ChromaDB persistent vector storage
* Background model initialization
* Progressive initialization status tracking
* Google Images–style responsive UI
* Lazy loading image grid
* Render deployment ready
* CPU/GPU auto-detection

---

## Architecture Overview

### Backend

* Flask API
* OpenCLIP (ViT-B-32, pretrained="openai")
* ChromaDB PersistentClient
* ThreadPoolExecutor for batch embedding
* Background initialization thread

### Frontend

* Pure HTML, CSS, Vanilla JS
* Real-time system status polling
* Drag & drop image upload
* Progressive results rendering

---

## Project Structure

```
ikea-lens/
│
├── app.py
├── dataset/
│   ├── chair/
│   ├── sofa/
│   └── table/
│
├── uploads/
├── chroma_db/
├── templates/
│   └── index.html
│
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Initialization (Background Thread)

When the server starts:

* Loads OpenCLIP model
* Connects to ChromaDB
* Scans dataset directory
* Generates embeddings for new images
* Stores embeddings with metadata:

  * category
  * image_name

The Flask server starts immediately while initialization runs in background (critical for Render free tier).

---

### 2. Image Embedding

Each image is processed as:

```python
features = model.encode_image(image)
features = features / features.norm(dim=-1, keepdim=True)
```

Embeddings are normalized for cosine similarity search.

---

### 3. Search Flow

1. User uploads image
2. Image embedding generated
3. ChromaDB queried for nearest vectors
4. Distance converted to similarity percentage
5. Top 20 results returned

Similarity formula:

```
similarity = (1 - distance) * 100
```

---

## Installation

### 1. Clone the repository

```
git clone [<your-repo-url>](https://github.com/sameermujahid/ikea-lens)
cd ikea-lens
```

### 2. Create virtual environment

```
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

Example requirements.txt:

```
flask
torch
open_clip_torch
chromadb
numpy
Pillow
```

---

## Dataset Format

Dataset must follow this structure:

```
dataset/
    category_name/
        image1.jpg
        image2.jpg
```

Example:

```
dataset/
    chair/
    sofa/
    dining_table/
```

Each folder name becomes the predicted category label.

---

## Running Locally

```
python app.py
```

Default port:

```
http://localhost:10000
```

---

## API Endpoints

### GET /

Loads frontend UI.

---

### GET /status

Returns system initialization state:

```json
{
  "model_loaded": true,
  "collection_ready": true,
  "initialization_status": "Ready",
  "initialization_progress": 100,
  "can_search": true
}
```

---

### POST /search

Form-data:

```
file: image file
```

Response:

```json
{
  "results": [
    {
      "category": "chair",
      "image_name": "chair1.jpg",
      "image_url": "/dataset_image/chair/chair1.jpg",
      "similarity_score": 87.5
    }
  ]
}
```

---

### GET /dataset_image/<category>/<image_name>

Serves dataset image.

---

## Deployment on Render

Important considerations:

* Use PORT from environment
* Background thread prevents boot timeout
* Limit ThreadPoolExecutor workers (max_workers=4)
* Use CPU mode if no GPU available

Render start command:

```
python app.py
```

---

## Performance Considerations

* GPU automatically used if available
* Embeddings computed only for new images
* ChromaDB persistent storage
* Threaded batch embedding
* Lazy frontend image loading

---

## Limitations

* No authentication
* No pagination API (frontend handles load more)
* No automatic dataset refresh after startup
* No batching for search requests

---

## Future Improvements

* Add pagination at API level
* Add batch search support
* Add text-to-image search
* Add re-ranking model
* Add caching layer
* Convert to FastAPI + async
* Dockerize for scalable deployment

---

## Technical Stack

Backend:

* Python
* Flask
* OpenCLIP
* ChromaDB
* PyTorch

Frontend:

* HTML
* CSS
* Vanilla JavaScript

---
