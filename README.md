# Signature Detection — YOLOv8 on Azure Functions

An Azure Function that detects handwritten signatures in document images using a custom-trained **YOLOv8** model. Given a document image (as a base64 string or byte buffer), it returns the number of signatures found, their bounding box coordinates, confidence scores, and — optionally — base64-encoded image crops of each signature.

---

## Features

- **YOLOv8-powered detection** — fast, accurate object detection with a fine-tuned `best.pt` model
- **Bounding region output** — polygon coordinates for every detected signature
- **Optional signature crops** — returns base64-encoded PNG crops of each detected region
- **Flexible image input** — accepts base64 data URIs or raw byte buffers
- **Azure Functions HTTP trigger** — deploy as a serverless REST endpoint (GET/POST)
- **Structured JSON response** — includes timestamp, transaction ID, signature count, and per-detection results

---

## Project Structure
signaturedetection/
├── init.py # Azure Function entry point (HTTP trigger handler)
├── predict.py # Loads YOLOv8 model and runs inference
├── extract.py # Crops and encodes detected signature regions
├── function.json # Azure Function binding configuration
├── best.pt # Custom-trained YOLOv8 weights
└── sample.dat # Sample binary input for local testing

---

## API

### Endpoint

`POST /api/signaturedetection`

### Request Body (JSON)

```json
{
  "file": "<base64 data URI or byte buffer JSON>",
  "settings": {
    "crops": true
  }
}

Field	Type	Required	Description
file	string	Yes	Base64 data URI (data:image/...;base64,...) or JSON byte buffer
settings.crops	boolean	No	If true, includes base64 image crops in response. Default: false

### Response Body (JSON)

{
  "context": {
    "timestamp": "2024-01-15T10:30:00.000Z",
    "transaction_id": "abc-123-..."
  },
  "signatures": 2,
  "result": [
    {
      "type": "signature",
      "confidence": 0.94,
      "bounding_regions": {
        "polygon": [
          {"x": 120, "y": 340},
          {"x": 120, "y": 410},
          {"x": 380, "y": 410},
          {"x": 380, "y": 340}
        ]
      },
      "content": "data:image/jpeg;base64,..."  // only if crops: true
    }
  ]
}

Tech Stack
Component	Technology
Detection model	YOLOv8 (Ultralytics)
Deployment	Azure Functions (Python v2)
Image processing	OpenCV, Pillow, NumPy
Deep learning	PyTorch (CPU)

Local Setup
Prerequisites
Python 3.10+
Azure Functions Core Tools
PyTorch (CPU build)
Installation
pip install -r requirements.txt
Run Locally
func start
The function will be available at http://localhost:7071/api/signaturedetection.

Model
The best.pt file contains custom YOLOv8 weights trained to detect handwritten signatures on document pages. The model was trained and exported using the Ultralytics framework.
