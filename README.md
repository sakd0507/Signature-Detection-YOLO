# Signature Detection — YOLOv8 on Azure Functions

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00D4FF" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Deployment-Azure%20Functions-0078D4" alt="Azure Functions">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C" alt="PyTorch">
</p>

An Azure Function that detects handwritten signatures in document images using a custom-trained YOLOv8 model. Given a document image (as a base64 string or byte buffer), it returns the number of signatures found, their bounding box coordinates, confidence scores, and — optionally — base64-encoded image crops of each signature.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Model Details](#model-details)
- [Local Development](#local-development)

---

## 🎯 Overview

This serverless application provides real-time signature detection capabilities for document processing workflows. Built on YOLOv8's state-of-the-art object detection architecture, the system can identify and extract handwritten signatures from various document types including contracts, forms, agreements, and official correspondence.

**Use Cases:**
- Document verification and authentication
- Automated contract processing
- Digital archival and indexing
- Compliance and audit workflows
- Form completion validation

---

## ✨ Features

### Core Capabilities

- ✅ **YOLOv8-Powered Detection** — Fast, accurate object detection with a fine-tuned `best.pt` model
- ✅ **Bounding Region Output** — Polygon coordinates for every detected signature
- ✅ **Optional Signature Crops** — Returns base64-encoded PNG crops of each detected region
- ✅ **Flexible Image Input** — Accepts base64 data URIs or raw byte buffers
- ✅ **Azure Functions HTTP Trigger** — Deploy as a serverless REST endpoint (GET/POST)
- ✅ **Structured JSON Response** — Includes timestamp, transaction ID, signature count, and per-detection results

### Advanced Features

- 📊 **Confidence Scoring** — Each detection includes a confidence score (0-1 scale)
- 📐 **Precise Localization** — Pixel-accurate bounding box coordinates
- 🔄 **Batch Processing Support** — Handle multiple documents via API calls
- ⚡ **Low Latency** — Optimized inference pipeline for quick response times
- 🔒 **Transaction Tracking** — Unique transaction IDs for request tracing

---

## 🏗️ Technical Architecture

### YOLOv8 Object Detection

**YOLO (You Only Look Once)** is a family of real-time object detection models. YOLOv8 represents the latest iteration with significant improvements:

#### Why YOLOv8 for Signature Detection?

1. **Single-Pass Architecture** 
   - Processes entire image in one forward pass
   - Much faster than region-based methods (R-CNN, Fast R-CNN)
   - Ideal for serverless environments with time constraints

2. **Anchor-Free Detection**
   - YOLOv8 uses anchor-free prediction
   - Better generalization to signature variations (size, aspect ratio, orientation)
   - Simplified training process

3. **Feature Pyramid Network (FPN)**
   - Multi-scale feature extraction
   - Detects signatures of varying sizes
   - Handles both small signatures and large flourishes

#### Model Architecture Flow

Input Image (base64/bytes)
↓
Decode & Preprocessing
↓
YOLOv8 Backbone (CSPDarknet)
↓
Neck (PAN - Path Aggregation Network)
↓
Detection Head
↓
Post-Processing (NMS - Non-Maximum Suppression)
↓
Bounding Boxes + Confidence Scores
↓
Optional: Crop Extraction
↓
JSON Response

### Component Breakdown

#### 1. **Image Input Handler** (`__init__.py`)
- Accepts HTTP requests (GET/POST)
- Parses base64 data URIs or JSON byte buffers
- Validates input format
- Generates unique transaction IDs

#### 2. **Prediction Engine** (`predict.py`)
- Loads custom-trained `best.pt` weights
- Runs YOLOv8 inference on input image
- Applies Non-Maximum Suppression (NMS) to eliminate duplicate detections
- Returns bounding box coordinates and confidence scores

**Key Functions:**
```python
def load_model():
    """Loads YOLOv8 model from best.pt weights"""
    
def predict(image_data):
    """
    Runs inference on image
    Returns: List of detections with bboxes and confidence
    """
```

#### 3. **Signature Extractor** (`extract.py`)
- Crops detected signature regions from original image
- Encodes crops as base64 PNG strings
- Handles edge cases (signatures near borders)

**Key Functions:**
```python
def extract_signature_crops(image, detections):
    """
    Extracts and encodes signature regions
    Returns: List of base64-encoded image crops
    """
```

### Deployment Architecture
Client Request
↓
Azure API Gateway
↓
Azure Functions Runtime
↓
Python 3.10+ Environment
↓
YOLOv8 Model (CPU Inference)
↓
JSON Response

### Request Format

#### Headers
Content-Type: application/json

#### Request Body

```json
{
  "file": "<base64 data URI or byte buffer JSON>",
  "settings": {
    "crops": true
  }
}
```

#### Field Specifications

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | `string` | **Yes** | Base64 data URI (`data:image/...;base64,...`) or JSON byte buffer |
| `settings.crops` | `boolean` | No | If `true`, includes base64 image crops in response. Default: `false` |

#### Example Request (Base64 Data URI)

```json
{
  "file": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD...",
  "settings": {
    "crops": true
  }
}
```

#### Example Request (Byte Buffer)

```json
{
  "file": {
    "type": "Buffer",
    "data": [255, 216, 255, 224, 0, 16, 74, 70, 73, 70, ...]
  },
  "settings": {
    "crops": false
  }
}
```

### Response Format

#### Success Response (200 OK)

```json
{
  "context": {
    "timestamp": "2024-01-15T10:30:00.000Z",
    "transaction_id": "abc-123-def-456"
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
      "content": "data:image/jpeg;base64,..."
    },
    {
      "type": "signature",
      "confidence": 0.87,
      "bounding_regions": {
        "polygon": [
          {"x": 450, "y": 680},
          {"x": 450, "y": 750},
          {"x": 710, "y": 750},
          {"x": 710, "y": 680}
        ]
      },
      "content": "data:image/jpeg;base64,..."
    }
  ]
}
```

#### Response Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `context.timestamp` | `string` | ISO 8601 timestamp of request processing |
| `context.transaction_id` | `string` | Unique identifier for request tracking |
| `signatures` | `integer` | Total number of signatures detected |
| `result` | `array` | List of detection objects |
| `result[].type` | `string` | Always `"signature"` for this model |
| `result[].confidence` | `float` | Detection confidence score (0-1) |
| `result[].bounding_regions.polygon` | `array` | Four corner coordinates (top-left, top-right, bottom-right, bottom-left) |
| `result[].content` | `string` | Base64-encoded PNG crop (only if `settings.crops: true`) |

#### Error Response (400 Bad Request)

```json
{
  "error": "Invalid image format",
  "transaction_id": "abc-123-def-456"
}
```

#### Error Response (500 Internal Server Error)

```json
{
  "error": "Model inference failed",
  "transaction_id": "abc-123-def-456"
}
```

---
### Core File Details

#### `__init__.py` - Azure Function Handler

```python
"""
Main entry point for Azure Function
- Accepts HTTP GET/POST requests
- Parses base64 or byte buffer image data
- Calls prediction engine
- Optionally extracts signature crops
- Returns structured JSON response
"""
```

#### `predict.py` - YOLOv8 Inference

```python
"""
Handles model loading and inference
- Loads best.pt on cold start
- Preprocesses input images
- Runs YOLOv8 detection
- Applies confidence thresholding
- Returns normalized coordinates
"""
```

#### `extract.py` - Crop Extraction

```python
"""
Extracts and encodes signature regions
- Takes original image + bounding boxes
- Crops each detected signature
- Converts to PNG format
- Base64 encodes for JSON response
"""
```

#### `function.json` - Azure Configuration

```json
{
  "scriptFile": "__init__.py",
  "bindings": [
    {
      "authLevel": "function",
      "type": "httpTrigger",
      "direction": "in",
      "name": "req",
      "methods": ["get", "post"],
      "route": "signaturedetection"
    },
    {
      "type": "http",
      "direction": "out",
      "name": "$return"
    }
  ]
}
```

---

## 🔧 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Detection Model** | YOLOv8 (Ultralytics) | State-of-the-art object detection |
| **Deep Learning Framework** | PyTorch (CPU) | Model inference engine |
| **Image Processing** | OpenCV, Pillow | Image decoding, cropping, encoding |
| **Numerical Computing** | NumPy | Array operations and transformations |
| **Deployment Platform** | Azure Functions (Python v2) | Serverless compute environment |
| **HTTP Framework** | Azure Functions Runtime | Request handling and routing |
| **Model Format** | PyTorch (.pt) | Native YOLOv8 weight format |

### Dependencies

```txt
ultralytics>=8.0.0        # YOLOv8 framework
torch>=2.0.0              # PyTorch (CPU build)
torchvision>=0.15.0       # Vision utilities
opencv-python>=4.8.0      # Image processing
pillow>=10.0.0            # Image encoding/decoding
numpy>=1.24.0             # Numerical operations
azure-functions>=1.18.0   # Azure Functions SDK
```

---

## 🤖 Model Details

### YOLOv8 Custom Training

The `best.pt` file contains YOLOv8 weights custom-trained for signature detection on document images.

#### Training Dataset Characteristics

- **Dataset Type**: Annotated document images with signature bounding boxes
- **Classes**: Single class ("signature")
- **Training Samples**: Diverse set of handwritten signatures
- **Variations Covered**:
  - Different signature styles (cursive, print, mixed)
  - Various document backgrounds (white, colored, textured)
  - Multiple signature sizes and aspect ratios
  - Different scanning qualities and resolutions

#### Model Specifications

| Attribute | Value |
|-----------|-------|
| **Architecture** | YOLOv8n/s/m (specific variant in best.pt) |
| **Input Size** | 640×640 pixels (default YOLOv8) |
| **Model Size** | ~6MB |
| **Inference Device** | CPU (serverless environment) |
| **Framework** | Ultralytics YOLOv8 |
| **Export Format** | PyTorch (.pt) |

#### Training Configuration (Example)

```yaml
# Typical YOLOv8 training config for signature detection
model: yolov8n.pt        # Base model
epochs: 100              # Training iterations
imgsz: 640               # Input image size
batch: 16                # Batch size
lr0: 0.01                # Initial learning rate
augment: true            # Data augmentation
mosaic: 1.0              # Mosaic augmentation probability
```

#### Performance Characteristics

- **Inference Time**: 50-200ms per image (CPU, depends on image size)
- **Confidence Threshold**: Typically 0.25-0.5 (configurable)
- **NMS IoU Threshold**: 0.45 (reduces duplicate detections)
- **Average Precision**: ~0.85-0.95 on signature detection tasks

### Model Usage in Code

```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Run inference
results = model.predict(
    source=image,
    conf=0.3,           # Confidence threshold
    iou=0.45,           # NMS IoU threshold
    imgsz=640,          # Input size
    device='cpu'        # CPU inference
)

# Extract detections
for result in results:
    boxes = result.boxes.xyxy      # Bounding box coordinates
    confidences = result.boxes.conf # Confidence scores
    classes = result.boxes.cls      # Class IDs (all 0 for signature)
```

---

## 💻 Local Development

### Prerequisites

Before running locally, ensure you have:

- **Python 3.10+** installed
- **Azure Functions Core Tools** v4.x ([Install guide](https://docs.microsoft.com/azure/azure-functions/functions-run-local))
- **PyTorch CPU build** (automatically installed via requirements.txt)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/signature-detection.git
cd signature-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
# Start Azure Functions runtime
func start

# Function will be available at:
# http://localhost:7071/api/signaturedetection
```

### Testing the API

#### Using cURL

```bash
# Test with base64 data URI
curl -X POST http://localhost:7071/api/signaturedetection \
  -H "Content-Type: application/json" \
  -d '{
    "file": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "settings": {
      "crops": true
    }
  }'
```

#### Using Python

```python
import requests
import base64

# Read and encode image
with open("test_document.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Create data URI
data_uri = f"data:image/jpeg;base64,{image_b64}"

# API request
response = requests.post(
    "http://localhost:7071/api/signaturedetection",
    json={
        "file": data_uri,
        "settings": {
            "crops": True
        }
    }
)

result = response.json()
print(f"Signatures detected: {result['signatures']}")
for idx, sig in enumerate(result['result']):
    print(f"Signature {idx+1}: Confidence {sig['confidence']:.2f}")
```

### Debugging

```bash
# Enable verbose logging
export AZURE_FUNCTIONS_ENVIRONMENT=Development

# Run with debugger
func start --verbose
```

---

##  Deployment to Azure

### Deploy via Azure CLI

```bash
# Login to Azure
az login

# Create resource group
az group create --name SignatureDetectionRG --location eastus

# Create storage account
az storage account create \
  --name sigdetectstorage \
  --resource-group SignatureDetectionRG \
  --location eastus

# Create function app
az functionapp create \
  --resource-group SignatureDetectionRG \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.10 \
  --functions-version 4 \
  --name signature-detection-app \
  --storage-account sigdetectstorage

# Deploy function
func azure functionapp publish signature-detection-app
```


---
