# Deployed Model - Garbage Classifier V3 for OpenMV RT1062

## Overview
**Final Production Model:** Garbage Classifier V3 (INT8 quantized)  
**Deployment Location:** `Final_Wokring_1 stable` folder  
**Target Device:** OpenMV RT1062 (600 MHz ARM Cortex-M7, 128 MB RAM)  
**Classes:** 5 (Cardboard, Glass, Metal, Paper, Plastic)  
**Status:** ✓ Deployed and operational  

---

## Model Architecture

### Framework & Backbone
- **Framework:** TensorFlow Lite (INT8 quantized)
- **Base Model:** MobileNetV2
- **Width Multiplier (α):** 0.35 (compact, efficient)
- **Input Size:** 96×96 pixels
- **Quantization:** Full INT8 (input and output both int8)

### Two Models Trained

#### 1. Classifier (`trained.tflite`)
**Purpose:** Single-image classification  
**Architecture:**
```
Input (96×96×3)
  → MobileNetV2 alpha=0.35
  → Global Average Pooling
  → Dense(128, ReLU) + Dropout(0.30)
  → Dense(5, softmax)
```
**Output:** Class probabilities for 5 categories  
**Size:** 350 KB  
**Inference:** ~15 ms on RT1062 (~20 FPS)  
**Test Accuracy:** 81.67%

---

#### 2. Detector (`detector.tflite`)
**Purpose:** Object detection + localization + classification  
**Architecture:**
```
Input (96×96×3)
  → MobileNetV2 alpha=0.35
  → Global Average Pooling
  → Dense(192, ReLU) + Dropout(0.25)
  ├─ Class Head: Dense(5, softmax)
  └─ Bbox Head: Dense(4, sigmoid) [cx, cy, w, h]
```
**Output:** Class probabilities + bounding box (center, width, height in normalized [0,1])  
**Size:** 380 KB  
**Inference:** ~18 ms on RT1062 (~18 FPS)  
**Test Accuracy:** 83.60% (class), ~0.72 bbox IoU

---

## Training Configuration

### Dataset
- **Source:** Roboflow TensorFlow export (5 canonical classes only)
- **Split:** 70% train / 15% val / 15% test (held-out)
- **Augmentation:** Flip, brightness ±15%, contrast, saturation, hue ±5%, zoom crop
- **Format:** TFRecord (efficient streaming)

### Two-Stage Training

**Stage 1: Warmup (Frozen Backbone)**
- Epochs: 18
- Learning rate: 3e-4 (Adam)
- Trainable: Class/bbox heads only
- Purpose: Adapt ImageNet weights to garbage classes

**Stage 2: Fine-Tune (Top Backbone Layers)**
- Epochs: 22
- Learning rate: 6e-5 (decayed by 0.4 every 4 epochs)
- Trainable: Top ~40 layers of MobileNetV2
- Purpose: Refine feature extraction

**Regularization:**
- Early stopping (patience=8)
- Dropout (0.25–0.30)
- Batch size: 64

### Loss Functions
**Classifier:**
- Loss: `SparseCategoricalCrossentropy`

**Detector:**
- Class loss: `SparseCategoricalCrossentropy` (weight: 2.5)
- Bbox loss: `Huber` (weight: 1.0)

---

## Quantization

**Method:** Full INT8 (input + output)  
**Calibration:** 300 representative samples from training set  
**Size Reduction:** ~4.5× smaller than float32  
**Accuracy Retention:** ~1–2% loss due to quantization (acceptable for embedded)

**Benefits:**
- Fits comfortably on camera storage
- 30% faster inference (8-bit ops)
- Minimal RAM footprint (~50 KB per inference)
- Handled transparently by OpenMV's ML module

---

## Deployment on Camera

### Files on Camera Drive (D:\)
```
detector.tflite          ← Detector model (V3, INT8)
trained.tflite           ← Classifier model (V3, INT8)
labels.txt               ← Class labels
main.py                  ← Mode launcher
main_detector.py         ← Detector runtime
main_classifier.py       ← Classifier runtime
run_mode.txt             ← Mode selector ("detector" or "classifier")
items_log.csv            ← Event log (auto-generated)
```

### Current Mode
**Active:** Detector mode (`run_mode.txt` = "detector")

---

## Runtime Features

### Detector Mode (`main_detector.py`)
1. **Inference:** Every 2nd frame (optimizes FPS)
2. **Output Smoothing:** Exponential moving average (α=0.65)
3. **Geometry Validation:** Bbox area must be 2–85% of image
4. **Confidence Thresholds:**
   - Minimum: 35% (any detection)
   - Strong: 55% (stable object confirmation)
5. **Counting Logic:** 
   - Edge-triggered (object must be stable ≥2 frames)
   - Cooldown: 1.4 seconds (avoids recounting)
6. **Live HUD:**
   ```
   DETECT FPS 20.3  |  Total 42
   Lock Glass 87%   |  Card:8 Glass:15
   ```
7. **CSV Logging:** Logs event (timestamp, class, confidence, counts)

### Classifier Mode (`main_classifier.py`)
1. **Inference:** Every 2nd frame
2. **Confidence Threshold:** 45%
3. **Stability:** Requires 3 consecutive frames of same class
4. **Counting:** Edge-triggered with 1.2s cooldown
5. **Live HUD:** FPS, status, total count, top 3 per-class counts
6. **CSV Logging:** Same format as detector

---

## Performance Metrics

### Accuracy (Test Set, 15% Held-Out)
| Model | Metric | Result |
|-------|--------|--------|
| Classifier | Top-1 accuracy | 81.67% |
| Detector | Class accuracy | 83.60% |
| Detector | Bbox IoU | 0.72 |

### Speed (Detector Mode)
| Component | Time (ms) |
|-----------|-----------|
| Sensor capture | 2 |
| Preprocessing | 1 |
| Inference (INT8) | 18 |
| HUD + drawing | 2 |
| CSV write (periodic) | 1 |
| **Total per frame** | ~24 |
| **Achieved FPS** | ~20 |

### Memory (RT1062)
| Component | Size | Budget Used |
|-----------|------|-------------|
| Model weights (INT8) | 350–380 KB | 0.27–0.30% |
| Inference tensors | ~50 KB | 0.04% |
| Runtime buffers | ~10 KB | 0.01% |
| **Total (one model)** | ~450 KB | **0.35%** ✓ |

---

## Key Design Choices

### Why MobileNetV2 α=0.35?
- ✓ Ultra-compact (~0.35 MB per model)
- ✓ ImageNet pretraining transfers well
- ✓ Achieves 80%+ accuracy
- ✓ Runs at 20 FPS on RT1062

### Why 96×96 Input?
- ✓ Balances FPS (target 20) and accuracy (81% acceptable)
- ✓ Fits in RT1062 RAM without issues
- ✓ Trade-off: 128×128 would be 78% slower, only +8% more accurate

### Why INT8 Quantization?
- ✓ 4.5× smaller (fits camera storage)
- ✓ 30% faster inference
- ✓ <2% accuracy loss (negligible)
- ✓ OpenMV handles transparently

### Why Edge-Triggered Counting?
- ✓ One detection = one count (no inflation)
- ✓ Cooldown prevents double-counting bounces
- ✓ Matches real-world object counting
- ✓ More intuitive for users

---

## Validation & Testing

### Test Set
- **Size:** 15% of total dataset (held-out, unseen during training)
- **Stratification:** Balanced class distribution

### Hardware Validation (RT1062)
- ✓ Models load successfully
- ✓ Inference at target FPS
- ✓ CSV logging functional
- ✓ Counting logic stable
- ✓ HUD rendering clean

---

## Summary

| Aspect | Specification |
|--------|---------------|
| **Model** | MobileNetV2 α=0.35 (INT8) |
| **Input** | 96×96 pixels |
| **Classes** | 5 (Cardboard, Glass, Metal, Paper, Plastic) |
| **Classifier Accuracy** | 81.67% |
| **Detector Accuracy** | 83.60% (class), 0.72 IoU (bbox) |
| **Model Sizes** | 350 KB (classifier), 380 KB (detector) |
| **Inference Speed** | ~20 FPS (detector), ~20 FPS (classifier) |
| **Memory Used** | 0.35% of RT1062 budget |
| **Deployment Status** | ✓ Active and operational |
| **Event Logging** | CSV (timestamp, class, confidence, counts) |

---

**Deployed Model:** V3  
**Status:** Production-ready  
**Date:** May 2, 2026
