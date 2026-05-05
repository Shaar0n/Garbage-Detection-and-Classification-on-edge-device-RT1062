# Project Structure

```
garbage_classifier_github/
│
├── README.md                           # Main project documentation
├── LICENSE                             # MIT License
├── CONTRIBUTING.md                     # Contribution guidelines
├── requirements.txt                    # Python dependencies (training)
├── requirements_inference_only.txt     # Minimal dependencies (deployment)
├── .gitignore                         # Git ignore rules
│
├── src/                               # Main application code
│   ├── main.py                        # Mode launcher (detector/classifier)
│   ├── main_classifier.py             # Pure classification mode
│   ├── main_detector.py               # Detection + classification mode
│   ├── labels.txt                     # Garbage class labels
│   └── run_mode.txt                   # Current mode configuration
│
├── scripts/                           # Training and utility scripts
│   ├── setup_gpu_env.ps1              # GPU environment setup
│   ├── prepare_dataset.py             # Dataset preprocessing
│   ├── train_tflite_model.py          # Main training script
│   ├── train_v3_final.py              # V3 training variant
│   ├── train_v4_improved.py           # V4 training variant
│   ├── train_detector_rt1062_v2.py    # Detector model training
│   ├── train_tiny_detector_rt1062.py  # Lightweight detector
│   ├── train_from_scratch.py          # Custom training
│   ├── test_model.py                  # Model testing/validation
│   ├── openmv_export.py               # Export for OpenMV
│   ├── compress_model_for_rt1062.py   # Model compression
│   ├── convert_best_detector_to_tflite.py
│   └── convert_detector_v2_to_tflite.py
│
├── models/                            # Pre-trained models
│   ├── garbage_classifier.tflite      # Deployment-ready model (0.83 MB)
│   ├── garbage_classifier_final.h5    # Full Keras model (4.6 MB)
│   ├── labels.txt                     # Class labels
│   └── model_metadata.json            # Model information
│
└── docs/                              # Documentation
    ├── README.md                      # (copied from root)
    ├── SETUP_GUIDE.md                 # Complete setup instructions
    ├── APPROACH_DETAILS.md            # Technical approach explanation
    ├── DEPLOYMENT.md                  # OpenMV deployment guide
    ├── QUICK_REFERENCE.txt            # Quick reference guide
    └── CONTRIBUTING.md                # (copied from root)
```

## Directory Descriptions

### `src/`
Contains the final working OpenMV deployment code:
- **main.py**: Dual-mode launcher that switches between detection and classification
- **main_classifier.py**: Standalone classifier for pure classification tasks
- **main_detector.py**: Detector with classification for object detection + classification
- **labels.txt**: Garbage class definitions (plastic, paper, metal, etc.)
- **run_mode.txt**: Runtime mode configuration

### `scripts/`
Training and processing utilities for host computer:
- **setup_gpu_env.ps1**: Initializes NVIDIA GPU environment
- **prepare_dataset.py**: Converts dataset to TFRecord format
- **train_*.py**: Various training pipeline options
- **openmv_export.py**: Prepares models for OpenMV deployment
- **compress_model_for_rt1062.py**: Optimizes models for RT1062 constraints

### `models/`
Pre-trained model weights and metadata:
- **garbage_classifier.tflite**: Optimized TensorFlow Lite model (0.83 MB) - deployment ready
- **garbage_classifier_final.h5**: Full Keras model (4.6 MB) - for retraining
- **model_metadata.json**: Class mappings and model info

### `docs/`
Comprehensive documentation:
- **SETUP_GUIDE.md**: Step-by-step environment setup
- **DEPLOYMENT.md**: OpenMV RT1062 deployment instructions
- **APPROACH_DETAILS.md**: Technical architecture and methodology
- **QUICK_REFERENCE.txt**: Quick command reference

## File Sizes

| Component | Size | Purpose |
|-----------|------|---------|
| TFLite Model | 0.83 MB | Production deployment |
| Keras Model | 4.6 MB | Retraining/fine-tuning |
| All Scripts | ~200 KB | Training & utilities |
| Documentation | ~50 KB | Guides & references |

## Quick Navigation

### To Deploy on OpenMV RT1062:
1. Read: `docs/DEPLOYMENT.md`
2. Use: `src/main.py` (or main_classifier.py / main_detector.py)
3. Upload: `models/garbage_classifier.tflite` to MicroSD

### To Retrain the Model:
1. Read: `docs/SETUP_GUIDE.md`
2. Prepare data: `scripts/prepare_dataset.py`
3. Train: `scripts/train_v4_improved.py` (or your preferred variant)
4. Export: `scripts/openmv_export.py`

### To Set Up Development Environment:
1. Read: `README.md` (Getting Started section)
2. Install: `pip install -r requirements.txt`
3. Configure GPU: `scripts/setup_gpu_env.ps1`

## Dependencies

### Training (Full Stack)
See `requirements.txt`:
- TensorFlow 2.13.0+
- NVIDIA CUDA & cuDNN
- OpenCV
- NumPy, Pandas
- Matplotlib (visualization)

### Inference Only (Lightweight)
See `requirements_inference_only.txt`:
- TensorFlow Lite
- NumPy
- Pillow
- (No GPU required)

### OpenMV Deployment
On-device: No external dependencies - uses OpenMV's built-in ML library

## GitHub Upload Checklist

- [x] Source code (Python + MicroPython scripts)
- [x] Pre-trained models (TFLite + Keras)
- [x] Complete documentation
- [x] Training scripts and utilities
- [x] Requirements files
- [x] License file
- [x] .gitignore file
- [x] Contributing guidelines
- [x] README with quick start

---

**This is a production-ready project structure optimized for:**
- ✓ Easy deployment to OpenMV RT1062
- ✓ Quick model training and retraining
- ✓ Community contributions
- ✓ Extensibility and maintenance
