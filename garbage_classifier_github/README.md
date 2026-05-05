# 🗑️ Garbage Classifier for OpenMV RT1062

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://www.tensorflow.org/)
[![OpenMV](https://img.shields.io/badge/OpenMV-RT1062-blueviolet.svg)](https://openmv.io/)

A complete end-to-end garbage classification system combining **deep learning model training** with **edge device deployment** on the OpenMV RT1062 camera. Features both **classification** and **detection** models with CSV logging and real-time HUD display.

## ✨ Features

- 🤖 **Dual Mode Operation**: Switch between pure classification and detection modes
- 🎯 **Object Detection + Classification**: Detect garbage items and classify them
- 📊 **CSV Logging**: Automatic logging of detected items with confidence scores
- 🎬 **Real-time HUD**: Live predictions with bounding boxes and labels
- 📈 **Count Display**: Running counter for detected garbage types
- ⚡ **GPU Acceleration**: NVIDIA CUDA support for fast training
- 💾 **Model Optimization**: TensorFlow Lite with INT8 quantization for RT1062
- 🚀 **Edge Deployment**: Lightweight models optimized for 1MB RAM constraint
- 📚 **Complete Documentation**: Setup guides, deployment instructions, and troubleshooting

## 🚀 Quick Start

### Deploy to OpenMV RT1062 (30 seconds)

1. Copy to MicroSD: `src/*.py`, `models/garbage_classifier.tflite`
2. Insert MicroSD into RT1062
3. Connect to OpenMV IDE and run `main.py`

**→ Full Deployment Guide: See docs/DEPLOYMENT.md**

### Train Your Own Model

```bash
pip install -r requirements.txt
python scripts/train_v4_improved.py
python scripts/openmv_export.py
```

**→ Complete Setup Guide: See docs/SETUP_GUIDE.md**

## 📋 What's Included

| Folder | Contents | Size |
|--------|----------|------|
| `src/` | Final OpenMV scripts | 22 KB |
| `models/` | Pre-trained models (TFLite + Keras) | 5.4 MB |
| `scripts/` | Training utilities (13 scripts) | 200 KB |
| `docs/` | Complete documentation (6 guides) | 250 KB |

## 🎯 Quick Facts

- **Model Size**: 0.83 MB (TFLite) - fits in RT1062 flash
- **Inference Speed**: 200-500ms per image on RT1062
- **FPS**: 2-5 FPS with 320×320 input
- **Accuracy**: ~92% on validation set
- **Classes**: 6 garbage types (Plastic, Paper, Metal, Glass, Organic, Other)

## 🛠️ Deployment Modes

### Classification Only (Fastest)
- Use: `main_classifier.py`
- Speed: 200-300ms/image
- Memory: Low footprint
- Output: Class + confidence

### Detection + Classification (Complete)
- Use: `main_detector.py`
- Speed: 300-500ms/image
- Output: Bounding boxes + classification

### Dual Mode (Recommended)
- Use: `main.py`
- Switch modes with button press

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| DEPLOYMENT.md | Step-by-step OpenMV setup |
| SETUP_GUIDE.md | Training environment setup |
| APPROACH_DETAILS.md | Technical architecture |
| QUICK_REFERENCE.txt | Quick command reference |
| PROJECT_STRUCTURE.md | File organization |

## ✅ First Time Checklist

- [ ] Read docs/DEPLOYMENT.md
- [ ] Format MicroSD card (FAT32)
- [ ] Copy 5 files to MicroSD
- [ ] Insert into RT1062
- [ ] Connect to OpenMV IDE
- [ ] Run main.py

## 🐛 Troubleshooting

**Model not loading?**
→ Ensure .tflite file in /sd/ root directory

**Poor accuracy?**
→ Check lighting, retrain with diverse data

**Slow inference?**
→ Reduce image size to 224×224

See docs/DEPLOYMENT.md for full troubleshooting guide.

## 🤝 Contributing

Contributions welcome! See CONTRIBUTING.md for development setup and guidelines.

## 📄 License

MIT License - See LICENSE file

## 📊 Project Stats

- Files: 32 | Total Size: 5.61 MB
- Pre-trained Models: 2 | Training Scripts: 4 variants
- Classes: 6 | Accuracy: ~92%

---

**Ready to deploy? Start with docs/DEPLOYMENT.md**

Happy garbage classification! 🗑️✨
