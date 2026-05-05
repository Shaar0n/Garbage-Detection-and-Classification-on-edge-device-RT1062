# Garbage Classification with TensorFlow Lite - Complete Setup Guide

## Project Overview
This project trains a TensorFlow Lite object detection and classification model for garbage classification and deploys it to OpenMV Cam RT1062.

**Features:**
- GPU-accelerated training with NVIDIA CUDA
- TensorFlow Object Detection API integration
- Automated model conversion to TFLite format
- Deployment scripts for OpenMV RT1062
- Optimized for edge device inference

## System Requirements

### Hardware
- NVIDIA GPU (RTX 2060 or better recommended for training)
  - Minimum: 4GB VRAM
  - Recommended: 8GB+ VRAM for faster training
- 16GB+ RAM
- 50GB+ free disk space (for dataset + models)

### Software
- Windows 10/11
- Python 3.10.x
- CUDA 11.8 compatible GPU

### Dataset
- Location: `C:\Users\shall\Mini_Proj\Garbage Classifier.v27i.tensorflow`
- Format: Roboflow TensorFlow format (CSV annotations)
- Splits: Train, Valid, Test

## Project Structure

```
C:\Users\shall\Mini_Proj\garbage_classifier\
├── scripts/                    # Python scripts
│   ├── setup_gpu_env.ps1      # GPU environment setup
│   ├── prepare_dataset.py      # Convert CSV to TFRecord
│   ├── train_tflite_model.py  # Training script
│   └── openmv_export.py       # Deployment preparation
├── data/                       # Dataset directory
│   └── tfrecords/             # Converted TF Records
├── models/                     # Model storage
├── output/                     # Training outputs
│   ├── trained_models/        # Trained model files
│   └── openmv_deployment/     # OpenMV deployment files
└── requirements.txt           # Python dependencies
```

## Quick Start Guide

### Step 1: Activate Virtual Environment
```powershell
cd C:\Users\shall\Mini_Proj
.\tfod_env\Scripts\Activate.ps1
```

### Step 2: Setup GPU Environment (First Time Only)
```powershell
cd C:\Users\shall\Mini_Proj\garbage_classifier\scripts
.\setup_gpu_env.ps1
```

This script will:
- Upgrade pip and setuptools
- Install NVIDIA CUDA/cuDNN support packages
- Install TensorFlow with GPU support
- Install all required dependencies
- Verify GPU setup

**Note:** GPU setup may take 10-15 minutes. It's a one-time setup.

### Step 3: Prepare Dataset
```powershell
cd C:\Users\shall\Mini_Proj\garbage_classifier\scripts
python prepare_dataset.py
```

Output:
- TFRecord files in `data/tfrecords/`
- Class mapping in `data/tfrecords/class_mapping.json`

### Step 4: Train Model
```powershell
python train_tflite_model.py
```

Training will:
- Load the dataset from TFRecords
- Train with GPU acceleration
- Save best model checkpoint
- Save final H5 model
- Convert to TFLite format automatically

**Expected Training Time:** 2-4 hours (depends on GPU and dataset size)

### Step 5: Deploy to OpenMV
```powershell
python openmv_export.py
```

Output:
- Quantized TFLite model
- OpenMV Python script
- Labels configuration
- Deployment guide

## Detailed Steps

### GPU Setup Verification
After running `setup_gpu_env.ps1`, verify GPU setup:

```python
import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
```

### Training Parameters (Adjustable)
Edit `train_tflite_model.py` to customize:

```python
self.IMG_SIZE = 320          # Input image size
self.BATCH_SIZE = 16         # Batch size (reduce if out of memory)
self.LEARNING_RATE = 1e-4    # Learning rate
self.EPOCHS = 50             # Number of epochs
```

**GPU Memory Recommendations:**
- 4GB VRAM: Use IMG_SIZE=224, BATCH_SIZE=8
- 6GB VRAM: Use IMG_SIZE=256, BATCH_SIZE=12
- 8GB VRAM: Use IMG_SIZE=320, BATCH_SIZE=16
- 11GB+ VRAM: Use IMG_SIZE=384, BATCH_SIZE=24

### Output Files

After successful training, check:

```
output/trained_models/
├── garbage_classifier.tflite      # Main model (TFLite format)
├── garbage_classifier_final.h5    # Final H5 model
├── best_model.h5                  # Best checkpoint
├── model_metadata.json            # Model information
└── logs/                          # Training logs (TensorBoard)

output/openmv_deployment/
├── garbage_classifier_quantized_int8.tflite
├── main_openmv.py                 # OpenMV script
├── labels.json                    # Class labels
└── DEPLOYMENT_GUIDE.md           # Detailed deployment guide
```

## OpenMV Deployment

### Prerequisites
1. OpenMV Cam RT1062
2. MicroSD card (16GB recommended)
3. OpenMV IDE installed
4. USB cable

### Deployment Steps

1. **Prepare MicroSD Card**
   - Format as FAT32
   - Create `/sd` directory root

2. **Copy Files to SD Card**
   ```
   /sd/garbage_classifier.tflite    (from openmv_deployment/)
   /sd/labels.json                  (from openmv_deployment/)
   /sd/main.py                      (from openmv_deployment/)
   ```

3. **Connect and Upload**
   - Connect camera to PC via USB
   - Open OpenMV IDE
   - Insert MicroSD card into camera
   - Upload files via IDE File Viewer

4. **Run Script**
   - Open `/sd/main.py` in OpenMV IDE
   - Click "Run" or press Ctrl+R
   - Watch frame buffer for results

### Model Specifications for OpenMV
- **Input Size:** 320x320 (configurable)
- **Output:** Classification scores (0-1 per class)
- **Inference Time:** ~200-500ms per frame
- **Memory Usage:** ~2-4 MB
- **Supported Operations:** TFLite BUILTINS

## Troubleshooting

### GPU Not Detected
```powershell
# Check GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty, install NVIDIA GPU support:
pip install nvidia-cudnn-cu11==8.6.0
pip install nvidia-cuda-runtime-cu11==11.8.89
```

### Out of Memory During Training
1. Reduce BATCH_SIZE in `train_tflite_model.py`
2. Reduce IMG_SIZE
3. Clear GPU memory: `gpuci`

### Model Conversion Fails
- Ensure training completed successfully
- Check that H5 model exists
- Verify TensorFlow version compatibility

### Poor Accuracy on OpenMV
1. Check lighting conditions match training data
2. Verify model is correctly uploaded
3. Adjust CONFIDENCE_THRESHOLD in script
4. Re-train with augmented data

## Performance Optimization

### For Faster Training
```python
# Increase batch size (if GPU memory allows)
self.BATCH_SIZE = 32

# Reduce image size
self.IMG_SIZE = 224

# Reduce epochs
self.EPOCHS = 30
```

### For Better Accuracy
```python
# Increase training time
self.EPOCHS = 100

# Use larger input size
self.IMG_SIZE = 384

# Add data augmentation (modify in prepare_dataset.py)
```

### For OpenMV Deployment
- Use quantized model (int8) for fastest inference
- Reduce INPUT_SIZE to 224x224 for real-time performance
- Enable frame buffering on camera

## Next Steps

1. **Data Collection:** Add more garbage images for better accuracy
2. **Fine-tuning:** Retrain on specific garbage types
3. **Integration:** Integrate with OpenMV peripheral modules
4. **Edge Deployment:** Optimize for battery life and power consumption

## Advanced Configuration

### Custom Model Architecture
Modify `build_model()` in `train_tflite_model.py` to use:
- EfficientNet (better accuracy)
- ResNet (more powerful)
- Custom architecture

### Data Augmentation
Add augmentation in `prepare_dataset.py`:
```python
# Add to parse_tfrecord function
image = tf.image.random_flip_left_right(image)
image = tf.image.random_flip_up_down(image)
image = tf.image.random_brightness(image, 0.2)
```

## Resources

- TensorFlow Lite: https://www.tensorflow.org/lite
- OpenMV Documentation: https://docs.openmv.io/
- CUDA Setup: https://docs.nvidia.com/cuda/cuda-installation-guide-windows/
- Object Detection API: https://github.com/tensorflow/models/tree/master/research/object_detection

## Support & Debugging

### Enable Debug Logging
Set in scripts:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0=all, 1=info, 2=warning, 3=error
```

### Check TensorFlow Installation
```python
import tensorflow as tf
tf.sysconfig.get_build_info()['cuda_version']
```

### Monitor Training with TensorBoard
```powershell
tensorboard --logdir=output\trained_models\logs
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| CUDA not found | Install CUDA 11.8 from NVIDIA |
| Out of memory | Reduce BATCH_SIZE or IMG_SIZE |
| Slow training | Increase BATCH_SIZE or reduce IMG_SIZE |
| Model accuracy low | Add more diverse training data |
| OpenMV script errors | Check file paths in /sd/ directory |

---

**Last Updated:** April 2026
**Version:** 1.0
