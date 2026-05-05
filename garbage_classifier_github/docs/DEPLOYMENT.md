# OpenMV RT1062 Deployment Guide

This guide provides step-by-step instructions for deploying the garbage classifier to the OpenMV RT1062 camera.

## Hardware Requirements

- **OpenMV RT1062 Camera Board**
- **MicroSD Card** (FAT32 formatted, 4GB+ recommended)
- **USB Cable** (USB 2.0 or 3.0)
- **OpenMV IDE** (download from https://openmv.io/)

## What You Get

The trained models include:
- **garbage_classifier.tflite** - Optimized TensorFlow Lite model (0.83 MB)
- **garbage_classifier_final.h5** - Full Keras model for retraining (4.6 MB)
- **model_metadata.json** - Class and model information

## Deployment Steps

### Step 1: Prepare MicroSD Card

1. Insert MicroSD card into your computer
2. Format as FAT32 (if not already formatted)
3. Copy these files to the root of the MicroSD:
   - `src/main.py` (or `src/main_classifier.py` or `src/main_detector.py`)
   - `src/main_classifier.py`
   - `src/main_detector.py`
   - `src/labels.txt`
   - `models/garbage_classifier.tflite`

### Step 2: Insert MicroSD into OpenMV RT1062

1. Power off the RT1062
2. Open the MicroSD card slot (usually on the side or back)
3. Insert the MicroSD card (ensure proper orientation)
4. Power on the RT1062

### Step 3: Connect to OpenMV IDE

1. Launch OpenMV IDE
2. Connect RT1062 via USB cable to your computer
3. IDE should auto-detect the camera
4. Go to **File** → **File Viewer** to browse MicroSD contents

### Step 4: Upload and Run

1. In File Viewer, navigate to `/sd/` (MicroSD root)
2. Upload required files if not already present
3. Open `main.py` in the IDE editor
4. Click the **Play** button to run the script
5. Check the **Frame Buffer** tab to see real-time results

## Script Selection

Three deployment scripts are provided:

### 1. **main.py** (Recommended - Dual Mode)
- Switches between detector and classifier modes
- Hold button P0 to toggle modes
- Logs detected items to CSV file
- Requires: `main_detector.py`, `main_classifier.py`, models

### 2. **main_classifier.py** (Classification Only)
- Pure garbage classification
- Single image classification
- Lower memory footprint
- ~200-300ms inference time

### 3. **main_detector.py** (Detection + Classification)
- Object detection with bounding boxes
- Classification of detected objects
- CSV logging of all detections
- ~300-500ms inference time

## Configuration

Edit these settings in the script:

```python
# Input resolution
IMG_WIDTH = 320
IMG_HEIGHT = 320

# Confidence thresholds
CLASSIFY_THRESHOLD = 0.5  # 50% confidence for classifier
DETECT_THRESHOLD = 0.4    # 40% confidence for detector

# CSV Logging
ENABLE_LOGGING = True
LOG_FILE = "/sd/items_log.csv"
```

## Monitoring & Debugging

### Check Frame Buffer
- Press **Frame Buffer** tab in OpenMV IDE
- Real-time preview of camera feed with predictions
- Bounding boxes and confidence scores displayed

### View Logs
1. Connect RT1062 via USB to computer
2. Open File Viewer in OpenMV IDE
3. Navigate to `/sd/items_log.csv`
4. Download and open in Excel or text editor

### Console Output
- Script prints predictions to IDE console
- Useful for debugging and performance monitoring
- Example output:
  ```
  Garbage Type: Plastic, Confidence: 0.92
  Detection Time: 285ms
  ```

## Performance Optimization

### Faster Inference (Target: 4-5 FPS)
- Reduce `IMG_WIDTH` and `IMG_HEIGHT` to 224×224
- Lower confidence thresholds (0.3-0.4)
- Disable CSV logging if not needed
- Disable frame buffer updates

### Better Accuracy
- Use full 320×320 input size
- Increase confidence thresholds (0.6-0.7)
- Ensure proper lighting conditions
- Test with diverse garbage samples

## Common Issues & Solutions

### Camera Not Detected
```
❌ "No device detected"
✅ Solution:
   - Check USB cable connection
   - Try different USB port
   - Restart IDE and camera
   - Update OpenMV firmware
```

### Model Not Loading
```
❌ "No such file" error for garbage_classifier.tflite
✅ Solution:
   - Verify file is in /sd/ root directory
   - Check filename matches exactly
   - Ensure TFLite model, not H5 file
   - Format MicroSD card and re-copy
```

### Poor Predictions
```
❌ "Model predicting wrong garbage type"
✅ Solutions:
   - Check lighting conditions (model trained in daylight)
   - Ensure object fills ~30-40% of frame
   - Verify labels.txt matches training classes
   - Retrain with more diverse garbage samples
   - Check confidence threshold settings
```

### Out of Memory Errors
```
❌ "MemoryError" or "Timeout"
✅ Solutions:
   - Use main_classifier.py (lighter than detector)
   - Reduce IMG_WIDTH/HEIGHT to 224×224
   - Disable logging temporarily
   - Restart the camera (power cycle)
   - Use int8 quantized model
```

### Slow Inference (>500ms per frame)
```
❌ "FPS too low (<2)"
✅ Solutions:
   - Reduce image size to 224×224
   - Use quantized model variant
   - Disable file I/O operations
   - Check camera frame rate settings
```

## Advanced Usage

### Custom Labels

Edit `/sd/labels.txt` to match your classes:
```
plastic
paper
metal
glass
organic
other
```

Ensure order matches training class indices.

### Batch Processing

Process multiple images:
```python
import sensor
from ml import classifier

for i in range(10):
    img = sensor.snapshot()
    predictions = classifier.predict(img)
    # Process prediction
```

### Integration with Other Sensors

Combine with RT1062's built-in sensors:
```python
import sensor
from ml import classifier

# Temperature-aware thresholding
temp = get_temperature()
if temp < 15:
    threshold = 0.6  # Higher threshold in cold
else:
    threshold = 0.5
```

## Model Retraining

To retrain the model with new data:

1. Collect new garbage images
2. Use `scripts/train_v4_improved.py` on host computer
3. Export to TFLite using `scripts/openmv_export.py`
4. Replace `garbage_classifier.tflite` on MicroSD

See [SETUP_GUIDE.md](docs/SETUP_GUIDE.md) for complete training instructions.

## Specifications

| Parameter | Value |
|-----------|-------|
| **Model Size** | 0.83 MB (TFLite) |
| **Input Size** | 320×320 RGB |
| **Inference Time** | 200-500ms (RT1062) |
| **Memory Usage** | ~4-6 MB |
| **Supported Frameworks** | TensorFlow Lite only |
| **Quantization** | INT8 (optional) |
| **Batch Size** | 1 (single image) |

## References

- [OpenMV Documentation](https://docs.openmv.io/)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [RT1062 Specifications](https://docs.openmv.io/openmvrt1062.html)

---

**Need help?** Check the troubleshooting section above or refer to [QUICK_REFERENCE.txt](docs/QUICK_REFERENCE.txt).
