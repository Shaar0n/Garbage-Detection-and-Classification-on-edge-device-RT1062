Scripts: model export, quantization, dataset prep and testing

This file documents the model-related scripts included in `scripts/` and provides short usage examples.

Prerequisites
- Python 3.10+
- Recommended virtualenv: create and activate before running commands

Example:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r ..\requirements_inference_only.txt
```

Scripts

- `train_v4_improved.py` — Training script used to produce the models included in `models/`.
  - Usage: `python train_v4_improved.py --config path/to/config.yaml`
  - Notes: check top of script for training parameters (IMG_SIZE, BATCH_SIZE, EPOCHS).

- `compress_model_for_rt1062.py` — Compress / quantize a TFLite model for OpenMV RT1062.
  - Usage: `python compress_model_for_rt1062.py --input models/trained.tflite --output models/trained_quant.tflite`
  - Produces an int8 or optimized TFLite variant suitable for RT1062.

- `openmv_export.py` — Prepares files for OpenMV deployment (renames, small helpers, generates `main_openmv.py` if needed).
  - Usage: `python openmv_export.py --tflite models/trained_quant.tflite --labels models/labels.txt --out output/openmv_deployment`
  - Output: deployment folder with model and example `main.py` for `/sd/`.

- `convert_best_detector_to_tflite.py` — Convert detection checkpoints to TFLite detector models.
  - Usage: `python convert_best_detector_to_tflite.py --checkpoint checkpoints/NEW --output models/detector.tflite`

- `convert_detector_v2_to_tflite.py` — Alternative conversion pipeline for older detector formats.
  - Usage: `python convert_detector_v2_to_tflite.py --input path --output models/detector_v2.tflite`

- `prepare_dataset.py` — Prepare dataset (CSV → TFRecord or host training format).
  - Usage: `python prepare_dataset.py --input data/raw --output data/tfrecords`
  - Notes: Edit paths inside script for dataset naming conventions.

- `test_model.py` — Evaluate model on test set or single image; includes benchmarking options.
  - Usage examples:
    - Evaluate: `python test_model.py --model models/trained.tflite --mode evaluate`
    - Single image: `python test_model.py --model models/trained.tflite --image samples/test.jpg`
    - Benchmark: `python test_model.py --model models/trained.tflite --benchmark`

Tips
- Use `requirements_inference_only.txt` for inference and conversion tools; full training needs `requirements.txt`.
- If models grow larger than GitHub limits, use Git LFS for `.h5` or large `.tflite` files.

Contact
- If a script requires a missing dependency or environment variable, open `scripts/<script>.py` and check its top-level imports and docstring for details.
