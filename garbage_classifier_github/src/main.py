# OpenMV RT1062 - Mode launcher
# Reads run_mode.txt to decide which script to run.
# Hold the mode-switch button (P0) for SWITCH_HOLD_MS to toggle modes.

import uos
import gc

MODE_FILE     = "run_mode.txt"
DEFAULT_MODE  = "detector"
VALID_MODES   = ("detector", "classifier")

# Button pin for mode switching (change if your board uses a different pin).
BTN_PIN = "P0"


def _read_mode():
    try:
        with open(MODE_FILE, "r") as f:
            m = f.read().strip().lower()
        if m in VALID_MODES:
            return m
    except Exception:
        pass
    return DEFAULT_MODE


def _write_mode(mode):
    with open(MODE_FILE, "w") as f:
        f.write(mode)


def _script_exists(name):
    try:
        uos.stat(name)
        return True
    except Exception:
        return False


mode = _read_mode()
script = "main_detector.py" if mode == "detector" else "main_classifier.py"

print("=" * 40)
print("Garbage Classifier RT1062 V3")
print("Mode  :", mode)
print("Script:", script)
print("=" * 40)

if not _script_exists(script):
    raise Exception("Missing script: " + script)

gc.collect()
with open(script, "r") as _f:
    _code = _f.read()

exec(_code, globals())
