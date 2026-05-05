# OpenMV RT1062 - Garbage Classifier V4-Fast
# Display : QVGA 320x240
# Inference: 96x96 center-crop (manual crop, no resize)
# Hold BTN_PIN 1.5s to switch to detector mode.

import sensor, time, ml, uos, gc

# ---- Config ---------------------------------------------------------------
MODEL_PATH      = "trained.tflite"
LABELS_PATH     = "labels.txt"
LOG_FILE        = "items_log.csv"
MODE_FILE       = "run_mode.txt"
BTN_PIN         = "P0"
CONF_THRESHOLD  = 0.62
STABLE_REQUIRED = 5
LOG_COOLDOWN_MS = 3000
SWITCH_HOLD_MS  = 1500
INF_SIZE        = 96
INF_ROI         = (112, 72, 96, 96)   # Center 96x96 crop from QVGA - no resize needed


# ---- Writable log path ----------------------------------------------------
def _find_log_path():
    for p in (LOG_FILE, "/sd/" + LOG_FILE, "/flash/" + LOG_FILE):
        try:
            with open(p, "a") as f:
                f.write("")
            return p
        except Exception:
            pass
    return None

LOG_PATH = _find_log_path()


# ---- CSV ------------------------------------------------------------------
def _ensure_header():
    if LOG_PATH is None:
        return
    needs = False
    try:
        needs = uos.stat(LOG_PATH)[6] == 0
    except Exception:
        needs = True
    if needs:
        try:
            with open(LOG_PATH, "w") as f:
                f.write("timestamp,mode,label,confidence\n")
        except Exception as e:
            print("CSV header err:", e)


def _log(label, conf):
    if LOG_PATH is None:
        return
    try:
        t  = time.localtime()
        ts = "%04d-%02d-%02d %02d:%02d:%02d" % (t[0],t[1],t[2],t[3],t[4],t[5])
        with open(LOG_PATH, "a") as f:
            f.write("%s,classifier,%s,%.3f\n" % (ts, label, conf))
    except Exception as e:
        print("CSV err:", e)


# ---- Button ---------------------------------------------------------------
_btn = None
try:
    from pyb import Pin
    _btn = Pin(BTN_PIN, Pin.IN, Pin.PULL_UP)
except Exception:
    pass

def _btn_down():
    return _btn is not None and _btn.value() == 0


# ---- Mode switch ----------------------------------------------------------
def _switch_mode():
    new = "detector"
    try:
        with open(MODE_FILE, "r") as f:
            cur = f.read().strip().lower()
        new = "detector" if cur == "classifier" else "classifier"
    except Exception:
        pass
    try:
        with open(MODE_FILE, "w") as f:
            f.write(new)
    except Exception:
        pass
    print("Switching ->", new)
    time.sleep_ms(300)
    try:
        import pyb; pyb.reset()
    except Exception:
        import machine; machine.reset()


# ---- Sensor ---------------------------------------------------------------
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)   # 320x240
sensor.skip_frames(time=2000)

# ---- Model ----------------------------------------------------------------
try:
    _sz = uos.stat(MODEL_PATH)[6]
    net = ml.Model(MODEL_PATH, load_to_fb=(_sz > gc.mem_free() - 64*1024))
except Exception as e:
    raise Exception("Load failed: " + MODEL_PATH + " " + str(e))

try:
    labels = [ln.strip() for ln in open(LABELS_PATH) if ln.strip()]
except Exception as e:
    raise Exception("Load failed: " + LABELS_PATH + " " + str(e))

num_cls = len(labels)
print("Classifier loaded | Classes:", labels)
print("Log path:", LOG_PATH)
_ensure_header()

# ---- Counts ---------------------------------------------------------------
counts          = {lbl: 0 for lbl in labels}
total_count     = 0
prev_stable_lbl = None   # Not reset on uncertain frames

# ---- Runtime state --------------------------------------------------------
clock          = time.clock()
frame_idx      = 0
last_label     = labels[0] if labels else "Unknown"
last_conf      = 0.0
stable_label   = labels[0] if labels else "Unknown"
stable_count   = 0
last_log_ms    = 0
btn_hold_start = None

# ---- UI layout (QVGA 320x240) --------------------------------------------
IW      = 320
IH      = 240
HDR_H   = 30
FTR_H   = 38
FTR_Y   = IH - FTR_H    # 202
MAIN_Y  = HDR_H          # 30

COL_GREEN  = (0,   220,  60)
COL_YELLOW = (255, 210,   0)
COL_ORANGE = (255, 130,   0)
COL_GREY   = (160, 160, 160)
COL_CYAN   = (0,   210, 210)
COL_PURPLE = (170,  60, 255)
COL_BG     = (20,   20,  20)
COL_DIM    = (80,   80,  80)
COL_RED    = (255,  50,  50)


def _draw_ui(img, label, conf, is_stable, fps):
    # -- Header --
    img.draw_rectangle(0, 0, IW, HDR_H, color=COL_BG, fill=True)
    img.draw_string(4, 10, "[CLS]", color=COL_PURPLE, scale=1)

    if is_stable:
        lc = COL_GREEN;  lt = label
    elif conf >= CONF_THRESHOLD * 0.75:
        lc = COL_YELLOW; lt = "~" + label
    else:
        lc = COL_GREY;   lt = "Scanning..."; conf = 0.0

    img.draw_string(50, 8, lt, color=lc, scale=2)
    if conf > 0:
        img.draw_string(232, 8, "%d%%" % int(conf*100), color=lc, scale=2)

    fps_s = "%.0f fps" % fps
    img.draw_string(IW - len(fps_s)*6 - 2, 2, fps_s, color=COL_YELLOW, scale=1)
    img.draw_rectangle(0, HDR_H-1, IW, 1, color=(50,50,50), fill=True)

    # Guide box showing inference region (center 96x96)
    img.draw_rectangle(INF_ROI[0], INF_ROI[1], INF_ROI[2], INF_ROI[3],
                       color=(60, 60, 60), thickness=1)

    # -- Confidence bar (just below header) --
    bar_w = int(conf * (IW - 4))
    if bar_w > 0:
        bc = COL_GREEN if is_stable else COL_YELLOW
        img.draw_rectangle(2, HDR_H+1, bar_w, 5, color=bc, fill=True)
    img.draw_rectangle(2, HDR_H+1, IW-4, 5, color=(50,50,50))

    # -- Footer --
    img.draw_rectangle(0, FTR_Y, IW, FTR_H, color=COL_BG, fill=True)
    img.draw_rectangle(0, FTR_Y, IW, 1, color=(50,50,50), fill=True)

    slot = IW // num_cls   # 64px per class
    for i, lbl in enumerate(labels):
        cnt = counts.get(lbl, 0)
        xc  = i * slot
        col = COL_GREEN if cnt > 0 else COL_DIM
        if is_stable and lbl == label:
            img.draw_rectangle(xc+1, FTR_Y+1, slot-2, FTR_H-2,
                               color=(50, 0, 80), fill=True)
        img.draw_string(xc+4, FTR_Y+4,  lbl[:5], color=COL_GREY, scale=1)
        img.draw_string(xc+4, FTR_Y+17, str(cnt), color=col,     scale=2)

    tot = "TOT:%d" % total_count
    img.draw_string(IW - len(tot)*6 - 4, FTR_Y+24, tot, color=COL_CYAN, scale=1)


# ---- Main loop ------------------------------------------------------------
while True:
    clock.tick()
    frame_idx += 1
    img = sensor.snapshot()

    # -- Inference: exact 96x96 center crop, no resize --
    try:
        inf_img = img.copy(roi=INF_ROI)  # already 96x96
        preds = net.predict([inf_img])[0].flatten().tolist()
        del inf_img
    except MemoryError:
        gc.collect()
        print("MemoryError")
        continue
    except Exception as e:
        if frame_idx % 20 == 0:
            print("Predict err:", e)
        img.draw_string(4, MAIN_Y+4, "Predict err", color=COL_RED, scale=1)
        continue

    # Dequantize if INT8 output
    mn = preds[0]; mx = preds[0]
    for v in preds:
        if v < mn: mn = v
        if v > mx: mx = v
    if mx > 1.2 or mn < -0.1:
        preds = [max(0.0, min(1.0, (float(v)+128.0)/256.0)) for v in preds]

    best_idx = 0
    for i in range(1, len(preds)):
        if preds[i] > preds[best_idx]:
            best_idx = i

    last_conf  = float(preds[best_idx])
    last_label = labels[best_idx] if best_idx < num_cls else "Unknown"

    # Stability
    if last_conf >= CONF_THRESHOLD:
        if last_label == stable_label:
            stable_count += 1
        else:
            stable_label = last_label
            stable_count = 1
    else:
        stable_count = 0

    is_stable = stable_count >= STABLE_REQUIRED

    # -- Counting (only when label changes, NOT reset on uncertain frames) --
    if is_stable and stable_label != prev_stable_lbl:
        counts[stable_label] = counts.get(stable_label, 0) + 1
        total_count += 1
        prev_stable_lbl = stable_label

    # CSV log (throttled)
    now_ms = time.ticks_ms()
    if is_stable and time.ticks_diff(now_ms, last_log_ms) >= LOG_COOLDOWN_MS:
        _log(stable_label, last_conf)
        last_log_ms = now_ms

    # -- UI --
    _draw_ui(img, last_label, last_conf, is_stable, clock.fps())

    if frame_idx % 30 == 0:
        print("[CLS] %s %.2f stable=%d tot=%d fps=%.1f" % (
            last_label, last_conf, stable_count, total_count, clock.fps()))

    # -- Button --
    if _btn_down():
        if btn_hold_start is None:
            btn_hold_start = time.ticks_ms()
        elif time.ticks_diff(time.ticks_ms(), btn_hold_start) >= SWITCH_HOLD_MS:
            _switch_mode()
    else:
        btn_hold_start = None

    if frame_idx % 60 == 0:
        gc.collect()
