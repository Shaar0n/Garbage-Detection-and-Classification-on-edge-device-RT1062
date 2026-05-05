# OpenMV RT1062 - Garbage Detector V4-Fast
# Display : QVGA 320x240 (full UI canvas)
# Inference: center 96x96 crop - no resize, model gets exact size
# Hold BTN_PIN 1.5s to switch to classifier mode.

import sensor, time, ml, uos, gc

# ---- Config ---------------------------------------------------------------
MODEL_PATH      = "detector.tflite"
LABELS_PATH     = "labels.txt"
LOG_FILE        = "items_log.csv"
MODE_FILE       = "run_mode.txt"
BTN_PIN         = "P0"
CONF_THRESHOLD  = 0.55
STRICT_CONF     = 0.65
MIN_MARGIN      = 0.08
STRICT_MARGIN   = 0.12
SMOOTH          = 0.65
STABLE_REQUIRED = 5
MIN_BOX_RATIO   = 0.01
MAX_BOX_RATIO   = 0.98
LOG_COOLDOWN_MS = 3000
SWITCH_HOLD_MS  = 1500
INF_SIZE        = 96    # Model input (must match trained model)
# Center 96x96 crop from QVGA 320x240 - no resize needed
INF_ROI         = (112, 72, 96, 96)   # (x, y, w, h)


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
    needs_header = False
    try:
        needs_header = uos.stat(LOG_PATH)[6] == 0
    except Exception:
        needs_header = True
    if needs_header:
        try:
            with open(LOG_PATH, "w") as f:
                f.write("timestamp,mode,label,confidence,bbox_cx,bbox_cy,bbox_w,bbox_h\n")
        except Exception as e:
            print("CSV header err:", e)


def _log(label, conf, cx, cy, bw, bh):
    if LOG_PATH is None:
        return
    try:
        t  = time.localtime()
        ts = "%04d-%02d-%02d %02d:%02d:%02d" % (t[0],t[1],t[2],t[3],t[4],t[5])
        with open(LOG_PATH, "a") as f:
            f.write("%s,detector,%s,%.3f,%.3f,%.3f,%.3f,%.3f\n" % (
                    ts, label, conf, cx, cy, bw, bh))
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
    new = "classifier"
    try:
        with open(MODE_FILE, "r") as f:
            cur = f.read().strip().lower()
        new = "classifier" if cur == "detector" else "detector"
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


# ---- Helpers --------------------------------------------------------------
def _flatten(x):
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(_flatten(item))
        return out
    if hasattr(x, "flatten"):
        return x.flatten().tolist()
    return [float(x)]


def _dequant(arr):
    mn = arr[0]; mx = arr[0]
    for v in arr:
        if v < mn: mn = v
        if v > mx: mx = v
    if mx > 1.2 or mn < -0.1:
        return [max(0.0, min(1.0, (float(v)+128.0)/256.0)) for v in arr]
    return [float(v) for v in arr]


def _second_best(sc, best):
    s = 0.0
    for i in range(len(sc)):
        if i != best and sc[i] > s:
            s = sc[i]
    return s


# ---- Sensor ---------------------------------------------------------------
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)   # 320x240 for display
# No windowing - we manually crop for inference
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
print("Detector loaded | Classes:", labels)
print("Log path:", LOG_PATH)
_ensure_header()

# ---- Counts ---------------------------------------------------------------
counts          = {lbl: 0 for lbl in labels}
total_count     = 0
prev_stable_lbl = None   # Only updated when a NEW label becomes stable

# ---- Runtime state --------------------------------------------------------
clock          = time.clock()
frame_idx      = 0
smoothed_cls   = None
smoothed_box   = [0.5, 0.5, 0.4, 0.4]
stable_label   = "Unknown"
stable_conf    = 0.0
stable_count   = 0
uncertain_cnt  = 0
last_log_ms    = 0
btn_hold_start = None

# ---- UI layout (QVGA 320x240) --------------------------------------------
IW       = 320
IH       = 240
HDR_H    = 30    # Header bar height
FTR_H    = 38    # Footer bar height
FTR_Y    = IH - FTR_H       # 202
MAIN_Y   = HDR_H             # 30
MAIN_H   = FTR_Y - MAIN_Y   # 172

COL_GREEN  = (0,   220,  60)
COL_YELLOW = (255, 210,   0)
COL_ORANGE = (255, 130,   0)
COL_RED    = (255,  50,  50)
COL_GREY   = (160, 160, 160)
COL_CYAN   = (0,   210, 210)
COL_BLUE   = (60,  160, 255)
COL_BG     = (20,   20,  20)
COL_DIM    = (80,   80,  80)


def _draw_ui(img, disp_label, disp_conf, is_stable, is_conf, fps):
    # -- Header --
    img.draw_rectangle(0, 0, IW, HDR_H, color=COL_BG, fill=True)

    # Mode tag
    img.draw_string(4, 10, "[DET]", color=COL_BLUE, scale=1)

    # Label
    if is_stable:
        lc = COL_GREEN;  lt = disp_label
    elif is_conf:
        lc = COL_YELLOW; lt = "~" + disp_label
    else:
        lc = COL_GREY;   lt = "Scanning..."; disp_conf = 0.0
    img.draw_string(50, 8, lt, color=lc, scale=2)

    # Confidence
    if disp_conf > 0:
        img.draw_string(230, 8, "%d%%" % int(disp_conf*100), color=lc, scale=2)

    # FPS (top-right)
    fps_s = "%.0f fps" % fps
    img.draw_string(IW - len(fps_s)*6 - 2, 2, fps_s, color=COL_YELLOW, scale=1)

    # Guide box showing inference region
    img.draw_rectangle(INF_ROI[0], INF_ROI[1], INF_ROI[2], INF_ROI[3],
                       color=(60, 60, 60), thickness=1)

    # Divider
    img.draw_rectangle(0, HDR_H-1, IW, 1, color=(50,50,50), fill=True)

    # -- Footer --
    img.draw_rectangle(0, FTR_Y, IW, FTR_H, color=COL_BG, fill=True)
    img.draw_rectangle(0, FTR_Y, IW, 1, color=(50,50,50), fill=True)

    slot = IW // num_cls  # 64px per class (5 classes -> 320px)
    for i, lbl in enumerate(labels):
        cnt = counts.get(lbl, 0)
        xc  = i * slot
        col = COL_GREEN if cnt > 0 else COL_DIM
        # Highlight current stable detection
        if is_stable and lbl == disp_label:
            img.draw_rectangle(xc+1, FTR_Y+1, slot-2, FTR_H-2,
                               color=(0, 60, 20), fill=True)
        img.draw_string(xc+4,  FTR_Y+4,  lbl[:5],  color=COL_GREY, scale=1)
        img.draw_string(xc+4,  FTR_Y+17, str(cnt),  color=col,      scale=2)

    # Total (bottom-right)
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
        outputs = net.predict([inf_img])
        del inf_img
    except MemoryError:
        gc.collect()
        print("MemoryError during capture")
        continue
    except Exception as e:
        if frame_idx % 20 == 0:
            print("Predict err:", e)
        img.draw_string(4, MAIN_Y+4, "Predict err", color=COL_RED, scale=1)
        continue

    # -- Parse outputs --
    try:
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        cls_raw = box_raw = None
        for out in outputs:
            vals = _flatten(out)
            if len(vals) == num_cls:
                cls_raw = vals
            elif len(vals) == 4:
                box_raw = vals

        if cls_raw is None or box_raw is None:
            if frame_idx % 30 == 0:
                print("Output shape mismatch - cls:", cls_raw is not None, "box:", box_raw is not None)
            continue

        cls = _dequant(cls_raw)
        box = _dequant(box_raw)

        # Normalise
        s = sum(cls)
        if s > 1e-6:
            cls = [v/s for v in cls]

        # EMA
        if smoothed_cls is None:
            smoothed_cls = cls[:]
        else:
            smoothed_cls = [SMOOTH*o+(1-SMOOTH)*n for o,n in zip(smoothed_cls, cls)]
        smoothed_box = [SMOOTH*o+(1-SMOOTH)*n for o,n in zip(smoothed_box, box)]

        # Best class (no max(default=))
        best_idx = 0
        for i in range(1, num_cls):
            if smoothed_cls[i] > smoothed_cls[best_idx]:
                best_idx = i
        best_sc = smoothed_cls[best_idx]
        second  = _second_best(smoothed_cls, best_idx)
        margin  = best_sc - second
        label   = labels[best_idx] if best_idx < num_cls else "Unknown"

        # Bbox projected onto full main display area (HDR_H..FTR_Y x 0..IW)
        # Model outputs normalized coords within the 96x96 crop; we scale
        # those proportionally to the full visible canvas so the box is large.
        cx, cy, bw, bh = smoothed_box
        cx = max(0.0, min(1.0, cx)); cy = max(0.0, min(1.0, cy))
        bw = max(0.05, min(1.0, bw)); bh = max(0.05, min(1.0, bh))

        x1 = max(0,       int((cx - bw*0.5) * IW))
        y1 = max(HDR_H,   int(HDR_H + (cy - bh*0.5) * MAIN_H))
        x2 = min(IW-1,    int((cx + bw*0.5) * IW))
        y2 = min(FTR_Y-1, int(HDR_H + (cy + bh*0.5) * MAIN_H))
        pw = max(8, x2-x1); ph = max(8, y2-y1)

        box_area  = (pw*ph) / float(IW*MAIN_H)
        geo_ok    = MIN_BOX_RATIO <= box_area <= MAX_BOX_RATIO
        confident = best_sc >= CONF_THRESHOLD and margin >= MIN_MARGIN
        strong    = best_sc >= STRICT_CONF    and margin >= STRICT_MARGIN

        # Stability
        if strong and geo_ok:
            if label == stable_label:
                stable_count += 1
            else:
                stable_label = label
                stable_conf  = best_sc
                stable_count = 1
            if best_sc > stable_conf:
                stable_conf = best_sc
            uncertain_cnt = 0
        else:
            uncertain_cnt += 1
            if uncertain_cnt >= 8 and confident and geo_ok:
                stable_label  = label
                stable_conf   = best_sc
                stable_count  = STABLE_REQUIRED
                uncertain_cnt = 0
            else:
                stable_count = 0

        is_stable = stable_count >= STABLE_REQUIRED
        is_conf   = confident and geo_ok

        # Bbox colour
        if is_stable:   box_col = COL_GREEN
        elif is_conf:   box_col = COL_ORANGE
        else:           box_col = COL_DIM
        img.draw_rectangle(x1, y1, pw, ph, color=box_col, thickness=2)

        # -- Counting (increments only when label CHANGES) --
        # prev_stable_lbl is NOT reset on uncertain frames, preventing re-counting
        if is_stable and stable_label != prev_stable_lbl:
            counts[stable_label] = counts.get(stable_label, 0) + 1
            total_count += 1
            prev_stable_lbl = stable_label

        # CSV log (throttled, separate from counting)
        now_ms = time.ticks_ms()
        if is_stable and time.ticks_diff(now_ms, last_log_ms) >= LOG_COOLDOWN_MS:
            _log(stable_label, stable_conf, cx, cy, bw, bh)
            last_log_ms = now_ms

        # -- UI --
        _draw_ui(img,
                 stable_label if is_stable else label,
                 stable_conf  if is_stable else best_sc,
                 is_stable, is_conf, clock.fps())

        if frame_idx % 30 == 0:
            print("[DET] %s %.2f m=%.2f stable=%d tot=%d fps=%.1f" % (
                label, best_sc, margin, stable_count, total_count, clock.fps()))

    except MemoryError:
        gc.collect()
        print("MemoryError in parse")
    except Exception as e:
        if frame_idx % 20 == 0:
            print("Loop err:", e)

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
