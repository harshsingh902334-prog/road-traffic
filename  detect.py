import cv2
import json
import os
import math
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH          = "yolov8n.pt"
CONF_THRES          = 0.35
MIN_EMERGENCY_FRAMES = 5   # ambulance confirmed after N frames
MIN_POLICE_FRAMES    = 4
PIXEL_TO_KMPH        = 0.18
MAX_TRACK_HISTORY    = 10

# ─────────────────────────────────────────
# FIX: north.mp4 missing → skip gracefully
# Add "north" back when you have the file
# ─────────────────────────────────────────
VIDEOS = {
    "north": "north.mp4",
    "south": "south.mp4",
    "east":  "east.mp4",
    "west":  "west.mp4"
}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

# Colors (BGR)
COLOR_AMBULANCE = (0, 0, 255)
COLOR_POLICE    = (255, 140, 0)
COLOR_VEHICLE   = (0, 255, 0)
COLOR_TEXT_BG   = (40, 40, 40)

# =========================================================
# LOAD MODEL — with clear error message
# =========================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"[ERROR] Model not found: {MODEL_PATH}\n"
        f"  Download: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    )

model = YOLO(MODEL_PATH)
print(f"[OK] Model loaded: {MODEL_PATH}")

# =========================================================
# HELPERS
# =========================================================
def clamp_box(x1, y1, x2, y2, w, h):
    return (
        max(0, min(x1, w - 1)),
        max(0, min(y1, h - 1)),
        max(0, min(x2, w - 1)),
        max(0, min(y2, h - 1))
    )


def get_crop(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def is_white_vehicle(frame, x1, y1, x2, y2):
    """Returns True if >50% of vehicle crop is near-white (ambulance clue)."""
    crop = get_crop(frame, x1, y1, x2, y2)
    if crop is None or crop.size == 0:
        return False
    hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 55, 255]))
    ratio = cv2.countNonZero(mask) / (crop.shape[0] * crop.shape[1])
    return ratio > 0.50


def detect_red_blue_flash(frame, x1, y1, x2, y2):
    """
    Checks top 35% of vehicle for red/blue beacon colors.
    Returns (has_red, has_blue).
    """
    crop = get_crop(frame, x1, y1, x2, y2)
    if crop is None or crop.size == 0:
        return False, False

    h   = crop.shape[0]
    top = crop[:max(1, int(h * 0.35)), :]

    hsv  = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    area = top.shape[0] * top.shape[1]
    if area == 0:
        return False, False

    r1 = cv2.inRange(hsv, np.array([0,   80, 80]), np.array([10,  255, 255]))
    r2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    red_mask  = cv2.bitwise_or(r1, r2)
    blue_mask = cv2.inRange(hsv, np.array([95, 80, 80]), np.array([130, 255, 255]))

    has_red  = cv2.countNonZero(red_mask)  / area > 0.01
    has_blue = cv2.countNonZero(blue_mask) / area > 0.01
    return has_red, has_blue


def estimate_speed(track_history, fps):
    """Pixel-based speed estimate (approximate). Returns km/h."""
    if len(track_history) < 2:
        return 0.0
    (x1, y1), f1 = track_history[0]
    (x2, y2), f2 = track_history[-1]
    frames = max(1, f2 - f1)
    px_dist = math.hypot(x2 - x1, y2 - y1)
    return round((px_dist / frames) * fps * PIXEL_TO_KMPH, 1)


def draw_label(frame, x1, y1, text, color):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    y_top = max(0, y1 - th - 10)
    cv2.rectangle(frame, (x1, y_top), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, text, (x1 + 4, y1 - 6), font, scale, (0, 0, 0), thick)


# =========================================================
# RESULT STORAGE
# =========================================================
emergency_dirs = []
full_report    = {}

# =========================================================
# PROCESS EACH DIRECTION
# =========================================================
for direction, path in VIDEOS.items():

    # ─── GRACEFUL SKIP if file missing ────────────────────
    if not os.path.exists(path):
        print(f"\n[SKIP] {path} not found — skipping {direction.upper()}")
        full_report[direction] = {
            "video": path,
            "skipped": True,
            "reason": "File not found",
            "ambulance_detected": False,
            "police_detected": False,
            "unique_vehicles": 0,
            "unique_ambulances": 0,
            "unique_police": 0,
            "frames_processed": 0
        }
        continue
    # ──────────────────────────────────────────────────────

    print(f"\n[Processing] {direction.upper()} ← {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {path}")
        full_report[direction] = {
            "video": path, "skipped": True, "reason": "Cannot open file",
            "ambulance_detected": False, "police_detected": False,
            "unique_vehicles": 0, "unique_ambulances": 0, "unique_police": 0,
            "frames_processed": 0
        }
        continue

    fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Resolution: {width}×{height}  FPS: {fps:.1f}  Frames: {total_frames}")

    out_path = f"output_{direction}.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    out      = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count      = 0
    ambulance_found  = False
    police_found     = False

    track_positions     = defaultdict(lambda: deque(maxlen=MAX_TRACK_HISTORY))
    emergency_counter   = defaultdict(int)
    police_counter      = defaultdict(int)

    unique_ambulance_ids = set()
    unique_police_ids    = set()
    unique_vehicle_ids   = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── YOLO tracking ──────────────────────────────────
        try:
            results = model.track(frame, persist=True, verbose=False, conf=CONF_THRES)[0]
        except Exception as e:
            print(f"  [WARN] Frame {frame_count} tracking error: {e}")
            out.write(frame)
            frame_count += 1
            continue
        # ───────────────────────────────────────────────────

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = model.names[cls_id].lower()

                if conf < CONF_THRES or label not in VEHICLE_CLASSES:
                    continue
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                unique_vehicle_ids.add(track_id)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                track_positions[track_id].append(((cx, cy), frame_count))

                # ── Heuristics ──────────────────────────────
                white_hint        = is_white_vehicle(frame, x1, y1, x2, y2)
                has_red, has_blue = detect_red_blue_flash(frame, x1, y1, x2, y2)

                is_ambulance = (
                    label in {"truck", "bus", "car"} and
                    white_hint and (has_red or has_blue)
                )
                is_police = (
                    label in {"car", "truck", "motorcycle"} and
                    has_red and has_blue
                )

                if is_ambulance:
                    emergency_counter[track_id] += 1
                if is_police:
                    police_counter[track_id] += 1

                confirmed_ambulance = emergency_counter[track_id] >= MIN_EMERGENCY_FRAMES
                confirmed_police    = police_counter[track_id]    >= MIN_POLICE_FRAMES

                if confirmed_ambulance:
                    color = COLOR_AMBULANCE
                    tag   = "AMBULANCE"
                    ambulance_found = True
                    unique_ambulance_ids.add(track_id)
                elif confirmed_police:
                    color = COLOR_POLICE
                    tag   = "POLICE"
                    police_found = True
                    unique_police_ids.add(track_id)
                else:
                    color = COLOR_VEHICLE
                    tag   = label.upper()

                speed      = estimate_speed(track_positions[track_id], fps)
                speed_text = f" {speed}km/h" if speed > 0 else ""

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, x1, y1, f"ID:{track_id} {tag}{speed_text}", color)
                cv2.circle(frame, (cx, cy), 4, color, -1)

        # ── Status overlay ──────────────────────────────────
        status = (
            f"{direction.upper()} | "
            f"Vehicles:{len(unique_vehicle_ids)} | "
            f"Ambulance:{'YES' if ambulance_found else 'NO'} | "
            f"Police:{'YES' if police_found else 'NO'}"
        )
        cv2.rectangle(frame, (10, 10), (min(width - 10, 700), 45), COLOR_TEXT_BG, -1)
        cv2.putText(frame, status, (18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        out.write(frame)
        frame_count += 1

        # Progress every 100 frames
        if frame_count % 100 == 0:
            print(f"  Frame {frame_count}/{total_frames} ...", end='\r')

    cap.release()
    out.release()

    if ambulance_found:
        emergency_dirs.append(direction)

    full_report[direction] = {
        "video":             path,
        "output_video":      out_path,
        "frames_processed":  frame_count,
        "ambulance_detected": ambulance_found,
        "police_detected":    police_found,
        "unique_vehicles":    len(unique_vehicle_ids),
        "unique_ambulances":  len(unique_ambulance_ids),
        "unique_police":      len(unique_police_ids),
        "skipped":            False
    }

    print(f"\n  Done → {out_path}")
    print(f"  Ambulance: {'YES 🚨' if ambulance_found else 'No'}")
    print(f"  Police:    {'YES 🚔' if police_found else 'No'}")
    print(f"  Frames:    {frame_count}")
    print(f"  Vehicles:  {len(unique_vehicle_ids)}")

# =========================================================
# SAVE emergency.json
# =========================================================
result = {
    "has_emergency":        len(emergency_dirs) > 0,
    "emergency_directions": emergency_dirs,
    "report":               full_report
}

with open("emergency.json", "w") as f:
    json.dump(result, f, indent=4)

print("\n" + "=" * 60)
print("ALL VIDEOS PROCESSED!")
print(f"Emergency lanes : {emergency_dirs if emergency_dirs else 'None'}")
skipped = [d for d, r in full_report.items() if r.get("skipped")]
if skipped:
    print(f"Skipped (missing): {skipped}")
print("Saved : emergency.json")
print("=" * 60)