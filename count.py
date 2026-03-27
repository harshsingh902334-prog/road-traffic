import cv2
import json
import os
from ultralytics import YOLO

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "yolov8n.pt"
CONF_THRES = 0.35

# Only count these vehicle classes
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# Input videos
videos = {
    "north": "north.mp4",
    "south": "south.mp4",
    "east":  "east.mp4",
    "west":  "west.mp4"
}

# Box colors
COLOR_BOX = (0, 255, 0)
COLOR_LINE = (0, 0, 255)
COLOR_COUNTED = (255, 0, 0)

# =========================================================
# LOAD MODEL
# =========================================================
model = YOLO(MODEL_PATH)

# =========================================================
# STORAGE
# =========================================================
final_counts = {}
detailed_report = {}

# =========================================================
# PROCESS EACH VIDEO
# =========================================================
for direction, path in videos.items():
    print(f"\nScanning {direction.upper()}...")

    if not os.path.exists(path):
        print(f"  [SKIP] File not found: {path}")
        final_counts[direction] = 0
        continue

    cap = cv2.VideoCapture(path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    # Output annotated video
    out_path = f"counted_{direction}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # ---------------------------
    # COUNTING LINE
    # ---------------------------
    # Horizontal line at 60% height
    line_y = int(height * 0.60)

    counted_ids = set()
    previous_positions = {}
    total_count = 0
    frame_num = 0

    # Optional class-wise counts
    class_counts = {
        "car": 0,
        "truck": 0,
        "bus": 0,
        "motorcycle": 0
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO tracking
        results = model.track(
            frame,
            persist=True,
            conf=CONF_THRES,
            verbose=False
        )[0]

        # Draw counting line
        cv2.line(frame, (0, line_y), (width, line_y), COLOR_LINE, 3)
        cv2.putText(frame, "COUNT LINE", (20, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_LINE, 2)

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id].lower()
                conf = float(box.conf[0])

                if conf < CONF_THRES:
                    continue

                if label not in VEHICLE_CLASSES:
                    continue

                if box.id is None:
                    continue

                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Vehicle center point
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Draw box
                color = COLOR_BOX
                if track_id in counted_ids:
                    color = COLOR_COUNTED

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)

                label_text = f"{label.upper()} ID:{track_id}"
                cv2.putText(frame, label_text, (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # ---------------------------
                # LINE CROSSING LOGIC
                # ---------------------------
                if track_id in previous_positions:
                    prev_cy = previous_positions[track_id]

                    # Vehicle crossed line from top to bottom or bottom to top
                    crossed = (
                        (prev_cy < line_y and cy >= line_y) or
                        (prev_cy > line_y and cy <= line_y)
                    )

                    if crossed and track_id not in counted_ids:
                        counted_ids.add(track_id)
                        total_count += 1
                        class_counts[label] += 1

                # Update last position
                previous_positions[track_id] = cy

        # Dashboard text
        cv2.rectangle(frame, (10, 10), (420, 80), (40, 40, 40), -1)
        cv2.putText(frame, f"{direction.upper()} COUNT: {total_count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Cars:{class_counts['car']}  Trucks:{class_counts['truck']}  Buses:{class_counts['bus']}  Bikes:{class_counts['motorcycle']}",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()

    final_counts[direction] = total_count
    detailed_report[direction] = {
        "total_vehicles": total_count,
        "cars": class_counts["car"],
        "trucks": class_counts["truck"],
        "buses": class_counts["bus"],
        "motorcycles": class_counts["motorcycle"],
        "frames_processed": frame_num,
        "output_video": out_path
    }

    print(f"  {direction}: {total_count} vehicles counted")
    print(f"    Cars: {class_counts['car']}, Trucks: {class_counts['truck']}, Buses: {class_counts['bus']}, Bikes: {class_counts['motorcycle']}")

# =========================================================
# SAVE JSON
# =========================================================
with open("counts.json", "w") as f:
    json.dump(final_counts, f, indent=4)

with open("counts_detailed.json", "w") as f:
    json.dump(detailed_report, f, indent=4)

print("\n" + "=" * 55)
print("FINAL COUNTS:", final_counts)
print("Saved: counts.json, counts_detailed.json")
print("=" * 55)