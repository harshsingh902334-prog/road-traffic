import cv2
import json
import os
import math
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

MODEL_PATH          = "yolov8n.pt"
CONF_THRES          = 0.35
MIN_EMERGENCY_FRAMES = 5
MIN_POLICE_FRAMES    = 4
PIXEL_TO_KMPH        = 0.18
MAX_TRACK_HISTORY    = 10

VIDEOS = {
    "north": "north.mp4",
    "south": "south.mp4",
    "east":  "east.mp4",
    "west":  "west.mp4"
}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
COLOR_AMBULANCE = (0, 0, 255)
COLOR_POLICE    = (255, 140, 0)
COLOR_VEHICLE   = (0, 255, 0)
COLOR_TEXT_BG   = (40, 40, 40)

model = YOLO(MODEL_PATH)
print(f"[OK] Model loaded: {MODEL_PATH}")

def clamp_box(x1, y1, x2, y2, w, h):
    return (max(0,min(x1,w-1)),max(0,min(y1,h-1)),max(0,min(x2,w-1)),max(0,min(y2,h-1)))

def get_crop(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1,y1,x2,y2 = clamp_box(x1,y1,x2,y2,w,h)
    if x2<=x1 or y2<=y1: return None
    return frame[y1:y2, x1:x2]

def is_white_vehicle(frame, x1, y1, x2, y2):
    crop = get_crop(frame, x1, y1, x2, y2)
    if crop is None or crop.size==0: return False
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,0,180]), np.array([180,55,255]))
    return cv2.countNonZero(mask)/(crop.shape[0]*crop.shape[1]) > 0.50

def detect_red_blue_flash(frame, x1, y1, x2, y2):
    crop = get_crop(frame, x1, y1, x2, y2)
    if crop is None or crop.size==0: return False, False
    h   = crop.shape[0]
    top = crop[:max(1,int(h*0.35)), :]
    hsv  = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    area = top.shape[0]*top.shape[1]
    if area==0: return False, False
    r1 = cv2.inRange(hsv, np.array([0,80,80]),   np.array([10,255,255]))
    r2 = cv2.inRange(hsv, np.array([170,80,80]), np.array([180,255,255]))
    red_mask  = cv2.bitwise_or(r1,r2)
    blue_mask = cv2.inRange(hsv, np.array([95,80,80]), np.array([130,255,255]))
    return cv2.countNonZero(red_mask)/area>0.01, cv2.countNonZero(blue_mask)/area>0.01

def estimate_speed(track_history, fps):
    if len(track_history)<2: return 0.0
    (x1,y1),f1 = track_history[0]
    (x2,y2),f2 = track_history[-1]
    return round((math.hypot(x2-x1,y2-y1)/max(1,f2-f1))*fps*PIXEL_TO_KMPH, 1)

def draw_label(frame, x1, y1, text, color):
    font=cv2.FONT_HERSHEY_SIMPLEX; scale=0.52; thick=2
    (tw,th),_=cv2.getTextSize(text,font,scale,thick)
    y_top=max(0,y1-th-10)
    cv2.rectangle(frame,(x1,y_top),(x1+tw+8,y1),color,-1)
    cv2.putText(frame,text,(x1+4,y1-6),font,scale,(0,0,0),thick)

emergency_dirs = []
full_report    = {}

for direction, path in VIDEOS.items():
    if not os.path.exists(path):
        print(f"\n[SKIP] {path} not found")
        full_report[direction] = {"video":path,"skipped":True,"ambulance_detected":False,"police_detected":False,"unique_vehicles":0,"unique_ambulances":0,"unique_police":0,"frames_processed":0}
        continue

    print(f"\n[Processing] {direction.upper()} <- {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {path}")
        continue

    fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {width}x{height} @ {fps:.1f}fps")

    out = cv2.VideoWriter(f"output_{direction}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    frame_count=0; ambulance_found=False; police_found=False
    track_positions   = defaultdict(lambda: deque(maxlen=MAX_TRACK_HISTORY))
    emergency_counter = defaultdict(int)
    police_counter    = defaultdict(int)
    unique_ambulance_ids=set(); unique_police_ids=set(); unique_vehicle_ids=set()

    while True:
        ret, frame = cap.read()
        if not ret: break
        try:
            results = model.track(frame, persist=True, verbose=False, conf=CONF_THRES)[0]
        except:
            out.write(frame); frame_count+=1; continue

        if results.boxes is not None and len(results.boxes)>0:
            for box in results.boxes:
                cls_id=int(box.cls[0]); conf=float(box.conf[0])
                label=model.names[cls_id].lower()
                if conf<CONF_THRES or label not in VEHICLE_CLASSES or box.id is None: continue
                track_id=int(box.id[0])
                x1,y1,x2,y2=map(int,box.xyxy[0])
                unique_vehicle_ids.add(track_id)
                cx=(x1+x2)//2; cy=(y1+y2)//2
                track_positions[track_id].append(((cx,cy),frame_count))
                white_hint=is_white_vehicle(frame,x1,y1,x2,y2)
                has_red,has_blue=detect_red_blue_flash(frame,x1,y1,x2,y2)
                if label in {"truck","bus","car"} and white_hint and (has_red or has_blue): emergency_counter[track_id]+=1
                if label in {"car","truck","motorcycle"} and has_red and has_blue: police_counter[track_id]+=1
                confirmed_ambulance=emergency_counter[track_id]>=MIN_EMERGENCY_FRAMES
                confirmed_police=police_counter[track_id]>=MIN_POLICE_FRAMES
                if confirmed_ambulance:
                    color=COLOR_AMBULANCE; tag="AMBULANCE"; ambulance_found=True; unique_ambulance_ids.add(track_id)
                elif confirmed_police:
                    color=COLOR_POLICE; tag="POLICE"; police_found=True; unique_police_ids.add(track_id)
                else:
                    color=COLOR_VEHICLE; tag=label.upper()
                speed=estimate_speed(track_positions[track_id],fps)
                speed_text=f" {speed}km/h" if speed>0 else ""
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                draw_label(frame,x1,y1,f"ID:{track_id} {tag}{speed_text}",color)
                cv2.circle(frame,(cx,cy),4,color,-1)

        status=f"{direction.upper()} | Vehicles:{len(unique_vehicle_ids)} | Ambulance:{'YES' if ambulance_found else 'NO'} | Police:{'YES' if police_found else 'NO'}"
        cv2.rectangle(frame,(10,10),(min(width-10,700),45),COLOR_TEXT_BG,-1)
        cv2.putText(frame,status,(18,34),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),2)
        out.write(frame); frame_count+=1

    cap.release(); out.release()
    if ambulance_found: emergency_dirs.append(direction)
    full_report[direction]={"video":path,"output_video":f"output_{direction}.mp4","frames_processed":frame_count,"ambulance_detected":ambulance_found,"police_detected":police_found,"unique_vehicles":len(unique_vehicle_ids),"unique_ambulances":len(unique_ambulance_ids),"unique_police":len(unique_police_ids),"skipped":False}
    print(f"  Done! Ambulance:{'YES' if ambulance_found else 'No'} | Police:{'YES' if police_found else 'No'} | Vehicles:{len(unique_vehicle_ids)}")

with open("emergency.json","w") as f:
    json.dump({"has_emergency":len(emergency_dirs)>0,"emergency_directions":emergency_dirs,"report":full_report},f,indent=4)

print("\n" + "="*60)
print("ALL DONE!")
print(f"Emergency lanes: {emergency_dirs if emergency_dirs else 'None'}")
print("Saved: emergency.json")
print("="*60)
