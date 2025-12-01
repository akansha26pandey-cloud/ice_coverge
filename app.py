import cv2
import numpy as np
import csv
from datetime import timedelta

ICE_THRESHOLD = 200        
MIN_AREA = 3000            
READY_T = 95               
RECHECK_T = 85             

INPUT_VIDEO = "videos/ice_coverage.mp4"
OUTPUT_VIDEO = "videos/annotated_output.mp4"
OUTPUT_CSV   = "videos/report.csv"

def classify(p):
    if p > READY_T:
        return "Ready"
    if p >= RECHECK_T:
        return "Needs Recheck"
    return "Reject"

def timestamp_from_frame(idx, fps):
    return str(timedelta(seconds=idx / fps))

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

csv_file = open(OUTPUT_CSV, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["frame", "timestamp", "crate_id", "coverage_pct", "classification"])

crate_id = 0
frame_id = 0

print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) != 4:
            continue  

        x, y, w2, h2 = cv2.boundingRect(approx)
        roi = frame[y:y+h2, x:x+w2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        _, mask = cv2.threshold(v, ICE_THRESHOLD, 255, cv2.THRESH_BINARY)

        ice_pixels = cv2.countNonZero(mask)
        total_pixels = w2 * h2
        pct = (ice_pixels / total_pixels) * 100

        cls = classify(pct)

        if cls == "Ready":
            color = (0, 255, 0)
        elif cls == "Needs Recheck":
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w2, y + h2), color, 2)
        cv2.putText(frame, f"{cls} ({pct:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if cls in ["Needs Recheck", "Reject"]:
            ts = timestamp_from_frame(frame_id, fps)
            writer.writerow([frame_id, ts, crate_id, pct, cls])

        crate_id += 1

    out.write(frame)
    frame_id += 1

cap.release()
out.release()
csv_file.close()

print("DONE!")
print("Annotated video ->", OUTPUT_VIDEO)
print("CSV report ->", OUTPUT_CSV)
