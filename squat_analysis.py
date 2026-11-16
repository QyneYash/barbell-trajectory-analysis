import argparse
import csv
import math
import os
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO


SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

BAR_WORDS = ("barbell", "weight", "plate", "bar", "shaft", "pole")
PLATE_WORDS = ("plate", "weight")
POLE_WORDS = ("pole", "shaft")  # Do not include "bar": it matches "barbell".
BARBELL_WORDS = ("barbell",)


def class_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(int(class_id), class_id)).lower()
    if isinstance(names, (list, tuple)) and 0 <= int(class_id) < len(names):
        return str(names[int(class_id)]).lower()
    return str(class_id).lower()


def get_detection_rows(result, names, conf_threshold):
    rows = []
    if result.boxes is None:
        return rows
    for box in result.boxes:
        conf = float(box.conf[0].item())
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls[0].item())
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
        rows.append({
            "name": class_name(names, cls_id),
            "conf": conf,
            "box": xyxy
        })
    return rows


def center_of(box):
    x1, y1, x2, y2 = box
    return (float((x1 + x2) / 2), float((y1 + y2) / 2))


def clamp_box(box, width, height):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    return [x1, y1, x2, y2]


def detect_barbell_geometry(rows, width, height):
    """
    Return left plate centre, right plate centre, shaft endpoints, pole box,
    source label. All detections are automatic.
    """
    plate_rows = [r for r in rows if any(w in r["name"] for w in PLATE_WORDS)]
    pole_rows = [r for r in rows if any(w in r["name"] for w in POLE_WORDS)]
    barbell_rows = [r for r in rows if any(w in r["name"] for w in BARBELL_WORDS)]

    # A "bar" class can be a shaft or an entire barbell. Prefer explicit plates.
    left_center = right_center = None
    shaft = None
    pole_box = None
    source = "NOT DETECTED"

    if len(plate_rows) >= 2:
        # Prefer two plate detections with the largest horizontal separation.
        plate_rows = sorted(plate_rows, key=lambda r: center_of(r["box"])[0])
        left_row, right_row = plate_rows[0], plate_rows[-1]
        left_center = center_of(left_row["box"])
        right_center = center_of(right_row["box"])
        shaft = (left_center, right_center)
        source = "TWO PLATES DETECTED"

    elif barbell_rows:
        best = max(barbell_rows, key=lambda r: r["conf"])
        x1, y1, x2, y2 = clamp_box(best["box"], width, height)
        bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
        cy = (y1 + y2) / 2.0
        # End-centres are estimates inside the automatically detected full barbell box.
        left_center = (x1 + 0.08 * bw, cy)
        right_center = (x2 - 0.08 * bw, cy)
        shaft = (left_center, right_center)
        source = "BARBELL BOX / ENDS ESTIMATED"

    elif len(plate_rows) == 1:
        # One visible plate: report one side; do not fabricate the hidden plate.
        only = max(plate_rows, key=lambda r: r["conf"])
        c = center_of(only["box"])
        if c[0] < width / 2:
            left_center = c
        else:
            right_center = c
        source = "ONE PLATE VISIBLE"

    if pole_rows:
        pole = max(pole_rows, key=lambda r: r["conf"])
        pole_box = clamp_box(pole["box"], width, height)
        px1, py1, px2, py2 = pole_box
        # Shaft centreline follows the pole detection box's long axis.
        if (px2 - px1) >= (py2 - py1):
            cy = (py1 + py2) / 2
            shaft = ((px1, cy), (px2, cy))
        else:
            cx = (px1 + px2) / 2
            shaft = ((cx, py1), (cx, py2))
        source = source + " + POLE / SHAFT BOX"

    elif shaft is not None:
        # The pole is the shaft joining the two detected/estimated barbell ends.
        source = source + " + POLE INFERRED"

    return left_center, right_center, shaft, pole_box, source



def draw_object_boxes(frame, rows, width, height):
    """Draw actual detector boxes, keeping barbell separate from pole."""
    plate_boxes, full_barbell_boxes, pole_boxes = [], [], []
    for row in rows:
        name = row["name"].lower().replace("_", " ").replace("-", " ")
        tokens = set(name.split())
        x1, y1, x2, y2 = [int(round(v)) for v in clamp_box(row["box"], width, height)]
        if "plate" in name or "weight" in name:
            color, label = (0, 220, 80), f"PLATE {row['conf']:.2f}"
            plate_boxes.append([x1, y1, x2, y2])
        elif "barbell" in name:
            color, label = (255, 170, 0), f"BARBELL {row['conf']:.2f}"
            full_barbell_boxes.append([x1, y1, x2, y2])
        elif "pole" in tokens or "shaft" in tokens or "bar" in tokens:
            color, label = (0, 255, 255), f"POLE / SHAFT {row['conf']:.2f}"
            pole_boxes.append([x1, y1, x2, y2])
        else:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
    return plate_boxes, full_barbell_boxes, pole_boxes


def draw_inferred_shaft_box(frame, left_box, right_box):
    """Draw an estimated narrow pole box between two detected plate boxes."""
    lb, rb = sorted([left_box, right_box], key=lambda b: (b[0] + b[2]) / 2)
    x1, x2 = int(lb[2]), int(rb[0])
    if x2 <= x1:
        return
    cy = int(round(((lb[1] + lb[3]) / 2 + (rb[1] + rb[3]) / 2) / 2))
    plate_h = max(1, min(lb[3] - lb[1], rb[3] - rb[1]))
    half = max(4, int(plate_h * 0.07))
    y1, y2 = cy - half, cy + half
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "INFERRED POLE / SHAFT", (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2, cv2.LINE_AA)



def draw_barbell_assembly_box(frame, plate_boxes, full_barbell_boxes):
    """Draw one outer box for the complete detected/estimated barbell."""
    if full_barbell_boxes:
        # The detector's full-barbell box is the most reliable outer box.
        b = max(full_barbell_boxes, key=lambda q: (q[2]-q[0])*(q[3]-q[1]))
        x1,y1,x2,y2 = [int(v) for v in b]
        label = "BARBELL DETECTED"
    elif len(plate_boxes) >= 2:
        # Approximate assembly extent from the two plate boxes.
        x1 = int(min(b[0] for b in plate_boxes))
        y1 = int(min(b[1] for b in plate_boxes))
        x2 = int(max(b[2] for b in plate_boxes))
        y2 = int(max(b[3] for b in plate_boxes))
        label = "BARBELL ASSEMBLY (EST.)"
    else:
        return
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 80, 220), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, max(18,y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,80,220), 2, cv2.LINE_AA)


def smooth_point(new_point, old_point, alpha=0.55):
    if new_point is None:
        return old_point
    if old_point is None:
        return np.asarray(new_point, dtype=np.float32)
    return alpha * np.asarray(new_point, dtype=np.float32) + (1-alpha) * old_point


def choose_person(result, previous_center=None):
    if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return None

    xy_all = result.keypoints.xy.cpu().numpy()
    conf_all = (result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None
                else np.ones(xy_all.shape[:2], dtype=np.float32))
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None

    candidates = []
    for i, pts in enumerate(xy_all):
        valid = conf_all[i] >= 0.20
        if int(valid.sum()) < 7:
            continue
        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = boxes[i]
            area = max(1.0, (x2-x1)*(y2-y1))
            center = ((x1+x2)/2, (y1+y2)/2)
        else:
            p = pts[valid]
            area = max(1.0, (p[:,0].max()-p[:,0].min()) *
                       (p[:,1].max()-p[:,1].min()))
            center = (float(p[:,0].mean()), float(p[:,1].mean()))
        if previous_center is None:
            score = area
        else:
            dist = math.hypot(center[0]-previous_center[0], center[1]-previous_center[1])
            score = area / (1.0 + dist * 0.002)
        candidates.append((score, i, center))
    if not candidates:
        return None
    _, idx, center = max(candidates, key=lambda x: x[0])
    return xy_all[idx], conf_all[idx], center


def draw_skeleton(frame, points, conf, threshold=0.18):
    valid = conf >= threshold
    h, w = frame.shape[:2]
    for a, b in SKELETON:
        if valid[a] and valid[b]:
            p1 = tuple(np.round(points[a]).astype(int))
            p2 = tuple(np.round(points[b]).astype(int))
            cv2.line(frame, p1, p2, (255, 190, 40), 3, cv2.LINE_AA)
    for i, p in enumerate(points):
        if valid[i]:
            x, y = np.round(p).astype(int)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)


def angle_abc(a, b, c):
    v1 = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    v2 = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    den = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if den < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2)/den, -1, 1))))


def body_metrics(points, conf, threshold=0.18):
    sides = []
    for hip, knee, ankle in ((11,13,15), (12,14,16)):
        if min(conf[hip], conf[knee], conf[ankle]) >= threshold:
            ang = angle_abc(points[hip], points[knee], points[ankle])
            leg = float(np.linalg.norm(points[hip] - points[ankle]))
            if ang is not None and leg > 1:
                sides.append((points[hip], ang, leg))
    if not sides:
        return None, None, None
    hip_center = np.mean([s[0] for s in sides], axis=0)
    knee_angle = float(np.mean([s[1] for s in sides]))
    leg_scale = float(np.median([s[2] for s in sides]))
    return hip_center, knee_angle, leg_scale


def draw_panel(frame, lines):
    h, w = frame.shape[:2]
    x, y = 12, 12
    panel_w = min(430, w-24)
    line_h = 23
    panel_h = min(45 + line_h*len(lines), h-24)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x,y), (x+panel_w,y+panel_h), (10,16,26), -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
    cv2.rectangle(frame, (x,y), (x+panel_w,y+panel_h), (80,220,160), 2)
    cv2.putText(frame, "SQUAT + BARBELL ANALYSIS", (x+10,y+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80,240,180), 2, cv2.LINE_AA)
    for i, (label, value) in enumerate(lines):
        yy = y+52+i*line_h
        if yy >= y+panel_h-3:
            break
        cv2.putText(frame, label, (x+10,yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (220,225,230), 1, cv2.LINE_AA)
        cv2.putText(frame, str(value), (x+int(panel_w*0.54),yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0,235,255), 1, cv2.LINE_AA)


def draw_path_chart(frame, path, rep_label):
    h, w = frame.shape[:2]
    cw, ch = min(260,w-24), min(150,h//3)
    if cw < 110 or ch < 70:
        return
    x0, y0 = w-cw-12, h-ch-12
    overlay = frame.copy()
    cv2.rectangle(overlay,(x0,y0),(x0+cw,y0+ch),(12,18,28),-1)
    cv2.addWeighted(overlay,0.85,frame,0.15,0,frame)
    cv2.rectangle(frame,(x0,y0),(x0+cw,y0+ch),(180,190,200),1)
    cv2.putText(frame,rep_label,(x0+7,y0+18),cv2.FONT_HERSHEY_SIMPLEX,
                0.43,(255,255,255),1,cv2.LINE_AA)
    if len(path) < 2:
        return
    xs=[p[0] for p in path]; ys=[p[1] for p in path]
    minx,maxx=min(xs),max(xs); miny,maxy=min(ys),max(ys)
    dx=max(1.0,maxx-minx); dy=max(1.0,maxy-miny)
    pts=[]
    for px,py in path:
        pts.append((x0+7+int((px-minx)/dx*(cw-14)),
                    y0+25+int((py-miny)/dy*(ch-32))))
    for i in range(1,len(pts)):
        cv2.line(frame,pts[i-1],pts[i],(0,220,255),2,cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", default="output")
    ap.add_argument("--detector", default="best/best.pt",
                    help="YOLO object detector weights that include barbell/plate class")
    ap.add_argument("--pose-model", default="yolo11s-pose.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--det-conf", type=float, default=0.20)
    ap.add_argument("--pose-conf", type=float, default=0.20)
    ap.add_argument("--start-drop", type=float, default=0.07)
    ap.add_argument("--bottom-drop", type=float, default=0.18)
    ap.add_argument("--return-drop", type=float, default=0.055)
    ap.add_argument("--bottom-knee-angle", type=float, default=120.0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.source):
        raise FileNotFoundError("Video not found: " + args.source)
    if not os.path.isfile(args.detector):
        raise FileNotFoundError(
            f"Object detector not found: {args.detector}\n"
            "Place your barbell detection weights at best\\best.pt or pass --detector."
        )

    device = str(args.device)
    if device != "cpu" and device != "-1":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available in this Python environment. Your RTX 3050 "
                "is visible to Windows, but PyTorch currently cannot use it. "
                "Install a CUDA-enabled PyTorch build before running with --device 0."
            )
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.output if os.path.isabs(args.output) else os.path.join(root,args.output)
    os.makedirs(out_dir,exist_ok=True)

    print("Loading automatic barbell detector:",args.detector)
    detector = YOLO(args.detector)
    names = detector.names
    print("Detector classes:",names)

    flat_names = [str(v).lower() for v in (names.values() if isinstance(names,dict) else names)]
    recognized = any(any(word in n for word in BAR_WORDS) for n in flat_names)
    if not recognized and len(flat_names) != 1:
        raise RuntimeError(
            "Your best.pt does not appear to contain a barbell/plate/pole class.\n"
            f"Classes found: {names}\n"
            "A generic model cannot reliably identify a barbell automatically. "
            "Use a detector trained with a barbell class or provide compatible weights."
        )
    if not recognized and len(flat_names) == 1:
        print("WARNING: single detector class will be treated as the barbell class:",
              flat_names[0])

    print("Loading human pose model:",args.pose_model)
    pose_model = YOLO(args.pose_model)

    cap=cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError("Could not open video: "+args.source)
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0: fps=30.0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration=total/fps if total else 0
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stem=os.path.splitext(os.path.basename(args.source))[0]
    video_out=os.path.join(out_dir,stem+"_automatic_analysis.mp4")
    frame_csv=os.path.join(out_dir,stem+"_frame_metrics.csv")
    rep_csv=os.path.join(out_dir,stem+"_rep_summary.csv")
    traj_csv=os.path.join(out_dir,stem+"_trajectory_per_rep.csv")

    writer=cv2.VideoWriter(video_out,cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    if not writer.isOpened():
        raise RuntimeError("Could not create output MP4: "+video_out)

    # Repetition state and calibration
    phase="CALIBRATING"
    standing_y=None
    calib=[]
    hip_smooth=None
    previous_person_center=None
    knee_hist=deque(maxlen=5)
    rep_count=0
    rep_start=None
    start_count=0
    bottom_count=0
    return_count=0
    rep_durations=[]
    rep_rows=[]
    frame_rows=[]
    trajectory_rows=[]
    active_path=[]
    previous_path=[]
    active_depth=0.0
    active_peak_speed=0.0
    active_peak_accel=0.0

    # Barbell tracking memory; short gaps are bridged, long gaps shown as lost.
    last_left=last_right=last_shaft=None
    bar_missing=0
    previous_bar_y=None
    previous_speed=0.0
    bar_speed=bar_accel=0.0
    peak_speed=peak_accel=0.0
    frame_idx=0

    while True:
        ok,frame=cap.read()
        if not ok: break
        t=frame_idx/fps
        dt=1.0/fps
        vis=frame.copy()

        # Automatic barbell / plates / pole detection.
        det_result=detector.predict(frame,imgsz=args.imgsz,conf=args.det_conf,
                                    device=device,verbose=False)[0]
        det_rows=get_detection_rows(det_result,names,args.det_conf)

        # If the detector has one class with a nonstandard label, treat its
        # highest-confidence box as the barbell candidate.
        if len(flat_names)==1 and not recognized and det_rows:
            for row in det_rows:
                row["name"]="barbell"

        plate_boxes, full_barbell_boxes, detected_pole_boxes = draw_object_boxes(
            vis, det_rows, width, height
        )

        left,right,shaft,pole_box,det_source=detect_barbell_geometry(
            det_rows,width,height
        )

        if left is not None:
            last_left=np.asarray(left,dtype=np.float32)
        if right is not None:
            last_right=np.asarray(right,dtype=np.float32)
        if shaft is not None:
            last_shaft=(np.asarray(shaft[0],dtype=np.float32),
                        np.asarray(shaft[1],dtype=np.float32))
            bar_missing=0
        else:
            bar_missing+=1

        # Draw only recent remembered detections; no indefinite fake persistence.
        left_draw=last_left if bar_missing<=8 else None
        right_draw=last_right if bar_missing<=8 else None
        shaft_draw=last_shaft if bar_missing<=8 else None

        if left_draw is not None:
            p=tuple(np.round(left_draw).astype(int))
            cv2.circle(vis,p,11,(0,255,80),3,cv2.LINE_AA)
            cv2.putText(vis,"LEFT PLATE", (p[0]+8,p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,80),1,cv2.LINE_AA)
        if right_draw is not None:
            p=tuple(np.round(right_draw).astype(int))
            cv2.circle(vis,p,11,(255,130,20),3,cv2.LINE_AA)
            cv2.putText(vis,"RIGHT PLATE",(p[0]+8,p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,130,20),1,cv2.LINE_AA)
        # Use the detector's pole box when available. Otherwise infer a tight
        # shaft box between the two plate boxes (never draw a full-frame line).
        if not detected_pole_boxes and len(plate_boxes) >= 2:
            ordered = sorted(plate_boxes, key=lambda b: (b[0] + b[2]) / 2)
            draw_inferred_shaft_box(vis, ordered[0], ordered[-1])
        elif shaft_draw is not None and not detected_pole_boxes and len(plate_boxes) < 2:
            p1=tuple(np.round(shaft_draw[0]).astype(int))
            p2=tuple(np.round(shaft_draw[1]).astype(int))
            cv2.line(vis,p1,p2,(0,255,255),3,cv2.LINE_AA)

        # Outer box around the complete barbell assembly.
        draw_barbell_assembly_box(vis, plate_boxes, full_barbell_boxes)

        bar_center=None
        if left is not None and right is not None:
            bar_center=((left[0]+right[0])/2,(left[1]+right[1])/2)
        elif shaft is not None:
            bar_center=((shaft[0][0]+shaft[1][0])/2,(shaft[0][1]+shaft[1][1])/2)
        elif left is not None:
            bar_center=left
        elif right is not None:
            bar_center=right

        if bar_center is not None:
            if previous_bar_y is not None:
                bar_speed=(bar_center[1]-previous_bar_y)/dt
                bar_accel=(bar_speed-previous_speed)/dt
            else:
                bar_speed=bar_accel=0.0
            previous_bar_y=bar_center[1]
            previous_speed=bar_speed
            peak_speed=max(peak_speed,abs(bar_speed))
            peak_accel=max(peak_accel,abs(bar_accel))
            if rep_start is not None:
                active_path.append((float(bar_center[0]),float(bar_center[1])))
                active_peak_speed=max(active_peak_speed,abs(bar_speed))
                active_peak_accel=max(active_peak_accel,abs(bar_accel))
                trajectory_rows.append([
                    rep_count+1,round(t,4),
                    round(left[0],2) if left else "",round(left[1],2) if left else "",
                    round(right[0],2) if right else "",round(right[1],2) if right else "",
                    round(bar_center[0],2),round(bar_center[1],2)
                ])
        else:
            bar_speed=bar_accel=0.0

        # Full-body pose.
        pose_result=pose_model.predict(frame,imgsz=args.imgsz,conf=args.pose_conf,
                                       device=device,verbose=False)[0]
        person=choose_person(pose_result,previous_person_center)
        hip_drop=None
        knee_angle=None
        if person is not None:
            pts,conf,previous_person_center=person
            draw_skeleton(vis,pts,conf)
            hip,knee_angle,leg_scale=body_metrics(pts,conf)
            if knee_angle is not None:
                knee_hist.append(knee_angle)
            knee_display=float(np.median(knee_hist)) if knee_hist else None

            if hip is not None and leg_scale is not None:
                hip_smooth=smooth_point(hip,hip_smooth,0.4)
                if standing_y is None:
                    calib.append(float(hip_smooth[1]))
                    # Calibrate using the earliest stable samples (about 1 second).
                    if len(calib)>=max(10,int(fps*0.8)):
                        standing_y=float(np.percentile(calib,20))
                        phase="STANDING"
                else:
                    hip_drop=max(0.0,(float(hip_smooth[1])-standing_y)/max(leg_scale,1.0))
                    # Repetition state machine uses both pelvis descent and knee flexion.
                    knee_is_bent=(knee_display is not None and knee_display<args.bottom_knee_angle)

                    if phase=="STANDING":
                        if hip_drop>=args.start_drop:
                            start_count+=1
                        else:
                            start_count=0
                        if start_count>=3:
                            phase="DESCENDING"
                            rep_start=max(0.0,t-2*dt)
                            active_path=[]
                            active_depth=hip_drop
                            active_peak_speed=active_peak_accel=0.0
                            bottom_count=return_count=0
                            start_count=0

                    elif phase=="DESCENDING":
                        active_depth=max(active_depth,hip_drop)
                        if hip_drop>=args.bottom_drop or (knee_is_bent and hip_drop>=args.start_drop*1.6):
                            bottom_count+=1
                        else:
                            bottom_count=0
                        if bottom_count>=2:
                            phase="BOTTOM"

                    elif phase=="BOTTOM":
                        active_depth=max(active_depth,hip_drop)
                        if hip_drop < args.bottom_drop-0.025:
                            phase="ASCENDING"

                    elif phase=="ASCENDING":
                        active_depth=max(active_depth,hip_drop)
                        if hip_drop<=args.return_drop and (
                            knee_display is None or knee_display>=args.bottom_knee_angle
                        ):
                            return_count+=1
                        else:
                            return_count=0
                        if return_count>=3 and rep_start is not None:
                            end_t=t
                            rep_duration=end_t-rep_start
                            if rep_duration>=0.45 and active_depth>=args.start_drop*1.6:
                                rep_count+=1
                                rep_durations.append(rep_duration)
                                rep_rows.append([
                                    rep_count,round(rep_start,4),round(end_t,4),
                                    round(rep_duration,4),round(active_depth,4),
                                    round(knee_display,2) if knee_display is not None else "",
                                    round(active_peak_speed,2),round(active_peak_accel,2)
                                ])
                                previous_path=list(active_path)
                            phase="STANDING"
                            rep_start=None
                            active_path=[]
                            bottom_count=return_count=start_count=0
                    hip_drop=max(0.0,(float(hip_smooth[1])-standing_y)/max(leg_scale,1.0))
        else:
            knee_display=float(np.median(knee_hist)) if knee_hist else None

        # Direction: image y increases downward.
        direction="DOWN" if bar_speed>8 else "UP" if bar_speed<-8 else "HOLD"
        avg_cycle=float(np.mean(rep_durations)) if rep_durations else None
        last_cycle=rep_durations[-1] if rep_durations else None
        current_cycle=t-rep_start if rep_start is not None else 0.0

        panel=[
            ("TOTAL REPS",rep_count),
            ("SQUAT PHASE",phase),
            ("KNEE ANGLE",f"{knee_display:.1f} deg" if knee_display is not None else "N/A"),
            ("HIP DROP",f"{hip_drop:.3f} leg length" if hip_drop is not None else "N/A"),
            ("CURRENT CYCLE",f"{current_cycle:.2f} s" if rep_start is not None else "--"),
            ("LAST CYCLE",f"{last_cycle:.2f} s" if last_cycle is not None else "--"),
            ("AVG CYCLE",f"{avg_cycle:.2f} s" if avg_cycle is not None else "--"),
            ("VIDEO TIME",f"{t:.2f}/{duration:.2f} s"),
            ("BAR SPEED",f"{bar_speed:.1f} px/s"),
            ("BAR ACCEL",f"{bar_accel:.1f} px/s2"),
            ("PEAK SPEED",f"{peak_speed:.1f} px/s"),
            ("OBJECT DETECTION",det_source[:28]),
        ]
        draw_panel(vis,panel)

        path_to_draw=active_path if active_path else previous_path
        draw_path_chart(vis,path_to_draw,
                        f"REP {rep_count+1 if active_path else rep_count} TRAJECTORY")

        frame_rows.append([
            round(t,4),rep_count+(1 if rep_start is not None else 0),phase,
            round(knee_display,2) if knee_display is not None else "",
            round(hip_drop,4) if hip_drop is not None else "",
            round(left[0],2) if left else "",round(left[1],2) if left else "",
            round(right[0],2) if right else "",round(right[1],2) if right else "",
            round(bar_center[0],2) if bar_center else "",
            round(bar_center[1],2) if bar_center else "",
            round(bar_speed,3),round(bar_accel,3),det_source
        ])

        # DRAW BARBELL TRAJECTORY DIRECTLY ON VIDEO
        if len(active_path) >= 2:
            path_points = np.array(
                active_path, dtype=np.int32
            ).reshape((-1, 1, 2))

            # Draw trajectory line
            cv2.polylines(
                vis,
                [path_points],
                False,
                (0, 255, 255),
                3,
                cv2.LINE_AA
            )

            # Mark current barbell position
            px, py = active_path[-1]

            cv2.circle(
                vis,
                (int(px), int(py)),
                6,
                (0, 0, 255),
                -1
            )

        writer.write(vis)
        if args.show:
            cv2.imshow("Automatic Squat Analysis - press Q to stop",vis)
        if cv2.waitKey(1)&0xFF==ord("q"):
            break
        frame_idx+=1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    with open(frame_csv,"w",newline="",encoding="utf-8") as f:
        cw=csv.writer(f)
        cw.writerow([
            "time_s","rep_number","phase","knee_angle_deg","hip_drop_leg_length",
            "left_plate_x","left_plate_y","right_plate_x","right_plate_y",
            "bar_center_x","bar_center_y","bar_vertical_speed_px_s",
            "bar_vertical_accel_px_s2","object_detection_status"
        ])
        cw.writerows(frame_rows)

    with open(rep_csv,"w",newline="",encoding="utf-8") as f:
        cw=csv.writer(f)
        cw.writerow([
            "rep_number","start_s","end_s","cycle_duration_s",
            "max_hip_drop_leg_length","knee_angle_at_completion_deg",
            "peak_bar_speed_px_s","peak_bar_acceleration_px_s2"
        ])
        cw.writerows(rep_rows)

    with open(traj_csv,"w",newline="",encoding="utf-8") as f:
        cw=csv.writer(f)
        cw.writerow([
            "rep_number","time_s","left_plate_x","left_plate_y",
            "right_plate_x","right_plate_y","bar_center_x","bar_center_y"
        ])
        cw.writerows(trajectory_rows)

    print("\n===== AUTOMATIC ANALYSIS COMPLETE =====")
    print("GPU:",torch.cuda.get_device_name(0) if torch.cuda.is_available() and device not in ("cpu","-1") else "CPU")
    print("Completed reps:",rep_count)
    if rep_durations:
        print("Average rep duration:",round(float(np.mean(rep_durations)),2),"seconds")
    print("Annotated video:",video_out)
    print("Frame metrics:",frame_csv)
    print("Rep summary:",rep_csv)
    print("Trajectory per rep:",traj_csv)
    print("All files saved in:",out_dir)


if __name__=="__main__":
    main()