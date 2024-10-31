import cv2
from yolov10.ultralytics import YOLOv10
from ensemble_boxes import weighted_boxes_fusion

def load_model(model_path):
    try:
        return YOLOv10(model_path)
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return None

def create_csrt_tracker():
    return cv2.legacy.TrackerCSRT.create()

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Video açılamadı: {video_path}")
        return None
    return cap

def detect_objects(model, frame):
    results = model.predict(frame)
    detected_boxes, confidences, labels = [], [], []
    if results:
        for result in results:
            for box in result.boxes:
                confidence = box.conf[0].item()
                (x1, y1, x2, y2) = box.xyxy[0].int().tolist()
                bbox = [x1 / frame.shape[1], y1 / frame.shape[0], x2 / frame.shape[1], y2 / frame.shape[0]]
                detected_boxes.append(bbox)
                confidences.append(confidence)
                labels.append(0)
    return detected_boxes, confidences, labels

def update_trackers(trackers, frame):
    ret, tracked_boxes = trackers.update(frame)
    tracked_confidences = [0.8] * len(tracked_boxes)
    tracked_boxes_normalized = [[tb[0] / frame.shape[1], tb[1] / frame.shape[0], (tb[0] + tb[2]) / frame.shape[1], (tb[1] + tb[3]) / frame.shape[0]] for tb in tracked_boxes]
    return ret, tracked_boxes_normalized, tracked_confidences

def fuse_boxes(detected_boxes, confidences, labels, tracked_boxes_normalized, tracked_confidences):
    all_boxes = detected_boxes + tracked_boxes_normalized
    all_confidences = confidences + tracked_confidences
    all_labels = labels + [1] * len(tracked_boxes_normalized)
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion([all_boxes], [all_confidences], [all_labels], iou_thr=0.5, skip_box_thr=0.3)
    return fused_boxes

def draw_boxes(frame, fused_boxes):
    for fused_box in fused_boxes:
        (x, y, w, h) = [int(v) for v in fused_box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def main():
    model_path = '/home/nurullah/Desktop/NPC-AI/runs/detect/yolov10x-1920/best.pt'
    video_path = '/home/nurullah/Desktop/Predict/TUYZ_2024_Ornek_Veri/TUYZ_2024_Ornek_Video.MP4'

    model = load_model(model_path)
    if model is None:
        return

    cap = initialize_video_capture(video_path)
    if cap is None:
        return

    trackers = cv2.legacy.MultiTracker.create()
    initial_trackers = True
    missing_counters = []
    max_missing_frames = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Kare alınamadı, atlanıyor.")
            continue

        detected_boxes, confidences, labels = detect_objects(model, frame)
        ret, tracked_boxes_normalized, tracked_confidences = update_trackers(trackers, frame)
        fused_boxes = fuse_boxes(detected_boxes, confidences, labels, tracked_boxes_normalized, tracked_confidences)
        fused_boxes = [[int(b[0] * frame.shape[1]), int(b[1] * frame.shape[0]), int((b[2] - b[0]) * frame.shape[1]), int((b[3] - b[1]) * frame.shape[0])] for b in fused_boxes]

        if initial_trackers or not ret:
            initial_trackers = False
            trackers = cv2.legacy.MultiTracker.create()
            missing_counters = []
            for box in fused_boxes:
                if box[2] <= 0 or box[3] <= 0:
                    print(f"Geçersiz sınır kutusu boyutları: {box}")
                    continue
                tracker = create_csrt_tracker()
                trackers.add(tracker, frame, tuple(box))
                missing_counters.append(0)
        else:
            new_trackers_needed = False
            for i, _ in enumerate(missing_counters):
                if i >= len(fused_boxes):
                    missing_counters[i] += 1
                else:
                    missing_counters[i] = 0
            for i, counter in reversed(list(enumerate(missing_counters))):
                if counter > max_missing_frames:
                    trackers.getObjects().remove(i)
                    missing_counters.pop(i)
                    new_trackers_needed = True
            if new_trackers_needed:
                for box in fused_boxes:
                    if box[2] <= 0 or box[3] <= 0:
                        print(f"Geçersiz sınır kutusu boyutları: {box}")
                        continue
                    tracker = create_csrt_tracker()
                    trackers.add(tracker, frame, tuple(box))
                    missing_counters.append(0)

        draw_boxes(frame, fused_boxes)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()