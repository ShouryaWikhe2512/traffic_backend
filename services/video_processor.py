from ultralytics import YOLO
import cv2
import numpy as np
import os
from typing import Dict, List, Tuple

class VideoProcessor:
    def __init__(self):
        # Load the smallest YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        
        self.category_map = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }
        
    def analyze_full_video(self, video_path: str, direction: str) -> Dict:
        """
        Scans the entire video at 1-second intervals for high-fidelity ranking.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"pressure": 0.0, "avg_queue": 0, "emergency": False}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration_sec = total_frames / fps
        
        weighted_scores = []
        emergency_hits = 0
        
        # High resolution: 1.0s intervals
        for t in range(0, int(duration_sec), 1):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            success, img = cap.read()
            if not success: break
            
            results = self.model.predict(img, verbose=False)[0]
            
            frame_w_queue = 0
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.3: continue
                
                # Apply same priority weights as standard mode
                if cls_id == 2: frame_w_queue += 1.0  # car
                elif cls_id == 3: frame_w_queue += 0.5 # motorcycle
                elif cls_id == 5: frame_w_queue += 2.5 # bus
                elif cls_id == 7: frame_w_queue += 3.0 # truck
                
                # Emergency Check
                crop = img[int(box.xyxy[0][1]):int(box.xyxy[0][3]), int(box.xyxy[0][0]):int(box.xyxy[0][2])]
                if crop.size > 0:
                    _, econf = self._classify_crop(crop)
                    if econf > 0.5: 
                        frame_w_queue += 5.0
                        emergency_hits += 1

            weighted_scores.append(frame_w_queue)

        cap.release()
        
        avg_w_queue = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0
        # Aggregated Pressure for ranking
        pressure = (avg_w_queue * 1.0) + (15.0 if emergency_hits > 3 else 0.0)
        
        return {
            "pressure": round(pressure, 2),
            "avg_queue": round(avg_w_queue, 1),
            "emergency_persistent": emergency_hits > 3,
            "duration": round(duration_sec, 1)
        }

    def process_frame(self, video_path: str, direction: str, current_timestamp: float = 0.0) -> Dict:
        """
        Extracts metadata from a video frame at a specific timestamp.
        Uses multi-stage verification: YOLO -> Crop -> Classifier -> Siren Detection.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return self._get_empty_result(direction)

        # Set to the specific timestamp (in milliseconds)
        cap.set(cv2.CAP_PROP_POS_MSEC, current_timestamp * 1000)
        success, img = cap.read()
        
        # For temporal validation (peek ahead for siren oscillation)
        temporal_frames = [img]
        if success:
            frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            for i in range(1, 4):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos + (i * 2))
                s, f = cap.read()
                if s: temporal_frames.append(f)
        
        cap.release()

        if not success or img is None:
            return self._get_empty_result(direction)

        h, w = img.shape[:2]
        roi_y1, roi_y2 = int(h * 0.1), int(h * 0.95) # Expanded ROI Height
        roi_x1, roi_x2 = int(w * 0.05), int(w * 0.95) # Expanded ROI Width
        
        # YOLO Inference
        results = self.model.predict(img, verbose=False)[0]
        
        queue_count = 0
        emergency_data = {"detected": False, "type": "none", "confidence": 0.0, "siren": 0.0}
        categories = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "emergency": 0}
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.25: continue # Slightly lower threshold for raw density
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # ROI Check
            if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                queue_count += 1
                
                # Check for emergency classification
                is_emergency = False
                if cls_id in [2, 5, 7]: # car, bus, truck
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        etype, econf = self._classify_crop(crop)
                        siren_score = self._detect_sirens(temporal_frames, [x1, y1, x2, y2])
                        
                        combined_conf = (econf * 0.4 + siren_score * 0.6)
                        if combined_conf > 0.35: # Sensitive trigger
                            emergency_data = {
                                "detected": True,
                                "type": etype if etype != "none" else "emergency_vehicle",
                                "confidence": round(float(combined_conf), 2),
                                "siren": round(float(siren_score), 2),
                                "direction": direction
                            }
                            categories["emergency"] += 1
                            is_emergency = True

                # If not emergency, count as standard vehicle
                if not is_emergency and cls_id in self.category_map:
                    categories[self.category_map[cls_id]] += 1

        density = queue_count / 30.0
        
        return {
            "direction": direction,
            "raw_queue": queue_count,
            "density": min(1.0, float(density)),
            "emergency_detected": emergency_data["detected"],
            "emergency_type": emergency_data["type"],
            "emergency_confidence": emergency_data["confidence"],
            "categories": categories,
            "metadata": {
                "timestamp": current_timestamp,
                "siren_score": emergency_data.get("siren"),
                "roi": [roi_x1, roi_y1, roi_x2, roi_y2]
            }
        }

    def _classify_crop(self, crop: np.ndarray) -> Tuple[str, float]:
        """Secondary color-based classifier for emergency vehicles."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Color masks
        white = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
        red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        
        white_p = np.sum(white > 0) / crop.size
        red_p = np.sum((red1 > 0) | (red2 > 0)) / crop.size
        
        if red_p > 0.08: return "fire_truck", 0.85
        if white_p > 0.25: return "ambulance", 0.80
        
        return "none", 0.0

    def _detect_sirens(self, frames: List[np.ndarray], bbox: List[int]) -> float:
        """Detects flashing red/blue patterns in the top region of the vehicle."""
        if len(frames) < 2: return 0.0
        x1, y1, x2, y2 = bbox
        
        intensities = []
        for f in frames:
            th = max(5, int((y2 - y1) * 0.2)) # Top 20%
            siren_region = f[max(0, y1):min(f.shape[0], y1+th), max(0, x1):min(f.shape[1], x2)]
            if siren_region.size == 0: continue
            
            hsv = cv2.cvtColor(siren_region, cv2.COLOR_BGR2HSV)
            red = cv2.inRange(hsv, (0, 150, 150), (10, 255, 255))
            blue = cv2.inRange(hsv, (100, 150, 150), (130, 255, 255))
            intensities.append(np.sum(red > 0) + np.sum(blue > 0))
            
        if not intensities: return 0.0
        
        # Oscillation check
        std = np.std(intensities)
        mean = np.mean(intensities) + 1e-5
        return min(1.0, (std / mean) * 5.0)

    def _get_empty_result(self, direction: str = "unknown") -> Dict:
        return {
            "direction": direction,
            "raw_queue": 0,
            "density": 0.0,
            "emergency_detected": False,
            "emergency_type": "none",
            "emergency_confidence": 0.0,
            "categories": {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "emergency": 0},
            "metadata": {"error": "EndOfStream"}
        }
