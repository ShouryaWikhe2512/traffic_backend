from fastapi import APIRouter, UploadFile, File, Form
import shutil
import os
import time
from typing import Dict
from services.video_processor import VideoProcessor
from services.intelligence import SignalOptimizer, KalmanFilter
from services.notification import NotificationService

router = APIRouter()

# Initialize Services
optimizer = SignalOptimizer()
processor = VideoProcessor()
kalman_filters = {
    "north": KalmanFilter(),
    "south": KalmanFilter(),
    "east": KalmanFilter(),
    "west": KalmanFilter()
}

# Twilio Integration
notification_service = NotificationService(
    account_sid="AC52a689d4989f028954cd42d52c73102a",
    auth_token="4b3256fbe379f5cdb18f84a0fd8d9e49",
    from_number="+16812533265"
)

# Global simulation state
SIMULATION_SESSION = {
    "start_time": None,
    "last_poll_time": 0.0,
    "emergency_history": {
        "north": 0, "south": 0, "east": 0, "west": 0
    }
}

@router.post("/upload-intersection")
async def upload_intersection(
    north: UploadFile = File(...),
    south: UploadFile = File(...),
    east: UploadFile = File(...),
    west: UploadFile = File(...),
    task_type: str = Form("frame") # Use Form to capture from FormData
):
    os.makedirs("temp_uploads", exist_ok=True)
    files = {"north": north, "south": south, "east": east, "west": west}
    file_paths = {}

    # Save files
    for direction, file in files.items():
        temp_path = f"temp_uploads/{direction}_{file.filename}"
        if not os.path.exists(temp_path):
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        file_paths[direction] = temp_path

    # --- BATCH PRE-ANALYSIS MODE ---
    if task_type == "pre-analysis":
        print("\n[SYSTEM] INITIATING GLOBAL BATCH INTELLIGENCE SCAN...")
        batch_results = {}
        for direction, path in file_paths.items():
            analysis = processor.analyze_full_video(path, direction)
            batch_results[direction] = analysis
            print(f"[BATCH] {direction.upper()} | Agg Pressure: {analysis['pressure']} | Avg Queue: {analysis['avg_queue']}")

        priority_sequence = sorted(
            batch_results.keys(), 
            key=lambda x: batch_results[x]["pressure"], 
            reverse=True
        )
        print(f"[SYSTEM] SCAN COMPLETE. RANKING: {' -> '.join(priority_sequence).upper()}\n")
        
        return {
            "status": "success",
            "task": "pre-analysis",
            "priority_sequence": priority_sequence,
            "batch_telemetry": batch_results
        }

    # --- STANDARD FRAME MODE ---
    now = time.time()
    last_poll = SIMULATION_SESSION.get("last_poll_time", 0.0)
    
    # Reset if simulation restarted or first run
    if SIMULATION_SESSION["start_time"] is None or (now - last_poll > 10):
        SIMULATION_SESSION["start_time"] = now
        SIMULATION_SESSION["emergency_history"] = {"north": 0, "south": 0, "east": 0, "west": 0}
    
    SIMULATION_SESSION["last_poll_time"] = now
    elapsed = now - SIMULATION_SESSION["start_time"]
    
    results = {}
    weighted_queues = {}
    confirmed_emergency = False
    confirmed_dir = None

    for direction, path in file_paths.items():
        detection = processor.process_frame(path, direction, elapsed)
        
        # 1. Temporal Emergency Validation
        history = SIMULATION_SESSION["emergency_history"]
        if detection.get("emergency_detected"):
            history[direction] += 1
            print(f"[SYSTEM] Emergency Spotting in {direction.upper()} (Count: {history[direction]}/2)")
        else:
            # Soft reset: Decrement instead of zeroing to handle flickering
            history[direction] = max(0, history[direction] - 1)
        
        is_validated = history[direction] >= 2 # Lowered from 3
        if is_validated and not confirmed_emergency:
            confirmed_emergency = True
            confirmed_dir = direction
            print(f"[SYSTEM] Emergency CONFIRMED in {direction.upper()} - Dispatching Dispatcher...")
            # Trigger Twilio Voice Alert
            notification_service.notify_emergency(direction, detection.get('emergency_type', 'Emergency Vehicle'))

        # 2. Weighted Queue Calculation (MATCH STATED USER FORMAT)
        cats = detection.get('categories', {})
        w_queue = float(
            cats.get('car', 0) * 1.0 +
            cats.get('motorcycle', 0) * 0.5 +
            cats.get('bus', 0) * 2.5 +
            cats.get('truck', 0) * 3.0 +
            cats.get('emergency', 0) * 5.0
        )
        
        smoothed = float(kalman_filters[direction].update(w_queue))
        
        results[direction] = {
            **detection,
            "emergency_validated": is_validated,
            "consecutive_frames": history[direction],
            "weighted_queue": float(f"{w_queue:.1f}"),
            "smoothed_queue": float(f"{smoothed:.1f}"),
            "pressure": 0.0 # Placeholder
        }
        weighted_queues[direction] = smoothed

    # 3. Dynamic Optimization
    signal_plan = optimizer.calculate_plan(
        queues=weighted_queues,
        emergency_flag=confirmed_emergency,
        emergency_dir=confirmed_dir
    )

    for d in results:
        results[d]["pressure"] = signal_plan["pressures"].get(d, 0.0)
        results[d]["assigned_green"] = signal_plan["splits"].get(d, 0)
        results[d]["split_percent"] = round((results[d]["assigned_green"] / (sum(signal_plan["splits"].values()) or 1)) * 100)

    # Clean UI Logging
    for d in ["north", "south", "east", "west"]:
        res = results[d]
        e_status = "!! EMERGENCY !!" if res['emergency_detected'] else "Normal"
        print(f"[{d.upper()}] T={elapsed:.1f}s | Q: {res['raw_queue']} | {e_status} | Cats: {res['categories']}")
    
    print(f"--- ATCS LIVE UPDATE (T={elapsed:.1f}s) | Plan: {signal_plan['mode']} -> {signal_plan['selected_phase']} ---\n")

    return {
        "intersection_telemetry": results,
        "signal_plan": signal_plan,
        "debug": {"simulation_time": round(elapsed, 1)}
    }
