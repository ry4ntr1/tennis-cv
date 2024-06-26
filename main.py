from pathlib import Path
from utils import (load_video_frames, export_video)
from config import (SAMPLE_DATA_DIR, TEST_OUTPUT_DIR, MODELS_DIR, TRACKER_STUB_DIR)
from trackers import PlayerTracker


def main():
    currDir = Path(__file__).resolve()
    print(currDir)
    
    video_path = f"{SAMPLE_DATA_DIR}/sample.mp4"
    video_frames = load_video_frames(video_path)
    
    player_tracker = PlayerTracker(model_path=f"{MODELS_DIR}/best.pt")
    
    player_detection = player_tracker.detect_frames(video_frames, read_from_stub=False, stub_path=f"{TRACKER_STUB_DIR}/player_detection.pkl")  
    
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detection)  
    
    export_video(output_video_frames, f"{TEST_OUTPUT_DIR}/output_video_frames.mp4")
    
if __name__ == "__main__":
    main()
    
