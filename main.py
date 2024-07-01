from utils import load_video_frames, export_video
from config import SAMPLE_DATA_DIR, TEST_OUTPUT_DIR, MODELS_DIR, TRACKER_STUB_DIR
from trackers import PlayerTracker, BallTracker
import os
from dotenv import load_dotenv


def main():
    # Load environment variables
    load_dotenv()

    video_path = f"{SAMPLE_DATA_DIR}/sample.mp4"
    video_frames = load_video_frames(video_path)

    player_tracker = PlayerTracker(model_path=f"{MODELS_DIR}/best.pt")

    player_detection = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path=f"{TRACKER_STUB_DIR}/player_detection.pkl",
    )

    ball_tracker = BallTracker(
        api_key=os.getenv("ROBOFLOW_API_KEY"),
        project_name="tennis-ball-m7zvw",
        version_number=3,
    )

    ball_detection = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path=f"{TRACKER_STUB_DIR}/ball_detection.pkl",
    )

    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detection)
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detection)

    export_video(output_video_frames, f"{TEST_OUTPUT_DIR}/output_video_frames.mp4")


if __name__ == "__main__":
    main()