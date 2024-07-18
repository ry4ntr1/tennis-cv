from utils import load_video_frames, export_video
from config import SAMPLE_DATA_DIR, TEST_OUTPUT_DIR, MODELS_DIR, TRACKER_STUB_DIR
from trackers import PlayerTracker, BallTracker
from keypoint_detection import KeypointDetector
from dotenv import load_dotenv
import cv2


def main():
    load_dotenv()

    video_path = f"{SAMPLE_DATA_DIR}/sample.mp4"
    video_frames = load_video_frames(video_path)

    player_tracker = PlayerTracker(model_path=f"{MODELS_DIR}/best.pt")

    player_detection = player_tracker.detect_frames(
        video_frames,
        read_from_stub=False,
        stub_path=f"{TRACKER_STUB_DIR}/player_detection.pkl",
    )

    # ball_tracker = BallTracker(
    #     api_key=os.getenv("ROBOFLOW_API_KEY"),
    #     project_name="tennis-ball-m7zvw",
    #     version_number=3,
    # )

    ball_tracker = BallTracker(model_path=f"{MODELS_DIR}/best.pt")

    ball_detection = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=False,
        stub_path=f"{TRACKER_STUB_DIR}/ball_detection.pkl",
    )

    keypoint_detector = KeypointDetector(model_path=f"{MODELS_DIR}/keypoints_model.pth")

    keypoint_predictions = keypoint_detector.predict(video_frames[0])
    output_video_frames = keypoint_detector.draw_keypoints_on_video(
        video_frames, keypoint_predictions
    )

    player_detection = player_tracker.interpolate_player_positions(player_detection)
    player_detection = player_tracker.choose_and_filter_players(
        keypoint_predictions, player_detection
    )

    ball_detection = ball_tracker.interpolate_ball_positions(ball_detection)

    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detection)
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detection)

    for i, frame in enumerate(output_video_frames):
        cv2.putText(
            frame,
            f"Frame: {i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    export_video(output_video_frames, f"{TEST_OUTPUT_DIR}/output_video_frames.mp4")


if __name__ == "__main__":
    main()
