import cv2
import pandas as pd
from utils import (
    load_video_frames,
    export_video,
    measure_distance,
    convert_pixel_distance_to_meters,
    convert_meters_to_pixel_distance,
    display_stats,
)
from config import (
    SAMPLE_DATA_DIR,
    TEST_OUTPUT_DIR,
    MODELS_DIR,
    TRACKER_STUB_DIR,
    DOUBLE_LINE_WIDTH,
)
from trackers import PlayerTracker, BallTracker
from keypoint_detection import KeypointDetector
from dotenv import load_dotenv
from mini_court import MiniCourt
from copy import deepcopy


def main():
    load_dotenv()

    # Load Video Frames
    video_path = f"{SAMPLE_DATA_DIR}/sample.mp4"
    video_frames = load_video_frames(video_path)

    # PlayerTracker: Init + Detection
    player_tracker = PlayerTracker(model_path=f"{MODELS_DIR}/best.pt")
    player_detection = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path=f"{TRACKER_STUB_DIR}/player_detection.pkl",
    )

    # BallTracker: Init + Detection
    ball_tracker = BallTracker(model_path=f"{MODELS_DIR}/best.pt")
    ball_detection = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=False,
        stub_path=f"{TRACKER_STUB_DIR}/ball_detection.pkl",
    )

    # KeypointDetector: Init + Prediction
    keypoint_detector = KeypointDetector(model_path=f"{MODELS_DIR}/keypoints_model.pth")
    keypoint_predictions = keypoint_detector.predict(video_frames[0])

    # MiniCourt: Init + Draw
    mini_court = MiniCourt(video_frames[0])

    # PlayerTracker: Interpolation + Filtering
    player_detection = player_tracker.interpolate_player_positions(player_detection)
    player_detection = player_tracker.choose_and_filter_players(
        keypoint_predictions, player_detection
    )

    # BallTracker: Interpolation
    ball_detection = ball_tracker.interpolate_ball_positions(ball_detection)

    # BallTracker: Detect Ball Hits
    ball_hit_frames = ball_tracker.detect_hits(ball_detection)

    # MiniCourt: Convert Player + Tennis Ball Bounding Boxes to Mini Court Coordinates
    player_mini_court_detection, ball_mini_court_detection = (
        mini_court.add_to_minicourt(
            player_detection, ball_detection, keypoint_predictions
        )
    )

    player_stats = [
        {
            "frame_num": 0,
            "player_1_number_of_shots": 0,
            "player_1_total_shot_speed": 0,
            "player_1_last_shot_speed": 0,
            "player_1_total_player_speed": 0,
            "player_1_last_player_speed": 0,
            "player_2_number_of_shots": 0,
            "player_2_total_shot_speed": 0,
            "player_2_last_shot_speed": 0,
            "player_2_total_player_speed": 0,
            "player_2_last_player_speed": 0,
        }
    ]

    for frame_idx in range(len(ball_hit_frames) - 1):
        # Calculate Hit Duration
        start = ball_hit_frames[frame_idx]
        end = ball_hit_frames[frame_idx + 1]
        hit_duration_seconds = (end - start) / 24

        # Calculate Distance
        ball_distance_px = measure_distance(
            ball_mini_court_detection[start][1], ball_mini_court_detection[end][1]
        )
        ball_distance_meters = convert_pixel_distance_to_meters(
            ball_distance_px, DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court()
        )
        ball_speed = (
            ball_distance_meters / hit_duration_seconds * 3.6
        )  # 3.6 to convert m/s to km/h

        # Get Player Who Shot Ball
        player_positions = player_mini_court_detection[start]
        player_who_hit = min(
            player_positions.keys(),
            key=lambda id: measure_distance(
                player_positions[id], ball_mini_court_detection[start][1]
            ),
        )

        # Opponent Player Speed
        opponent_player = 1 if player_who_hit == 2 else 2
        distance_covered_by_opponent_px = measure_distance(
            player_mini_court_detection[start][opponent_player],
            player_mini_court_detection[end][opponent_player],
        )
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
            distance_covered_by_opponent_px,
            DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court(),
        )
        opponent_speed = (
            distance_covered_by_opponent_meters / hit_duration_seconds * 3.6
        )

        current_frame_stats = deepcopy(player_stats[-1])
        current_frame_stats["frame_num"] = start
        current_frame_stats[f"player_{player_who_hit}_number_of_shots"] += 1
        current_frame_stats[f"player_{player_who_hit}_total_shot_speed"] += ball_speed
        current_frame_stats[f"player_{player_who_hit}_last_shot_speed"] = ball_speed
        current_frame_stats[f"player_{opponent_player}_total_player_speed"] += (
            opponent_speed
        )
        current_frame_stats[f"player_{opponent_player}_last_player_speed"] = (
            opponent_speed
        )

        player_stats.append(current_frame_stats)

    player_stats_df = pd.DataFrame(player_stats)
    frames_df = pd.DataFrame({"frame_num": range(len(video_frames))})
    player_stats_df = pd.merge(frames_df, player_stats_df, on="frame_num", how="left")
    player_stats_data_df = player_stats_df.ffill()

    player_stats_data_df["player_1_avg_shot_speed"] = (
        player_stats_data_df["player_1_total_shot_speed"]
        / player_stats_data_df["player_1_number_of_shots"]
    )
    player_stats_data_df["player_2_avg_shot_speed"] = (
        player_stats_data_df["player_2_total_shot_speed"]
        / player_stats_data_df["player_2_number_of_shots"]
    )
    player_stats_data_df["player_1_avg_player_speed"] = (
        player_stats_data_df["player_1_total_player_speed"]
        / player_stats_data_df["player_1_number_of_shots"]
    )
    player_stats_data_df["player_2_avg_player_speed"] = (
        player_stats_data_df["player_2_total_player_speed"]
        / player_stats_data_df["player_2_number_of_shots"]
    )

    # Draw Bounding Boxes: Players + Ball + Keypoints
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detection)
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detection)
    output_video_frames = keypoint_detector.draw_keypoints_on_video(
        video_frames, keypoint_predictions
    )

    # Add MiniCourt To Video Frames
    output_video_frames = mini_court.add_court_to_frames(output_video_frames)

    # Draw Player + Ball Positions on MiniCourt
    output_video_frames = mini_court.draw_positions_on_court(
        output_video_frames, player_mini_court_detection, color=(0, 255, 255)
    )
    output_video_frames = mini_court.draw_positions_on_court(
        output_video_frames, ball_mini_court_detection
    )

    # Draw Player Stats on Video Frames
    output_video_frames = display_stats(output_video_frames, player_stats_data_df)

    # Draw Frame Number (Top Left Corner)
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

    # Export Video To Specified Path
    export_video(output_video_frames, f"{TEST_OUTPUT_DIR}/output_video_frames.mp4")


if __name__ == "__main__":
    main()
