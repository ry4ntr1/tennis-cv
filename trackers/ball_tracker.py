
from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
from tqdm import tqdm
import logging

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

class BallTracker:
    def __init__(self, model_path):
        """
        Initialize the BallTracker with the YOLO model from the specified path.

        :param model_path: Path to the YOLO model file.
        """
        self.model = YOLO(model_path)

    def detect_hits(self, ball_positions):
        # Extract the ball positions for key 1 from each dictionary in the ball_positions list
        ball_positions = [x.get(1, []) for x in ball_positions]

        # Convert the list of ball positions into a pandas DataFrame with columns x1, y1, x2, y2
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        # Initialize a new column "ball_hit" with all values set to 0
        df_ball_positions["ball_hit"] = 0

        # Calculate the vertical midpoint (mid_y) of the ball's bounding box
        df_ball_positions["mid_y"] = (
            df_ball_positions["y1"] + df_ball_positions["y2"]
        ) / 2

        # Calculate the rolling mean of the mid_y values over a window of 5 frames
        df_ball_positions["mid_y_rolling_mean"] = (
            df_ball_positions["mid_y"]
            .rolling(window=5, min_periods=1, center=False)
            .mean()
        )

        # Calculate the difference between consecutive rolling mean values of mid_y
        df_ball_positions["delta_y"] = df_ball_positions["mid_y_rolling_mean"].diff()

        # Set the minimum number of frames required to detect a hit
        minimum_change_frames_for_hit = 25

        # Iterate over each frame, up to a point where there are enough frames left to detect a hit
        for i in range(
            1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)
        ):
            # Detect a change in direction from positive to negative in delta_y
            negative_position_change = (
                df_ball_positions["delta_y"].iloc[i] > 0
                and df_ball_positions["delta_y"].iloc[i + 1] < 0
            )

            # Detect a change in direction from negative to positive in delta_y
            positive_position_change = (
                df_ball_positions["delta_y"].iloc[i] < 0
                and df_ball_positions["delta_y"].iloc[i + 1] > 0
            )

            # If a change in direction is detected
            if negative_position_change or positive_position_change:
                change_count = 0
                # Count subsequent changes in direction within a certain range
                for change_frame in range(
                    i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1
                ):
                    # Check for subsequent negative position changes
                    negative_position_change_following_frame = (
                        df_ball_positions["delta_y"].iloc[i] > 0
                        and df_ball_positions["delta_y"].iloc[change_frame] < 0
                    )

                    # Check for subsequent positive position changes
                    positive_position_change_following_frame = (
                        df_ball_positions["delta_y"].iloc[i] < 0
                        and df_ball_positions["delta_y"].iloc[change_frame] > 0
                    )

                    # Increment change count based on detected changes
                    if (
                        negative_position_change
                        and negative_position_change_following_frame
                    ):
                        change_count += 1
                    elif (
                        positive_position_change
                        and positive_position_change_following_frame
                    ):
                        change_count += 1

                # Mark the frame as a hit if the number of changes exceeds the threshold
                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, "ball_hit"] = 1

        # Extract the indices of the frames where ball_hit is 1
        frame_nums_with_ball_hits = df_ball_positions[
            df_ball_positions["ball_hit"] == 1
        ].index.tolist()

        return frame_nums_with_ball_hits

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolate missing ball positions in a sequence of frames.

        :param ball_positions: List of dictionaries containing ball positions for each frame.
        :return: Interpolated list of ball positions.
        """
        # Extract coordinates from ball_positions, replacing empty positions with [None, None, None, None]
        extracted_positions = []
        for pos in ball_positions:
            if pos:
                coordinates = list(pos.values())[0]
                extracted_positions.append(coordinates)
            else:
                extracted_positions.append([None, None, None, None])

        # Return original ball_positions if extracted_positions is empty or contains only None values
        if not extracted_positions or all(
            len(pos) == 0 or all(v is None for v in pos) for pos in extracted_positions
        ):
            return ball_positions

        try:
            # Create a DataFrame from the extracted positions
            df_ball_positions = pd.DataFrame(
                extracted_positions, columns=["x1", "y1", "x2", "y2"]
            )
        except ValueError as e:
            # Print the contents causing the ValueError for debugging
            print("ValueError:", e)
            print("extracted_positions causing error:", extracted_positions)
            raise e

        # Interpolate missing values and backfill remaining NaNs
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # Convert DataFrame back to list of dictionaries
        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect balls in a list of frames.

        :param frames: List of frames to be processed.
        :param read_from_stub: Flag indicating whether to read detections from a pre-saved file.
        :param stub_path: Path to the file for saving/loading detections.
        :return: List of ball detections for each frame.
        """
        ball_detections = []

        # Load detections from stub file if specified
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections

        # Detect balls in each frame
        for frame in tqdm(frames, desc="Detecting Balls"):
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        # Save detections to stub file if specified
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        """
        Detect balls in a single frame.

        :param frame: Frame to be processed.
        :return: Dictionary of detected balls with their bounding boxes.
        """
        # Run YOLO model prediction on the frame with a confidence threshold
        results = self.model.predict(frame, conf=0.125)[0]

        ball_dict = {}
        # Filter detections to include only 'tennis_ball' class (assuming class id 2)
        for i, box in enumerate(results.boxes):
            if int(box.cls) == 2:
                result = box.xyxy.tolist()[0]
                ball_dict[i] = result

        return ball_dict

    def draw_bboxes(self, video_frames, ball_detections):
        """
        Draw bounding boxes around detected balls in video frames.

        :param video_frames: List of video frames.
        :param ball_detections: List of ball detection dictionaries for each frame.
        :return: List of video frames with bounding boxes drawn.
        """
        output_video_frames = []

        # Draw bounding boxes on each frame
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Ensure ball_dict is a dictionary
            if isinstance(ball_dict, dict):
                for track_id, bbox in ball_dict.items():
                    x1, y1, x2, y2 = bbox
                    # Draw the label "Tennis Ball" above the bounding box
                    cv2.putText(
                        frame,
                        "Tennis Ball",
                        (int(bbox[0]), int(bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                    )
                    # Draw the bounding box around the ball
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2
                    )
            output_video_frames.append(frame)

        return output_video_frames
