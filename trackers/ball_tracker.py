from ultralytics import YOLO
import cv2
import pickle
import pandas as pd


class BallTracker:
    def __init__(self, model_path):
        # Initialize the YOLO model with the given model path
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
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
        ball_detections = []

        # Load detections from stub file if specified
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections

        # Detect balls in each frame
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        # Save detections to stub file if specified
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
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
        output_video_frames = []
        # Draw bounding boxes on each frame
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Ensure ball_dict is a dictionary
            if isinstance(ball_dict, dict):
                for track_id, bbox in ball_dict.items():
                    x1, y1, x2, y2 = bbox
                    cv2.putText(
                        frame,
                        f"Ball ID: {track_id}",
                        (int(bbox[0]), int(bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                    )
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2
                    )
            output_video_frames.append(frame)

        return output_video_frames
