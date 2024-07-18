from ultralytics import YOLO
import cv2
import pickle
import pandas as pd


class BallTracker:
    def __init__(self, model_path):
        # Initialize the YOLO model with the given model path
        self.model = YOLO(model_path)

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
