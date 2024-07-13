import pickle
import cv2
from roboflow import Roboflow
import os


class BallTracker:
    def __init__(self, api_key, project_name, version_number):
        """
        Initializes the BallTracker with a Roboflow model.
        :param api_key: API key for Roboflow.
        :param project_name: Name of the project in Roboflow.
        :param version_number: Version number of the model in Roboflow.
        """
        rf = Roboflow(api_key=api_key)
        self.project = rf.workspace().project(project_name)
        self.model = self.project.version(version_number).model

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detects balls in a list of frames.
        :param frames: List of frames to be processed.
        :param read_from_stub: Flag indicating whether to read detections from a pre-saved file.
        :param stub_path: Path to the file for saving/loading detections.
        :return: List of ball detections for each frame.
        """
        if read_from_stub and stub_path:
            return self._load_detections(stub_path)

        ball_detections = []
        for i, frame in enumerate(frames):
            print(f"Processing frame {i + 1}/{len(frames)}")
            ball_detections.append(self.detect_frame(frame))

        if stub_path:
            self._save_detections(ball_detections, stub_path)

        return ball_detections

    def detect_frame(self, frame):
        """
        Detects tennis balls in a single frame.
        :param frame: Frame to be processed.
        :return: Dictionary of detected tennis balls with their bounding boxes.
        """
        temp_filename = "temp_frame.jpg"
        cv2.imwrite(temp_filename, frame)
        print(f"Saved frame to {temp_filename}")

        try:
            predictions = self.model.predict(
                temp_filename, confidence=40, overlap=30
            ).json()
            print(f"Predictions: {predictions}")

            ball_dict = {}
            for i, pred in enumerate(predictions["predictions"]):
                if pred["class"] == "tennis_ball":
                    x1 = int(pred["x"] - pred["width"] / 2)
                    y1 = int(pred["y"] - pred["height"] / 2)
                    x2 = int(pred["x"] + pred["width"] / 2)
                    y2 = int(pred["y"] + pred["height"] / 2)
                    ball_dict[i] = [x1, y1, x2, y2]

            return ball_dict
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {}

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                print(f"Deleted temporary file {temp_filename}")

    def draw_bboxes(self, video_frames, ball_detections):
        """
        Draws bounding boxes around detected balls in video frames.
        :param video_frames: List of video frames.
        :param ball_detections: List of ball detection dictionaries for each frame.
        :return: List of video frames with bounding boxes drawn.
        """
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            self._draw_frame_bboxes(frame, ball_dict)
            output_video_frames.append(frame)

        return output_video_frames

    def _draw_frame_bboxes(self, frame, ball_dict):
        """
        Helper method to draw bounding boxes on a single frame.
        :param frame: Frame to be processed.
        :param ball_dict: Dictionary of detected balls with their bounding boxes.
        """
        for track_id, bbox in ball_dict.items():
            x1, y1, x2, y2 = bbox
            cv2.putText(
                frame,
                f"Tennis_Ball: {track_id}",
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    def _save_detections(self, detections, path):
        """
        Saves the detection results to a file.
        :param detections: List of detection results.
        :param path: Path to the file where detections will be saved.
        """
        try:
            with open(path, "wb") as f:
                pickle.dump(detections, f)
        except Exception as e:
            print(f"Error saving detections: {e}")

    def _load_detections(self, path):
        """
        Loads the detection results from a file.
        :param path: Path to the file from which detections will be loaded.
        :return: List of detection results.
        """
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading detections: {e}")
            return []
