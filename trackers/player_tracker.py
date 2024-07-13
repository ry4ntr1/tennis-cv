import pickle
import cv2
from ultralytics import YOLO
import sys

sys.path.append("../utils")


class PlayerTracker:
    def __init__(self, model_path):
        """
        Initializes the PlayerTracker with a YOLO model.
        :param model_path: Path to the YOLO model.
        """
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detects players in a list of frames.
        :param frames: List of frames to be processed.
        :param read_from_stub: Flag indicating whether to read detections from a pre-saved file.
        :param stub_path: Path to the file for saving/loading detections.
        :return: List of player detections for each frame.
        """
        if read_from_stub and stub_path:
            return self._load_detections(stub_path)

        player_detections = [self.detect_frame(frame) for frame in frames]

        if stub_path:
            self._save_detections(player_detections, stub_path)

        return player_detections

    def detect_frame(self, frame):
        """
        Detects players in a single frame.
        :param frame: Frame to be processed.
        :return: Dictionary of detected players with their bounding boxes.
        """
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            if box.id is not None:  # Check if box.id is not None
                track_id = int(box.id.tolist()[0])
                bbox = box.xyxy.tolist()[0]
                object_cls_id = box.cls.tolist()[0]
                object_cls_name = id_name_dict[object_cls_id]

                if object_cls_name == "players":
                    player_dict[track_id] = bbox

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        """
        Draws bounding boxes around detected players in video frames.
        :param video_frames: List of video frames.
        :param player_detections: List of player detection dictionaries for each frame.
        :return: List of video frames with bounding boxes drawn.
        """
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            self._draw_frame_bboxes(frame, player_dict)
            output_video_frames.append(frame)

        return output_video_frames

    def _draw_frame_bboxes(self, frame, player_dict):
        """
        Helper method to draw bounding boxes on a single frame.
        :param frame: Frame to be processed.
        :param player_dict: Dictionary of detected players with their bounding boxes.
        """
        for track_id, bbox in player_dict.items():
            x1, y1, x2, y2 = bbox
            cv2.putText(
                frame,
                f"Player ID: {track_id}",
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

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
