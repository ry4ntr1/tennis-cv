import pickle
import cv2
from ultralytics import YOLO
import sys
import pandas as pd
from utils import approx_center, euclidean_distance
from tqdm import tqdm
import logging
# Add the "../utils" directory to the system path to import custom utilities if needed
sys.path.append("../utils")

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

class PlayerTracker:
    def __init__(self, model_path):
        """
        Initialize the PlayerTracker with the YOLO model from the specified path.

        :param model_path: Path to the YOLO model file.
        """
        self.model = YOLO(model_path)

    def interpolate_player_positions(self, player_positions):
        """
        Interpolate missing player positions in a sequence of frames.

        :param player_positions: List of dictionaries containing player positions for each frame.
        :return: Interpolated list of player positions.
        """
        # Extract coordinates from player_positions, replacing empty positions with [None, None, None, None]
        extracted_positions = []
        for pos in player_positions:
            if len(pos) == 2:
                coordinates = list(pos.values())
                extracted_positions.append(coordinates)
            else:
                extracted_positions.append(
                    [[None, None, None, None], [None, None, None, None]]
                )

        # Flatten the list of coordinates for the DataFrame
        flattened_positions = []
        for frame in extracted_positions:
            flattened_positions.extend(frame)

        # Return original player_positions if extracted_positions is empty or contains only None values
        if not flattened_positions or all(
            len(pos) == 0 or all(v is None for v in pos) for pos in flattened_positions
        ):
            return player_positions

        try:
            # Create a DataFrame from the extracted positions
            df_player_positions = pd.DataFrame(
                flattened_positions, columns=["x1", "y1", "x2", "y2"]
            )
        except ValueError as e:
            # Print the contents causing the ValueError for debugging
            print("ValueError:", e)
            print("flattened_positions causing error:", flattened_positions)
            raise e

        # Interpolate missing values and backfill remaining NaNs
        df_player_positions = df_player_positions.interpolate()
        df_player_positions = df_player_positions.bfill()

        # Convert DataFrame back to list of dictionaries
        player_positions = []
        for i in range(0, len(df_player_positions), 2):
            player_positions.append(
                {
                    1: df_player_positions.iloc[i].tolist(),
                    2: df_player_positions.iloc[i + 1].tolist(),
                }
            )

        return player_positions

    def choose_and_filter_players(self, court_keypoints, player_detections):
        """
        Choose key players based on court keypoints and filter the detected players.

        :param court_keypoints: List of court keypoints.
        :param player_detections: List of dictionaries containing player detections for each frame.
        :return: Filtered list of player detections.
        """
        # Choose players from the first frame based on court keypoints
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(
            court_keypoints, player_detections_first_frame
        )

        # Filter player detections to keep only chosen players
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {
                track_id: bbox
                for track_id, bbox in player_dict.items()
                if track_id in chosen_player
            }
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        """
        Select the two players closest to the court keypoints.

        :param court_keypoints: List of court keypoints.
        :param player_dict: Dictionary of detected players with their bounding boxes.
        :return: List of chosen player track IDs.
        """
        distances = []
        for track_id, bbox in player_dict.items():
            # Calculate the approximate center of the player bounding box
            player_center = approx_center(bbox)

            # Find the minimum distance from the player center to the court keypoints
            min_distance = float("inf")
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = euclidean_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # Sort the distances in ascending order
        distances.sort(key=lambda x: x[1])

        # Ensure that there are at least two players detected
        if len(distances) < 2:
            print("Warning: Not enough players detected to choose two")
            return []

        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect players in a list of frames.

        :param frames: List of frames to be processed.
        :param read_from_stub: Flag indicating whether to read detections from a pre-saved file.
        :param stub_path: Path to the file for saving/loading detections.
        :return: List of player detections for each frame.
        """
        if read_from_stub and stub_path:
            return self._load_detections(stub_path)

        # Detect players in each frame with progress tracking using tqdm
        player_detections = []
        for frame in tqdm(frames, desc="Detecting Players"):
            detection = self.detect_frame(frame)
            player_detections.append(detection)

        # If stub_path is provided, save player detections to .pkl file
        if stub_path:
            self._save_detections(player_detections, stub_path)

        return player_detections

    def detect_frame(self, frame):
        """
        Detect players in a single frame.

        :param frame: Frame to be processed.
        :return: Dictionary of detected players with their bounding boxes.
        """
        # Perform tracking on the single frame
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        # Dictionary to store detected players with their bounding boxes
        player_dict = {}

        # Loop through each detected bounding box
        for box in results.boxes:
            # Ensure box.id is not None before proceeding
            if box is None:
                print("Empty: ", box)

            if box.id is not None:
                track_id = int(box.id.tolist()[0])
                bbox = box.xyxy.tolist()[0]
                object_cls_id = box.cls.tolist()[0]
                object_cls_name = id_name_dict[object_cls_id]

                # Check if the detected object is a player
                if object_cls_name == "players":
                    player_dict[track_id] = bbox

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        """
        Draw bounding boxes around detected players in video frames.

        :param video_frames: List of video frames.
        :param player_detections: List of player detection dictionaries for each frame.
        :return: List of video frames with bounding boxes drawn.
        """
        # List to store video frames with bounding boxes
        output_video_frames = []

        # Loop through each frame and its corresponding player detections
        for frame, player_dict in zip(video_frames, player_detections):
            self._draw_frame_bboxes(
                frame, player_dict
            )  # Draw bounding boxes on the frame
            output_video_frames.append(frame)  # Add the annotated frame to the list

        return output_video_frames

    def _draw_frame_bboxes(self, frame, player_dict):
        """
        Helper method to draw bounding boxes on a single frame.

        :param frame: Frame to be processed.
        :param player_dict: Dictionary of detected players with their bounding boxes.
        """
        # Loop through each player detection
        for track_id, bbox in player_dict.items():
            x1, y1, x2, y2 = bbox  # Get the bounding box coordinates

            # Draw the tracking ID text above the bounding box
            cv2.putText(
                frame,
                f"Player ID: {track_id}",
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
            # Draw the bounding box around the player
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    def _save_detections(self, detections, path):
        """
        Save the detection results to a file.

        :param detections: List of detection results.
        :param path: Path to the file where detections will be saved.
        """
        try:
            # Open the file in write-binary mode and save detections using pickle
            with open(path, "wb") as f:
                pickle.dump(detections, f)
        except Exception as e:
            print(f"Error saving detections: {e}")

    def _load_detections(self, path):
        """
        Load the detection results from a file.

        :param path: Path to the file from which detections will be loaded.
        :return: List of detection results.
        """
        try:
            # Open the file in read-binary mode and load detections using pickle
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading detections: {e}")
            return []
