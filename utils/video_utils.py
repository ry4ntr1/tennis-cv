import cv2

def load_video_frames(video_path):
    """
    Loads all frames from a specified video file.

    Parameters:
    video_path (str): Path to the video file.

    Returns:
    list: A list containing all frames from the video as numpy arrays.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise IOError(f"Cannot open video file at {video_path}")

    video_frames = []
    while True:
        frame_success, frame = video_capture.read()
        if not frame_success:
            break
        video_frames.append(frame)
    video_capture.release()
    return video_frames

def export_video(frames_list, output_path):
    """
    Saves a list of frames as a video file.

    Parameters:
    frames_list (list): List of video frames to be saved.
    output_path (str): Path where the output video will be saved.

    Raises:
    ValueError: If frames_list is empty.
    """
    if not frames_list:
        raise ValueError("The list of frames is empty.")

    frame_height, frame_width = frames_list[0].shape[:2]
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 24, (frame_width, frame_height))

    for frame in frames_list:
        video_writer.write(frame)
    video_writer.release()
