import cv2
from tqdm import tqdm


def load_video_frames(video_path):
    """
    Loads all frames from a specified video file.

    Parameters:
    video_path (str): Path to the video file.

    Returns:
    list: A list containing all frames from the video as numpy arrays.
    """

    # Create a VideoCapture object to read frames from the video file
    video_capture = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not video_capture.isOpened():
        raise IOError(f"Unable to open video file at: {video_path}")

    video_frames = []
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc="Loading Video Frames..."):
        # Read a whether frame was read, and the actual frame from the video
        frame_success, frame = video_capture.read()

        if not frame_success:
            break

        video_frames.append(frame)

    # Release the VideoCapture object
    video_capture.release()

    return video_frames


def export_video(video_frames, output_path, fps=24):
    """
    Exports a list of video frames to a new video file.

    Parameters:
    video_frames (list): A list containing frames to be written to the video.
    output_path (str): Path to save the output video file.
    fps (int): Frames per second for the output video.
    """

    # Check if the list of frames is empty
    if not video_frames:
        raise ValueError("Unable to export video frames. The frame list is empty.")

    # Get the height and width of the frames from the first frame in the list
    height, width, _ = video_frames[0].shape
    size = (width, height)

    # Define the codec for the video file (here, "mp4v" for MPEG-4 encoding)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Create a VideoWriter object to write the frames to the video file
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    # Check if the VideoWriter object was successfully created
    if not out.isOpened():
        raise IOError(f"Unable to create video file at: {output_path}")

    for frame in tqdm(video_frames, desc="Exporting Video Analysis..."):
        out.write(frame)

    out.release()

    print(f"\nSaved To: {output_path}")
