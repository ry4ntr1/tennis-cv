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

import cv2

def export_video(video_frames, output_path):
    if not video_frames:
        raise ValueError("No video frames to export.")
    
    height, width, layers = video_frames[0].shape
    size = (width, height)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_path, fourcc, 24, size)  
    
    for frame in video_frames:
        out.write(frame)
    
    out.release()
    print(f"Video exported to {output_path}")

