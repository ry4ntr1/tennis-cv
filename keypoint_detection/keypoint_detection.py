import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models


class KeypointDetector:
    def __init__(self, model_path):
        # Set device to MPS (Metal Performance Shaders) if available, otherwise use CPU
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Load a ResNet-50 model without pre-trained weights
        self.model = models.resnet50(weights=None)

        # Replace the final fully connected layer to output 28 values (14 keypoints with x, y coordinates)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)

        # Load model weights from the given path
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Move the model to the specified device
        self.model.to(self.device)

        # Set the model to evaluation mode
        self.model.eval()

        # Define image transformation pipeline
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, image):
        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations to the image and add a batch dimension
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Run the model in inference mode with no gradient calculation
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Remove the batch dimension and convert the tensor to a NumPy array
        keypoints = outputs.squeeze().cpu().numpy()

        # Rescale keypoints to the original image size
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints

    def draw_keypoints(self, image, keypoints, color=(0, 0, 255), text_size=0.5):
        # Iterate through the keypoints and draw them on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv2.putText(
                image,
                str(i // 2),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_size,
                color,
                2,
            )
            cv2.circle(image, (x, y), 5, color, -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        # Draw keypoints on each frame of the video
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
