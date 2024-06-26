from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
TRAINING_DIR = BASE_DIR / 'training' 

# Data Directories
DATA_DIR = BASE_DIR / 'data'
KEYPOINTS_DIR = DATA_DIR / 'keypoints'
TENNIS_BALL_DIR = DATA_DIR / 'tennis_balls'

UTILS_DIR = BASE_DIR / 'utils'


directories = [DATA_DIR, MODELS_DIR, TRAINING_DIR, TENNIS_BALL_DIR, KEYPOINTS_DIR]
for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)
    
