from pathlib import Path

# Constants
SINGLE_LINE_WIDTH = 8.23
DOUBLE_LINE_WIDTH = 10.97
HALF_COURT_LINE_HEIGHT = 11.88
SERVICE_LINE_WIDTH = 6.4
DOUBLE_ALLY_DIFFERENCE = 1.37
NO_MANS_LAND_HEIGHT = 5.48

PLAYER_1_HEIGHT_METERS = 1.88
PLAYER_2_HEIGHT_METERS = 1.91

# Directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
TRAINING_DIR = BASE_DIR / 'training' 
SAMPLE_DATA_DIR = BASE_DIR / 'sample_data'

# Data Directories
DATA_DIR = BASE_DIR / 'data'
KEYPOINTS_DIR = DATA_DIR / 'keypoints'
TENNIS_BALL_DIR = DATA_DIR / 'tennis_balls'

# Utils Directory
UTILS_DIR = BASE_DIR / 'utils'

# Output Directories
TEST_OUTPUT_DIR = BASE_DIR / 'test_output'
TRACKER_STUB_DIR = BASE_DIR / 'tracker_stubs'

directories = [ MODELS_DIR, TRAINING_DIR, SAMPLE_DATA_DIR,TEST_OUTPUT_DIR, TRACKER_STUB_DIR, UTILS_DIR, DATA_DIR, TENNIS_BALL_DIR, KEYPOINTS_DIR]
for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)