TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/train.csv"

NEW_TRAIN_PATH = "data/processed/train_processed.csv"
NEW_TEST_PATH = "data/processed/test_processed.csv"
MODELS_OUTPUT = 'models/'

ID = "PassengerId"
TARGET = "Survived"
MAX_RUNTIME_SECS = 60 * 3
DROP_COLS = [ID, "Name", "Ticket", "Cabin"]
SEED = 2022
