DATA_PATH = "data/Dataset_with_RainStatus_V5.csv"
DATE_COL = "Date"
RAIN_COL = "PRECTOTCORR"
TEMP_MAX_COL = "T2M_MAX"
TEMP_MIN_COL = "T2M_MIN"

WINDOW = 30
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 42

# Rain classification threshold
RAIN_PROB_THRESHOLD = 0.50

# Weighted rainfall loss settings, in real mm/day before scaling/log transform.
# These are converted to scaled log thresholds inside training.
RAIN_WEIGHT_THRESHOLDS_MM = [10.0, 20.0, 40.0]
RAIN_WEIGHT_VALUES = [2.0, 3.0, 5.0]
