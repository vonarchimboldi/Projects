export TRAINING_DATA=data/raw/sarcos_inv/train.csv
export TEST_DATA=data/raw/sarcos_inv/test.csv
export VALIDATION_DATA=data/primary/validation_set.csv
export CALIBRATION_SET=data/primary/calibration_set.csv

export MODEL=$1

python -m src.load_data