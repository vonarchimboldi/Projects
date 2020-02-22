export TRAINING_DATA=data/primary/sarcos_inv/train.csv
export TEST_DATA=data/primary/sarcos_inv/test.csv
export VALIDATION_DATA=data/primary/sarcos_inv/validation_set.csv
export CALIBRATION_SET=data/primary/sarcos_inv/calibration_set.csv
export PROBLEM_TYPE=regression

export DATASET=sarcos_inv
export MODEL=$1
python -m src.train
python -m src.predict
python -m src.intervals