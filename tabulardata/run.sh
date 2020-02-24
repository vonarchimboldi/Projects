export TRAINING_DATA=data/primary/amazon/train.csv
export TEST_DATA=data/primary/amazon/test.csv
export VALIDATION_DATA=data/primary/amazon/validation_set.csv
export CALIBRATION_SET=data/primary/amazon/calibration_set.csv
export PROBLEM_TYPE=classification
export CALIBRATIONMETHOD='isotonic'

export DATASET=amazon
export MODEL=$1
python -m src.train
python -m src.predict
python -m src.intervals
python -m src.calibrate