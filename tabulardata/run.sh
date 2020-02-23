export TRAINING_DATA=data/primary/poker_hands/train.csv
export TEST_DATA=data/primary/poker_hands/test.csv
export VALIDATION_DATA=data/primary/poker_hands/validation_set.csv
export CALIBRATION_SET=data/primary/poker_hands/calibration_set.csv
export PROBLEM_TYPE=multiclass

export DATASET=poker_hands
export MODEL=$1
python -m src.train
python -m src.predict
#python -m src.intervals