export TRAINING_DATA=input/bank-train.csv
export VALIDATION_DATA=input/validation_set.csv
export TEST_DATA=input/bank-test.csv
export CALIBRATION_SET=input/calibration_set.csv
export FEATURE_SELECTION=NO
export FEATURE_SELECTION_METHOD=vif
export PROBLEM_TYPE=classification
export LOSS=quantile
export MODEL=$1
export LOSS=other
export CALIBRATION=YES
export alpha=0.95
export NUM_FOLDS=1

python -m src.createfolds
#FOLD=0 python -m src.train
#FOLD=1 python -m src.train
FOLD=2 python -m src.train
#FOLD=3 python -m src.train
#FOLD=4 python -m src.train
python -m src.predict
python -m src.intervals