export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/bank-test.csv
export FEATURE_SELECTION=NO
export FEATURE_SELECTION_METHOD=mixed
export MODEL=$1
export LOSS=log
export CALIBRATION=YES
export alpha =0.95
export NUM_FOLDS =1

#FOLD=0 python -m src.train
#FOLD=1 python -m src.train
#FOLD=2 python -m src.train
#FOLD=3 python -m src.train
#FOLD=4 python -m src.train
#python -m src.predict