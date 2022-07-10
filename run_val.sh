#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1
export EXPERIMENT=$2
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export DATASET=${DATASET:-ScannetSparseVoxelizationDataset}
export MODEL=${MODEL:-Res16UNet34C}
export OPTIMIZER=${OPTIMIZER:-SGD}
export LR=${LR:-1e-2}
export BATCH_SIZE=${BATCH_SIZE:-12}
export SCHEDULER=${SCHEDULER:-SquaredLR}
export MAX_ITER=${MAX_ITER:-60000}

export OUTPATH=./outputs/$DATASET/$MODEL/$LOG/
export IS_EXPORT=${IS_EXPORT:-False}

export SUBMIT=${SUBMIT:-False}


#export VERSION=$(git rev-parse HEAD)

# We use the best-in-validation weights
# for test submission
if ${SUBMIT}
then
	export WEIGHTS=${OUTPATH}/weights.pth
else
	export WEIGHTS=${OUTPATH}/checkpoint_${MODEL}best_val.pth
fi

# Save the experiment detail and dir to the common log file
mkdir -p $OUTPATH

rm ./models
echo 'ln -s $OUTPATH/models ./models'
ln -s $OUTPATH/models ./models

# put the arguments on the first line for easy resume
echo "
    --log_dir $OUTPATH \
    --dataset $DATASET \
    --model $MODEL \
    --train_limit_numpoints 1200000 \
    --lr $LR \
    --optimizer $OPTIMIZER \
    --batch_size $BATCH_SIZE \
    --scheduler $SCHEDULER \
    --max_iter $MAX_ITER \
    $3" 

echo Logging output to "$LOG"
#echo $(pwd) >> $LOG
#echo "Version: " $VERSION >> $LOG
#echo "Git diff" >> $LOG
#echo "" >> $LOG
#git diff | tee -a $LOG
#echo "" >> $LOG
#nvidia-smi | tee -a $LOG

#time python -W ignore main.py \
	#--log_dir $OUTPATH \
	#--dataset $DATASET \
	#--model $MODEL \
	#--train_limit_numpoints 1200000 \
	#--lr $LR \
	#--optimizer $OPTIMIZER \
	#--batch_size $BATCH_SIZE \
	#--scheduler $SCHEDULER \
	#--max_iter $MAX_ITER \
	#$3 

echo 'WHETHER SUBMIY ::::'+${SUBMIT}

time python -W ignore main.py \
	--log_dir $OUTPATH \
	--test_config ${OUTPATH}config.json \
	--weights ${WEIGHTS} \
	--val_batch_size $BATCH_SIZE \
	--is_export $IS_EXPORT \
	--submit $SUBMIT \
	$3 

rm ./models
ln -s ./models_ models
