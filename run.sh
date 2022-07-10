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
#export LR=${LR:-1e-3}
export LR=${LR:-1e-2}
export BATCH_SIZE=${BATCH_SIZE:-12}
export ITER_SIZE=${ITER_SIZE:-1}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1E-4}
export SCHEDULER=${SCHEDULER:-SquaredLR}
export MAX_ITER=${MAX_ITER:-60000}
export MAX_POINTS=${MAX_POINTS:-120000}
#export EPOCHS=${EPOCHS:-200}

export RESUME=${RESUME:-none}
# export RESUME=${RESUME:-True}
export POINTS=${POINTS:-8192}
export VOXEL_SIZE=${VOXEL_SIZE:-0.1}
export ENABLE_POINT_BRANCH=${ENABLE_POINT_BRANCH:-false}

export USE_AUX=${USE_AUX:-false}
export DISTILL=${DISTILL:-false}
export MP=${MP:-False}


export OUTPATH=./outputs/$DATASET/$MODEL/$LOG/
#export VERSION=$(git rev-parse HEAD)

export IS_DEBUG=${IS_DEBUG:-none}

# Save the experiment detail and dir to the common log file
mkdir -p $OUTPATH

# put the arguments on the first line for easy resume
echo "
    --log_dir $OUTPATH \
    --dataset $DATASET \
    --model $MODEL \
	--train_limit_numpoints 1200000 \
    --lr $LR \
    --optimizer $OPTIMIZER \
	--weight_decay $WEIGHT_DECAY \
    --batch_size $BATCH_SIZE \
    --iter_size $ITER_SIZE \
    --scheduler $SCHEDULER \
    --max_iter $MAX_ITER \
	--voxel_size $VOXEL_SIZE \
	--submit $SUBMIT
    $3" 

echo Logging output to "$LOG"
#echo $(pwd) >> $LOG
#echo "Version: " $VERSION >> $LOG
#echo "Git diff" >> $LOG
#echo "" >> $LOG
#git diff | tee -a $LOG
#echo "" >> $LOG
#nvidia-smi | tee -a $LOG

rm ./models
ln -s ./models_ ./models

time python -W ignore main.py \
	--log_dir $OUTPATH \
	--dataset $DATASET \
	--model $MODEL \
	--train_limit_numpoints $MAX_POINTS \
	--lr $LR \
	--optimizer $OPTIMIZER \
	--weight_decay $WEIGHT_DECAY \
	--batch_size $BATCH_SIZE \
	--iter_size $ITER_SIZE \
	--scheduler $SCHEDULER \
	--max_iter $MAX_ITER \
	--num_points $POINTS \
	--resume $RESUME \
	--use_aux $USE_AUX \
	--distill $DISTILL \
	--multiprocess $MP \
	--voxel_size $VOXEL_SIZE \
	--is_debug $IS_DEBUG \
	--enable_point_branch $ENABLE_POINT_BRANCH \
	$3 

#time python -W ignore main.py \
    #--log_dir $OUTPATH \
    #--test_config ${OUTPATH}config.json \
    #--weights ${OUTPATH}weights.pth \
    #$3 
	##2>&1 | tee -a "$LOG"
