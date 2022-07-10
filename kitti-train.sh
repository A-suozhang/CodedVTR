export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

export BATCH_SIZE=2
export ITER_SIZE=1
export MODEL=Res16UNetTestA

export OPTIMIZER=SGD

export DATASET=SemanticKITTI

export MAX_ITER=15000
export MAX_POINTS=2400000
export LR=5e-2
#export MP=True

export VOXEL_SIZE=0.1
export WEIGHT_DECAY=1.e-4

export LOG=$1
# export ENABLE_POINT_BRANCH=True # SPVCNN feature

./run.sh $2 \
		-default 

