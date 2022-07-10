export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

export BATCH_SIZE=2
export ITER_SIZE=1

export MODEL=Res16UNetTestA

#export OPTIMIZER=SGD
export OPTIMIZER=Adam

export DATASET=SemanticKITTI

export LR=2e-3
export MAX_ITER=15000
export MAX_POINTS=150000  # bs=4, cont-attn

export MP=True
export VOXEL_SIZE=0.05
export WEIGHT_DECAY=1.e-5

export LOG=$1
#export ENABLE_POINT_BRANCH=True # SPVCNN feature

./run.sh $2 \
		-default 

