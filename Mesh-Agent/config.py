GT_VOX_RESO_L = 16
GT_VOX_RESO_H = 64
REF_IMG_RESO = 128

LOOP_NUM=10
ACTION_NUM=360
LOOP_FEAT_DIM=4
MAX_STEP=100

# hyperparamters for training
BATCH_SIZE = 64
LR = 0.00008                 # learning rate
EPSILON = 0.98               # greedy policy
GAMMA = 0.9                  # reward discount
TARGET_REPLACE_ITER = 4000   # target update frequency
MEMORY_LONG_CAPACITY = 200000
MEMORY_SELF_CAPACITY = 100000 # shared by D_short and D_self
DAGGER_EPOCH=1
DAGGER_ITER=4
DAGGER_LEARN=4000
RL_EPOCH=2000

                                                                                                                                                                                                                                                                        
