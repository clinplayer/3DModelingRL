GT_VOX_RESO_L = 16
GT_VOX_RESO_H = 64
REF_IMG_RESO = 128

BOX_NUM=27
ACTION_NUM=756
LOOP_NUM=10
MAX_STEP=300

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
RL_EPOCH=200

COLORS= [[166,206,227 ],[ 31,120,180 ],[ 178,223,138 ],[ 51,160,44 ],\
[251,154,153 ],[ 227,26,28 ],[ 253,191,111 ],[ 255,127,0 ],\
[ 166,216,84 ],[ 106,61,154 ],[ 255,255,153 ],[ 177,89,40 ],\
[ 102,194,165 ],[ 252,141,98 ],[ 141,160,203 ],[ 231,138,195 ],\
[ 166,216,84 ],[ 255,217,47 ],[27,158,119],[217,95,2],\
[117,112,179], [231,41,138], [102,166,30], [230,171,2], \
[166,118,29], [102,102,102], [229,196,148]]