# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

# Set up the working directories.
DD_FOLDER="DD_full"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${DD_FOLDER}/${EXP_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DD_FOLDER}/${EXP_FOLDER}/train_res"
DATASET="${WORK_DIR}/${DATASET_DIR}/${DD_FOLDER}/tfrecord"

mkdir -p "${WORK_DIR}/${DATASET_DIR}/${DD_FOLDER}/exp"
mkdir -p "${TRAIN_LOGDIR}"

NUM_ITERATIONS=29200
python3 "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="resnet_v1_50_beta" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=300,300 \
  --train_batch_size=8 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=False \
  --tf_initial_checkpoint="${INIT_FOLDER}/resnet_v1_50/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}"