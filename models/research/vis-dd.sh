CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

DD_FOLDER="DD_full"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${DD_FOLDER}/${EXP_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DD_FOLDER}/${EXP_FOLDER}/train"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DD_FOLDER}/${EXP_FOLDER}/vis"
DATASET="${WORK_DIR}/${DATASET_DIR}/${DD_FOLDER}/tfrecord"

python3 "${WORK_DIR}"/vis.py \
--logtostderr \
--vis_split="tet" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--vis_crop_size=300,300 \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--vis_logdir="${VIS_LOGDIR}" \
--dataset_dir="${DATASET}" \
--max_number_of_iterations=1