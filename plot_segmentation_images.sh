#!/bin/bash

# 你要畫的類別，改這裡就好（bottle, capsule, grid, pill, ...）
DATASET_NAME=bottle

# 對應 segmentation & mask 的 npy 存放目錄
CKPT_DIR=results/MVTecAD_Results_Img/simplenet_mvtec/run/models/0/mvtec_${DATASET_NAME}

# 對應原始圖片資料夾 (test images)
IMAGE_FOLDER=data4/MVTec_ad/${DATASET_NAME}/test

# 對應 ground truth mask 資料夾
MASK_FOLDER=data4/MVTec_ad/${DATASET_NAME}/ground_truth

# 你想要把結果存在哪裡
SAVE_FOLDER=${CKPT_DIR}/vis_${DATASET_NAME}

# 執行 plot_segmentation_images.py
python plot_segmentation_images.py \
    --image_folder ${IMAGE_FOLDER} \
    --segmentation_file ${CKPT_DIR}/segmentations_${DATASET_NAME}.npy \
    --savefolder ${SAVE_FOLDER} \
    --mask_folder ${MASK_FOLDER}

echo "✅ Segmentation visualization for ${DATASET_NAME} saved to ${SAVE_FOLDER}"

