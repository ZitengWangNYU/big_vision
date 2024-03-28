# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/big_vision
python -m big_vision.trainers.proj.image_text.siglip \
    --config big_vision/configs/proj/image_text/siglip_lit_laion400m.py 
    # siglip_lit_laion400m, siglip_lit_coco, siglip_replication