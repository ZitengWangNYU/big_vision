# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/my_repo/big_vision
python -m big_vision.trainers.proj.image_text.siglip \
    --config big_vision/configs/proj/image_text/siglip_replication.py:\
    batch_size=1024, scan=True, fsdp=4, dtype='float32', debug=True
    # --workdir gs://us-central2-storage/tensorflow_datasets/siglip_replication_`date '+%m-%d_%H%M'`

    # --config big_vision/configs/proj/image_text/siglip_lit_laion400m.py \
    # --workdir gs://us-central2-storage/tensorflow_datasets/siglip_lit_laion400m_`date '+%m-%d_%H%M'`
    # --config big_vision/configs/proj/image_text/siglip_lit_coco.py