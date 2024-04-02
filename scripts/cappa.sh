# !/bin/bash

export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets

cd ~/big_vision
python -m big_vision.trainers.proj.cappa.generative \
  --config big_vision/configs/proj/cappa/pretrain.py
  # --config big_vision/configs/proj/cappa/cappa_replication.py
  