r"""SigLIP (https://arxiv.org/abs/2303.15343) Replication.

Training receipe:
- train dataset: laion400m/images
  - metadata info for the downloaded data in gcloud storage:
    - total_num_bytes: 7,927,590,520,336 (7.9TB)
    - len(shard_lengths): 62,917 (number of tfrecord files)
    - total_samples: 327,702,052 (328M)
  - train
    - image resolution: 224*224
    - tokenizer: 32K vocabulary sentencepiece; trained on C4 dataset; output has 16 maximum tokens 
    - batch size: 32,768 (TPUv4-32), 4,096 (TPUv4-8 & SigLIP Base), 2,048 (TPUv4-8 & CLIP Base)
    - optimizer: β2 = 0.95 (not 0.999, stablizing the training)
    - strategy: from scratch < unlocked ft. on pre-trained ViT-AugReg-B/16 < ft. without weight decay (Figure 4)
    - combo: TPUv4-32 & 16,384 & 2.4B (Figure 4); 
  - model
    - image
      - ViT-B/16 # TO_LEARN: what is ViT-AugReg-B/16
      - TO_DETERMINE: head or no head?
      - randomly initialized
      - from scratch

Bash for training:

big_vision.trainers.proj.image_text.siglip \
    --config big_vision/configs/proj/image_text/siglip_replication.py:batch_size=512 \
    --workdir gs://$GS_BUCKET_NAME/big_vision/`date '+%Y-%m-%d_%H%M'`
"""

import big_vision.configs.common as bvcc
from big_vision.configs.proj.image_text import common
from ml_collections import ConfigDict


def get_config(arg=None):
  """The base configuration."""
  arg = bvcc.parse_arg(
      arg, res=224, runlocal=False, token_len=16, init='', img_head=False, batch_size=512)
  config = ConfigDict()

  config.input = {}
  config.input.data = dict(name='laion400m/images', split='train')
  config.input.batch_size = arg.batch_size if not arg.runlocal else 32
  config.input.shuffle_buffer_size = 250_000  if not arg.runlocal else 50
  config.input.pack = True # TO_DETERMINE: pack or not pack? TODO: examine the variance of sequence length

  config.total_steps = 6000 if not arg.runlocal else 1 # DEBUG 3B/512=5_859_375

  config.init_shapes = [(1, arg.res, arg.res, 3), (1, arg.token_len,)] # TO_LEARN: where is it used?
  config.init_types = ['float32', 'int32'] # TO_LEARN: where is it used?
  VARIANT, RES = 'B/16', 224
  CKPT, TXTVARIANT, EMBDIM, SEQLEN, VOCAB = {
      ('B/16', 224): ('/mnt/vlm-pd/ckpts/siglip/webli_en_b16_224_63724782.npz', 'B', 768, 64, 32_000),
      ('B/16', 256): ('/mnt/vlm-pd/ckpts/siglip/webli_en_b16_256_60500360.npz', 'B', 768, 64, 32_000),
      ('B/16', 384): ('/mnt/vlm-pd/ckpts/siglip/webli_en_b16_384_68578854.npz', 'B', 768, 64, 32_000),
      ('B/16', 512): ('/mnt/vlm-pd/ckpts/siglip/webli_en_b16_512_68580893.npz', 'B', 768, 64, 32_000),
      ('L/16', 256): ('/mnt/vlm-pd/ckpts/siglip/webli_en_l16_256_60552751.npz', 'L', 1024, 64, 32_000),
      ('L/16', 384): ('/mnt/vlm-pd/ckpts/siglip/webli_en_l16_384_63634585.npz', 'L', 1024, 64, 32_000),
      ('So400m/14', 224): ('/mnt/vlm-pd/ckpts/siglip/webli_en_so400m_224_57633886.npz', 'So400m', 1152, 16, 32_000),
      ('So400m/14', 384): ('/mnt/vlm-pd/ckpts/siglip/webli_en_so400m_384_58765454.npz', 'So400m', 1152, 64, 32_000),
      ('B/16-i18n', 256): ('/mnt/vlm-pd/ckpts/siglip/webli_i18n_b16_256_66117334.npz', 'B', 768, 64, 250_000),
  }[VARIANT, RES]

  TOKENIZERS = {
      32_000: 'c4_en',
      250_000: 'mc4',
  }

  tokenizer = lambda inkey: (
      # f'tokenize(max_len={SEQLEN}, model="{TOKENIZERS[VOCAB]}", eos="sticky", pad_value=1, inkey="{inkey}")'
      f'tokenize(max_len={SEQLEN}, model="{TOKENIZERS[VOCAB]}",sample_if_multi=False, lower=False, eos="sticky", pad_value=1, inkey="{inkey}")'
  )
  pp_laion400m = (
      f'decode|resize({RES})|flip_lr|randaug(2,10)|value_range(-1,1)'
      f'|flatten|{tokenizer("caption")}'
  )
  config.input.pp = pp_laion400m

  config.pp_modules = ['ops_general', 'ops_image', 'ops_text',
                       'archive.randaug'] 

  config.log_training_steps = 50
  config.ckpt_steps = 1000

  # Model section
  config.model_name = 'proj.image_text.two_towers'
  config.model_load = {}
  config.model_init = CKPT
  config.model = ConfigDict()
  config.model.image_model = 'vit'
  config.model.text_model = 'proj.image_text.text_transformer'
  config.model.image = dict(variant=VARIANT, pool_type='map')
  config.model.text = dict(variant=TXTVARIANT, vocab_size=VOCAB)

  config.model.out_dim = (None, EMBDIM)  # (image_out_dim, text_out_dim)
  config.model.temperature_init = 10.0
  config.model.bias_init = -10.0

  if VARIANT[0] == 'B':
    config.optax_name = 'scale_by_adam'
  else:
    config.optax_name = 'big_vision.scale_by_adafactor'
  config.optax = dict(beta2_cap=0.95)

  config.lr = 1e-3 if arg.batch_size!=32_768 else 3e-4
  config.wd = 1e-4 if arg.batch_size!=32_768 else 3e-5
  warmup_steps = max(int(0.03 * config.total_steps), 100)
  config.schedule = [
      ('.*', dict(decay_type='cosine', warmup_steps=warmup_steps, mult=1.0)),# 1e-6)), # TO_DETERMINE: 1.0 or a very small value?
  ]

  config.grad_clip_norm = 1.0

  config.evals = {}
  config.evals.retrieval_coco = common.get_coco(
      pp_img=f'resize({arg.res})|value_range(-1, 1)',
      pp_txt=tokenizer('texts'),
      log_steps=1000,
  )

  return config
