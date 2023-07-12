import tensorflow as tf
tf.config.experimental_run_functions_eagerly(False)
import sonnet as snt
from tqdm import tqdm
# from IPython.display import clear_output
import numpy as np
import pandas as pd
import time
import os
import enformer
import attention_module


assert snt.__version__.startswith('2.0')

# @title `get_targets(organism)`
def get_targets(organism):
  targets_txt = f'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_{organism}.txt'
  return pd.read_csv(targets_txt, sep='\t')

# @title `get_dataset(organism, subset, num_threads=8)`
import glob
import json
import functools


def organism_path(organism):
  return os.path.join('gs://basenji_barnyard/data', organism)


def get_dataset(organism, subset, num_threads=8):
  metadata = get_metadata(organism)
  dataset = tf.data.TFRecordDataset('./data_human_tfrecords_test-0-7.tfr',
                                    compression_type='ZLIB',
                                    num_parallel_reads=num_threads)
  dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                        num_parallel_calls=num_threads)
  return dataset


def get_metadata(organism):
  # Keys:
  # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
  # pool_width, crop_bp, target_length
  # path = os.path.join(organism_path(organism), 'statistics.json')
  path = './statistics.json'
  with tf.io.gfile.GFile(path, 'r') as f:
    return json.load(f)


def tfrecord_files(organism, subset):
  # Sort the values by int(*).
  return sorted(tf.io.gfile.glob(os.path.join(
      organism_path(organism), 'tfrecords', f'{subset}-*.tfr'
  )), key=lambda x: int(x.split('-')[-1].split('.')[0]))


def deserialize(serialized_example, metadata):
  """Deserialize bytes stored in TFRecordFile."""
  feature_map = {
      'sequence': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.string),
  }
  example = tf.io.parse_example(serialized_example, feature_map)
  sequence = tf.io.decode_raw(example['sequence'], tf.bool)
  sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
  sequence = tf.cast(sequence, tf.float32)

  target = tf.io.decode_raw(example['target'], tf.float16)
  target = tf.reshape(target,
                      (metadata['target_length'], metadata['num_targets']))
  target = tf.cast(target, tf.float32)

  return {'sequence': sequence,
          'target': target}

human_dataset = get_dataset('human', 'train').batch(1).repeat()
mouse_dataset = get_dataset('mouse', 'train').batch(1).repeat()
human_mouse_dataset = tf.data.Dataset.zip((human_dataset, mouse_dataset)).prefetch(2)

it = iter(mouse_dataset)
example = next(it)

# Example input
it = iter(human_mouse_dataset)
example = next(it)
for i in range(len(example)):
  print(['human', 'mouse'][i])
  print({k: (v.shape, v.dtype) for k,v in example[i].items()})
print(example)

def create_step_function(model, optimizer):

  @tf.function
  def train_step(batch, head, optimizer_clip_norm_global=0.2):
    with tf.GradientTape() as tape:
      outputs = model(batch['sequence'], is_training=True)[head]
      loss = tf.reduce_mean(
          tf.keras.losses.poisson(batch['target'], outputs))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply(gradients, model.trainable_variables)

    return loss
  return train_step

learning_rate = tf.Variable(0., trainable=False, name='learning_rate')
optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
num_warmup_steps = 5000
target_learning_rate = 0.0005

model = enformer.Enformer(channels=1536 // 4,  # Use 4x fewer channels to train faster.
                          num_heads=8,
                          num_transformer_layers=11,
                          pooling_type='max')

train_step = create_step_function(model, optimizer)
steps_per_epoch = 20
num_epochs = 5

data_it = iter(human_mouse_dataset)
global_step = 0

for epoch_i in range(num_epochs):
  for i in tqdm(range(steps_per_epoch)):
    global_step += 1

    if global_step > 1:
      learning_rate_frac = tf.math.minimum(
          1.0, global_step / tf.math.maximum(1.0, num_warmup_steps))
      learning_rate.assign(target_learning_rate * learning_rate_frac)

    batch_human, batch_mouse = next(data_it)

    loss_human = train_step(batch=batch_human, head='human')
    loss_mouse = train_step(batch=batch_mouse, head='mouse')

  # End of epoch.
  print('')
  print('loss_human', loss_human.numpy(),
        'loss_mouse', loss_mouse.numpy(),
        'learning_rate', optimizer.learning_rate.numpy()
        )