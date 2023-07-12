import pandas as pd
import glob
import json
import functools
import tensorflow as tf
import os


#Processing data
def get_targets(organism):
  targets_txt = f'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_{organism}.txt'
  return pd.read_csv(targets_txt, sep='\t')

# @title `get_dataset(organism, subset, num_threads=8)`
def organism_path(organism):
  return os.path.join('gs://basenji_barnyard/data', organism)


def get_metadata(organism):
  # Keys:
  # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
  # pool_width, crop_bp, target_length

  # path = os.path.join(organism_path(organism), 'statistics.json')
  path = "/home/liuyue/PycharmProjects/data_enformer/statistics.json"
  with tf.io.gfile.GFile(path, 'r') as f:
    return json.load(f)


def tfrecord_files(organism, subset):
  # Sort the values by int(*).
  return sorted(tf.io.gfile.glob(os.path.join(
      organism_path(organism), 'tfrecords', f'{subset}-*.tfr'
  )), key=lambda x: int(x.split('-')[-1].split('.')[0]))


def get_dataset(organism, subset, num_threads=8):
  metadata = get_metadata(organism)
  dataset = tf.data.TFRecordDataset("/home/liuyue/PycharmProjects/data_enformer/data_human_tfrecords_test-0-7.tfr",
                                    compression_type='ZLIB',
                                    num_parallel_reads=num_threads)
  # dataset = tf.data.TFRecordDataset(tfrecord_files(organism, subset),
  #                                   compression_type='ZLIB',
  #                                   num_parallel_reads=num_threads)
  dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                        num_parallel_calls=num_threads)
  return dataset





#自定义定制器
#将不同的tfr文件读入后，处理成模型feed需要的样子；有利于数据的可迁移
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



###Train the model
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

