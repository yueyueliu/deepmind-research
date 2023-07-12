import tensorflow as tf
import sonnet as snt
from tqdm import tqdm
from IPython.display import clear_output
import numpy as np
import pandas as pd
import time
import os
import enformer
from Process import *
from evaluate import *

# df_targets_human = get_targets('human')
df_targets_human = pd.read_csv("/home/liuyue/PycharmProjects/data_enformer/target_human.csv",)
df_targets_human = df_targets_human.drop('Unnamed: 0', axis=1)
print(df_targets_human.head())

human_dataset = get_dataset('human', 'train').batch(1).repeat()#sequence(None,131072,4)Target(None,896,5313)
mouse_dataset = get_dataset('mouse', 'train').batch(1).repeat()
human_mouse_dataset = tf.data.Dataset.zip((human_dataset, mouse_dataset)).prefetch(2)#dict{2}

# Example input
it = iter(human_dataset)
example = next(it)
# for i in range(len(example)):
#   print(['human', 'mouse'][i])
#   print({k: (v.shape, v.dtype) for k,v in example[i].items()})
print(example)

## Model training
learning_rate = tf.Variable(0., trainable=False, name='learning_rate')
optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
num_warmup_steps = 5000
target_learning_rate = 0.0005

model = enformer.Enformer(channels=1536 // 4,  # Use 4x fewer channels to train faster.
                          num_heads=8,
                          num_transformer_layers=11,
                          pooling_type='max')

train_step = create_step_function(model, optimizer)

# Train the model
steps_per_epoch = 1
num_epochs = 5

data_it = iter(human_mouse_dataset)
global_step = 0
for epoch_i in range(num_epochs):
  for i in tqdm(range(steps_per_epoch)):
    global_step += 1#global_step=1

    if global_step > 1:
      learning_rate_frac = tf.math.minimum(
          1.0, global_step / tf.math.maximum(1.0, num_warmup_steps))
      learning_rate.assign(target_learning_rate * learning_rate_frac)

    batch_human, batch_mouse = next(data_it)

    loss_human = train_step(batch=batch_human, head='human')#dict:2：sequence': (TensorShape([1, 131072, 4]), tf.float32), 'target': (TensorShape([1, 896, 5313]), tf.float32)}
    # loss_mouse = train_step(batch=batch_mouse, head='mouse')

  # End of epoch.
  print('')
  print('loss_human', loss_human.numpy(),
        # 'loss_mouse', loss_mouse.numpy(),
        'learning_rate', optimizer.learning_rate.numpy()
        )




#Evaluate
metrics_human = evaluate_model(model,
                               dataset=get_dataset('human', 'valid').batch(1).prefetch(2),
                               head='human',
                               max_steps=100)
print('')
print({k: v.numpy().mean() for k, v in metrics_human.items()})

# metrics_mouse = evaluate_model(model,
#                                dataset=get_dataset('mouse', 'valid').batch(1).prefetch(2),
#                                head='mouse',
#                                max_steps=100)
# print('')
# print({k: v.numpy().mean() for k, v in metrics_mouse.items()})

