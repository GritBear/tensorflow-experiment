# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the regression network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.

TensorFlow install instructions:
https://tensorflow.org/get_started/os_setup.html

MNIST tutorial:
https://tensorflow.org/tutorials/mnist/tf/index.html
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.python.platform
import tensorflow as tf

def inference(inputs, hidden1_units, hidden2_units, num_outputs):
  """Build the regression model up to where it may be used for inference.

  Args:
    inputs: inputs placeholder, from inputs().
    hidden1: Size of the first hidden layer.
    hidden2: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed acts.
  """

  """
  Get the last dimension of input placeholder because for FNN, the input is (batch_size, inputDim)
  """
  global inputDim 
  global NUM_OUTPUTS 

  inputDim = inputs.get_shape().as_list()[-1] # get the last dimension. 
  NUM_OUTPUTS = num_outputs

  # Hidden 1
  with tf.name_scope('hidden1') as scope:
    weights = tf.Variable(
        tf.truncated_normal([inputDim, hidden1_units],
                            stddev=1.0 / math.sqrt(float(inputDim))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(inputs, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2') as scope:
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('output_linear') as scope:
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_OUTPUTS],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_OUTPUTS]),
                         name='biases')
    acts = tf.matmul(hidden2, weights) + biases
  return acts


def loss(acts, targets):
  """Calculates the loss from the acts and the targets.

  Args:
    acts: output tensor, float - [batch_size, NUM_OUTPUTS].
    targets: targets tensor, float - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  loss = tf.nn.l2_loss(tf.sub(acts, targets), name='L2_loss')

  return loss


def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.AdagradOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(acts, targets):
  """Evaluate the quality of the acts at predicting the label.

  Args:
    acts: acts tensor, float - [batch_size, NUM_OUTPUTS].
    targets: targets tensor, int32 - [batch_size], with values in the
      range [0, NUM_OUTPUTS).

  Returns:
    A scalar float tensor with L2 loss.
  """
  loss = tf.nn.l2_loss(tf.sub(acts, targets))
  return loss
