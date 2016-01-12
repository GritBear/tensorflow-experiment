from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import io_util

class DataSet(object):
  # training_target is n by 1 vector (not an array)
  def __init__(self, training_set, training_target, submission_set):
    self._training_set = training_set
    self._training_target = training_target
    self._submission_set = submission_set

    self._num_examples = training_set.shape[0]
    self._input_dim = training_set.shape[1]
    self._num_tests = submission_set.shape[0]
    self._test_total_dim = submission_set.shape[1]

    print("Training Dim")
    print(self._input_dim)

    print("Testing Dim")
    print(self._test_total_dim)

    assert self._test_total_dim == self._input_dim

    print("Training Samples:")
    print(self._num_examples)
    
    print("Testing Samples")
    print(self._num_tests)

    # Define training validation split
    validationPercentage = 0.15
    self._numValidation = int(validationPercentage * self._num_examples)
    self._numTrain = self._num_examples - self._numValidation

    self._train_input = training_set[0:self._numTrain, :]
    self._train_target = training_target[0:self._numTrain]
    self._validation_input = training_set[self._numTrain:-1, :]
    self._validation_target = training_target[self._numTrain:-1]

    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def input(self):
    return self._train_input

  @property
  def output(self):
    return self._train_target

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  @property
  def testInput(self):
    return self._submission_set

  @property
  def validationSet(self):
    return self._validation_input, self._validation_target

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)

      self._train_input = self._train_input[perm]
      self._train_target = self._train_target[perm]

      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    end = self._index_in_epoch
    return self._train_input[start:end], self._train_target[start:end]


def read_input():
  trainFile = 'train.csv'
  testFile = 'test.csv'

  print("Reading Training Sets")
  TrainData, header = io_util.extract_txt_arr(trainFile)
  Train_set = TrainData[:,1:-1]
  Train_target = TrainData[:,-1]

  print("Reading Submission Sets")
  TestData, header = io_util.extract_txt_arr(testFile)
  Test_set = TestData[:,1:]

  # process training data
  print("Compose DataSets")
  dataSet = DataSet(Train_set, Train_target, Test_set)

  
  return dataSet