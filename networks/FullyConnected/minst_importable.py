from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import fullyconnected as net
from input_data_minst import read_data_sets

def init(input_dim = 28*28, num_class = 10, max_steps = 2000, hidden1 = 128, hidden2 = 32, batch_size = 100, train_dir = 'data'):
    # Basic model parameters as external flags.
    global FLAGS
    
    class Flag(object):
        pass
    
    FLAGS = Flag()
    
    FLAGS.learning_rate = 0.01
    FLAGS.input_dim = input_dim
    FLAGS.num_class = num_class
    FLAGS.max_steps = max_steps
    FLAGS.hidden1 = hidden1
    FLAGS.hidden2 = hidden2
    FLAGS.batch_size = batch_size
    FLAGS.train_dir = train_dir
    FLAGS.fake_data = False

def placeholder_inputs(batch_size, input_dim):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
    batch_size: The batch size will be baked into both placeholders.

    Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    input_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         input_dim))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return input_placeholder, labels_placeholder


def fill_feed_dict(data_set, inputs_pl, labels_pl):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }

    Args:
    data_set: The set of inputs and labels, from input_data.read_data_sets()
    inputs_pl: The input placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.
    inputs_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
    feed_dict = {
      inputs_pl: inputs_feed,
      labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            inputs_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.

    Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    inputs_placeholder: The input placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of input and labels to evaluate, from
      input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   inputs_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
        
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
    """Train for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test.
    data_sets = read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        inputs_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size, FLAGS.input_dim)
        
        #initialize input output dimension
        net.initConfig(FLAGS.input_dim, FLAGS.num_class)

        # Build a Graph that computes predictions from the inference model.
        logits = net.inference(inputs_placeholder,
                                 FLAGS.hidden1,
                                 FLAGS.hidden2)

        # Add to the Graph the Ops for loss calculation.
        loss = net.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = net.training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = net.evaluation(logits, labels_placeholder)

        # Build the summary operation based on the TF collection of Summaries.
        #summary_op = tf.merge_all_summaries()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        # And then after everything is built, start the training loop.
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(data_sets.train,
                                     inputs_placeholder,
                                     labels_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                                   feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                #summary_str = sess.run(summary_op, feed_dict=feed_dict)
                #summary_writer.add_summary(summary_str, step)
                #summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.train_dir, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        inputs_placeholder,
                        labels_placeholder,
                        data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        inputs_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        inputs_placeholder,
                        labels_placeholder,
                        data_sets.test)
                
#def main(_):
#    run_training()


if __name__ == '__main__':
    tf.app.run()