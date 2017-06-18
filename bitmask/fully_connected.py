#!/usr/bin/env python3
import data_loader
import tensorflow as tf

IMAGE_SIZE = 20
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
BATCH_SIZE = 100
def inference(images, hidden1_units, hidden2_units):
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([3*IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(3*IMAGE_PIXELS))),
                            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
                                name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Bitmask output
    with tf.name_scope('Bitmask_output'):
        weights = tf.Variable( tf.truncated_normal([hidden2_units, IMAGE_PIXELS],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
        biases = tf.Variable(tf.zeros([IMAGE_PIXELS]), name='biases')
        out_mask = tf.sigmoid(tf.matmul(hidden2, weights) + biases)

def loss(outs, labels):
    return tf.nn.l2_loss(outs - labels, name='loss')

def training(loss, learning_rate):
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(outs, labels):
    return tf.reduce_mean(tf.nn.l2_loss(outs - labels))

def placeholder_inputs(batch_size):
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    input_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 3*IMAGE_PIXELS))
    output_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, input_placeholder, output_placeholder):
    [input_images, output_images, input_mask, output_mask]  = data_loader.get_batch(batch_size = BATCH_SIZE, image_size = IMAGE_SIZE)
    input_images.reshape((BATCH_SIZE, -1))
    output_images.reshape((BATCH_SIZE, -1))
    input_mask.reshape((BATCH_SIZE, -1))
    output_mask.reshape((BATCH_SIZE, -1))

    input_vec = np.concatenate((input_images, output_images, input_mask), axis = 1)

    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    feed_dict = {
      input_placeholder: input_vec,
      output_placeholder: output_mask,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    pass





def main():
    [a, b, c, d]  = data_loader.get_batch(batch_size = 1, image_size = 5)

if __name__ == "__main__":
    main()
