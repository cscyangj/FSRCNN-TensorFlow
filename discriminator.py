import tensorflow as tf
import tensorflow.contrib.layers as layers

def discriminator(dis_inputs, channels=64, is_training=True):

    # Define the convolution building block
    def conv2(batch_input, output_channel=64, kernel=5, stride=1, use_bias=False, scope='conv'):
        with tf.variable_scope(scope):
                return layers.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME',
                                activation_fn=None, weights_initializer=layers.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer() if use_bias else None)

    # Define Lrelu
    def lrelu(inputs, alpha=0.2):
        return tf.nn.relu(inputs) - alpha * tf.nn.relu(-inputs)

    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, output_channel, kernel_size, stride, use_bias=False, scope='conv1')
            net = layers.batch_norm(net, decay=0.9, epsilon=0.001, activation_fn=lrelu,
                        scale=False, fused=True, is_training=is_training)

        return net

    with tf.device('/gpu:0'):
        with tf.variable_scope('discriminator_unit'):
            with tf.variable_scope('input_stage'):
                net = lrelu(conv2(dis_inputs, channels, 5, 2))

            net = discriminator_block(net, channels*2, 5, 2, 'disblock_1')

            net = discriminator_block(net, channels*4, 5, 2, 'disblock_2')

            net = layers.flatten(net)

            #with tf.variable_scope('dense_layer_1'):
            #    net = tf.layers.dense(net, channels*16, activation=lrelu, kernel_initializer=layers.xavier_initializer())

            with tf.variable_scope('dense_layer_1'):
                net = tf.layers.dense(net, 1, activation=tf.nn.sigmoid, kernel_initializer=layers.xavier_initializer())

    return net

