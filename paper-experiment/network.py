import tensorflow as tf


def network_batchnorm(input_tensor, params, is_training=True):
    #initializer = tf.initializers.random_normal(0.0, 0.01)
    initializer = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)
    with tf.variable_scope('network'):
        with tf.variable_scope('cnn'):
            conv1 = tf.layers.conv2d(input_tensor, params['filter_num'][0], 3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='conv1')
            conv1 = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
            conv1 = tf.nn.relu(conv1)
            conv1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

            conv2 = tf.layers.conv2d(conv1, params['filter_num'][1], 3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='conv2')
            conv2 = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
            conv2 = tf.nn.relu(conv2)
            conv2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

            conv3 = tf.layers.conv2d(conv2, params['filter_num'][2], 3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='conv3')
            conv3 = tf.layers.batch_normalization(conv3, training=is_training, name='conv3_bn')
            conv3 = tf.nn.relu(conv3)
            conv3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2)

            conv4 = tf.layers.conv2d(conv3, params['filter_num'][3], 3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='conv4')
            conv4 = tf.layers.batch_normalization(conv4, training=is_training, name='conv4_bn')
            conv4 = tf.nn.relu(conv4)
            conv4 = tf.layers.max_pooling2d(conv4, pool_size=2, strides=2)

            conv5 = tf.layers.conv2d(conv4, params['filter_num'][4], 3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='conv5')
            conv5 = tf.layers.batch_normalization(conv5, training=is_training, name='conv5_bn')
            conv5 = tf.nn.relu(conv5)
            conv5 = tf.layers.max_pooling2d(conv5, pool_size=2, strides=2)

        with tf.variable_scope('fully'):
            flattened = tf.layers.flatten(conv5, name='flatten')
            fc1 = tf.layers.dense(inputs=flattened,
                                  units=params['hidden_units'][0],
                                  kernel_regularizer=regularizer,
                                  kernel_initializer=initializer, name='fc1')
            fc1 = tf.layers.batch_normalization(fc1, training=is_training, name='fc1_bn')
            fc1 = tf.nn.relu(fc1)

            fc2 = tf.layers.dense(inputs=fc1,
                                  units=params['hidden_units'][1],
                                  kernel_regularizer=regularizer,
                                  kernel_initializer=initializer, name='fc2')
            fc2 = tf.layers.batch_normalization(fc2, training=is_training, name='fc2_bn')
            bottlenck = tf.nn.leaky_relu(fc2, name='bottleneck')
            logits = tf.layers.dense(inputs=bottlenck,
                                     units=params['n_classes'],
                                     kernel_regularizer=regularizer,
                                     kernel_initializer=initializer,name='logit')
    return logits, bottlenck


def network_standard(input_tensor, params, is_training=True):
    #initializer = tf.initializers.random_normal(0.0, 0.01)
    initializer = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)
    with tf.variable_scope('network'):
        with tf.variable_scope('cnn'):
            conv1 = tf.layers.conv2d(input_tensor, params['filter_num'][0], 3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='conv1',
                                     activation=tf.nn.relu)
            conv1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

            conv2 = tf.layers.conv2d(conv1, params['filter_num'][1], 3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='conv2',
                                     activation=tf.nn.relu)
            conv2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

            conv3 = tf.layers.conv2d(conv2, params['filter_num'][2], 3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='conv3',
                                     activation=tf.nn.relu)
            conv3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2)

            conv4 = tf.layers.conv2d(conv3, params['filter_num'][3], 3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='conv4',
                                     activation=tf.nn.relu)
            conv4 = tf.layers.max_pooling2d(conv4, pool_size=2, strides=2)

            conv5 = tf.layers.conv2d(conv4, params['filter_num'][4], 3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='conv5',
                                     activation=tf.nn.relu)
            conv5 = tf.layers.max_pooling2d(conv5, pool_size=2, strides=2)

        with tf.variable_scope('fully'):
            flattened = tf.layers.flatten(conv5)
            fc1 = tf.layers.dense(inputs=flattened,
                                  units=params['hidden_units'][0],
                                  kernel_initializer=initializer, name='fc1',
                                  kernel_regularizer=regularizer,
                                  activation=tf.nn.relu)
            bottleneck = tf.layers.dense(inputs=fc1,
                                  units=params['hidden_units'][1],
                                  kernel_initializer=initializer, name='bottleneck',
                                  kernel_regularizer=regularizer,
                                  activation=None)
            relu_bottleneck = tf.nn.relu(bottleneck)
            logits = tf.layers.dense(inputs=relu_bottleneck,
                                  units=params['n_classes'],
                                  kernel_regularizer=regularizer,
                                  kernel_initializer=initializer, name='logit')
    return logits, bottleneck


def encoder_label_real(input_tensor, label_fake, params, is_training=True):
    initializer = tf.initializers.random_normal(0.0, 0.01)
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)
    with tf.variable_scope('encoder_label_real'):
        with tf.variable_scope('cnn'):
            conv1 = tf.layers.conv2d(input_tensor, params['filter_num'][0], 3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='conv1')
            conv1 = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
            conv1 = tf.nn.relu(conv1)
            conv1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

            conv2 = tf.layers.conv2d(conv1, params['filter_num'][1], 3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='conv2')
            conv2 = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
            conv2 = tf.nn.relu(conv2)
            conv2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

            conv3 = tf.layers.conv2d(conv2, params['filter_num'][2], 3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='conv3')
            conv3 = tf.layers.batch_normalization(conv3, training=is_training, name='conv3_bn')
            conv3 = tf.nn.relu(conv3)
            conv3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2)

            conv4 = tf.layers.conv2d(conv3, params['filter_num'][3], 3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='conv4')
            conv4 = tf.layers.batch_normalization(conv4, training=is_training, name='conv4_bn')
            conv4 = tf.nn.relu(conv4)
            conv4 = tf.layers.max_pooling2d(conv4, pool_size=2, strides=2)

            conv5 = tf.layers.conv2d(conv4, params['filter_num'][4], 3,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='conv5')
            conv5 = tf.layers.batch_normalization(conv5, training=is_training, name='conv5_bn')
            conv5 = tf.nn.relu(conv5)
            conv5 = tf.layers.max_pooling2d(conv5, pool_size=2, strides=2)

        with tf.variable_scope('dense'):
            flattened = tf.layers.flatten(conv5, name='flatten')
            concatenated = tf.concat([flattened, label_fake], axis=1)
            fc1 = tf.layers.dense(inputs=concatenated,
                                  units=params['hidden_units'][0],
                                  kernel_regularizer=regularizer,
                                  kernel_initializer=initializer, name='fc1')
            fc1 = tf.layers.batch_normalization(fc1, training=is_training, name='fc1_bn')
            fc1 = tf.nn.relu(fc1)

            fc2 = tf.layers.dense(inputs=fc1,
                                  units=params['hidden_units'][1],
                                  kernel_regularizer=regularizer,
                                  kernel_initializer=initializer, name='fc2')
            fc2 = tf.layers.batch_normalization(fc2, training=is_training, name='fc2_bn')

            bottlenck = tf.nn.relu(fc2, name='bottleneck')
            label_real = tf.layers.dense(inputs=bottlenck,
                                     units=params['n_classes'],
                                     kernel_regularizer=regularizer,
                                     activation=None, #tf.nn.sigmoid,
                                     kernel_initializer=initializer,name='label_real')
    return label_real, bottlenck


def encoder_z_type(center_diff, params, is_training=True):
    initializer = tf.initializers.random_normal(0.0, 0.01)
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)
    with tf.variable_scope('encoder_z_type'):
        fc1 = tf.layers.dense(inputs=center_diff,
                              units=100,
                              activation=None,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              name='fc1')
        fc1_bn = tf.layers.batch_normalization(fc1, training=is_training, name='fc1_bn')
        fc1_bn = tf.nn.relu(fc1_bn)
        fc2 = tf.layers.dense(inputs=fc1_bn,
                              units=100,
                              activation=None,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              name='fc2')
        fc2_bn = tf.layers.batch_normalization(fc2, training=is_training, name='fc2_bn')
        fc2_bn = tf.nn.relu(fc2_bn)
        z_type_mean = tf.layers.dense(inputs=fc2_bn,
                                      units=params['z_type_dim'],
                                      activation=None,
                                      kernel_initializer=initializer,
                                      kernel_regularizer=regularizer,
                                      name='z_type_mean')
        z_type_std = tf.layers.dense(inputs=fc2_bn,
                                      units=params['z_type_dim'],
                                      activation=tf.nn.softplus,
                                      kernel_initializer=initializer,
                                      kernel_regularizer=regularizer,
                                      name='z_type_std') + 1e-6
    return z_type_mean, z_type_std


def decoder_v(z_type, label_real, params, is_training):
    initializer = tf.initializers.random_normal(0.0, 0.01)
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)

    with tf.variable_scope('decoder_v'):
        with tf.variable_scope('concatenate'):
            c = tf.concat([z_type, label_real], axis=1)
            fc = tf.layers.dense(inputs=c, units=1000,
                                 activation=None,
                                 kernel_initializer=initializer,
                                 name='fc')
            fc = tf.layers.batch_normalization(fc, training=is_training, name='fc_bn')
            reshaped = tf.reshape(fc, [-1, 1, 1, 1000])

        with tf.variable_scope('transposed_conv2d'):
            conv1_t = tf.layers.conv2d_transpose(reshaped,
                                                 params['filter_num'][4], 7,
                                                 strides=7,
                                                 kernel_initializer=initializer,
                                                 kernel_regularizer=regularizer,
                                                 activation=None,
                                                 name='conv1_t')
            conv1_tbn = tf.layers.batch_normalization(conv1_t, training=is_training,
                                                     name='conv1_tbn')
            conv1_tbn = tf.nn.relu(conv1_tbn)

            conv2_t = tf.layers.conv2d_transpose(conv1_tbn,
                                                 params['filter_num'][3], 2,
                                                 strides=2,
                                                 kernel_initializer=initializer,
                                                 kernel_regularizer=regularizer,
                                                 activation=None,
                                                 name='conv2_t')
            conv2_tbn = tf.layers.batch_normalization(conv2_t, training=is_training,
                                                     name='conv2_tbn')
            conv2_tbn = tf.nn.relu(conv2_tbn)

            conv3_t = tf.layers.conv2d_transpose(conv2_tbn,
                                                 params['filter_num'][2], 2,
                                                 strides=2,
                                                 kernel_initializer=initializer,
                                                 kernel_regularizer=regularizer,
                                                 activation=None,
                                                 name='conv3_t')
            conv3_tbn = tf.layers.batch_normalization(conv3_t, training=is_training,
                                                     name='conv3_tbn')
            conv3_tbn = tf.nn.relu(conv3_tbn)

            conv4_t = tf.layers.conv2d_transpose(conv3_tbn,
                                                 params['filter_num'][1], 2,
                                                 strides=2,
                                                 kernel_initializer=initializer,
                                                 kernel_regularizer=regularizer,
                                                 activation=None,
                                                 name='conv4_t')
            conv4_tbn = tf.layers.batch_normalization(conv4_t, training=is_training,
                                                     name='conv4_tbn')
            conv4_tbn = tf.nn.relu(conv4_tbn)

            conv5_t = tf.layers.conv2d_transpose(conv4_tbn,
                                                 params['filter_num'][0], 2,
                                                 strides=2,
                                                 kernel_initializer=initializer,
                                                 kernel_regularizer=regularizer,
                                                 activation=None,
                                                 name='conv5_t')
            conv5_tbn = tf.layers.batch_normalization(conv5_t, training=is_training,
                                                     name='conv5_tbn')
            conv6_t = tf.layers.conv2d_transpose(conv5_tbn,
                                                 1, 2,
                                                 strides=2,
                                                 kernel_initializer=initializer,
                                                 kernel_regularizer=regularizer,
                                                 activation=None,
                                                 name='conv6_t')
            conv6_tbn = tf.layers.batch_normalization(conv6_t, training=is_training,
                                                     name='conv6_tbn')
            decoded_v = tf.nn.tanh(conv6_tbn, name='decoded_v')

    return decoded_v


def decoder_label_fake(z_type, label_real, params, is_training):
    initializer = tf.initializers.random_normal(0.0, 0.01)
    regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)

    with tf.variable_scope('decoder_label_fake'):
        with tf.variable_scope('concatenate'):
            c = tf.concat([z_type, label_real], axis=1)

        with tf.variable_scope('dense'):
            fc1 = tf.layers.dense(inputs=c, units=500,
                                    activation=None,
                                    kernel_initializer=initializer,
                                    name='fc1')
            fc1_bn = tf.layers.batch_normalization(fc1, training=is_training,
                                                name='fc1_bn')
            fc1_bn = tf.nn.relu(fc1_bn)
            fc2 = tf.layers.dense(inputs=fc1_bn, units=128,
                                    activation=None,
                                    kernel_initializer=initializer,
                                    name='fc2')
            fc2_bn = tf.layers.batch_normalization(fc2, training=is_training,
                                                name='fc2_bn')
            fc2_bn = tf.nn.relu(fc2_bn)

            decoded_label_fake_logit = tf.layers.dense(inputs=fc2_bn, units=params['n_classes'],
                                    activation=None,
                                    kernel_initializer=initializer,
                                    name='decoded_label_fake_logit')

    return decoded_label_fake_logit




