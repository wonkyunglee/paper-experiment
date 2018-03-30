import tensorflow as tf


def network_batchnorm(input_tensor, params, is_training=True):
    initializer = tf.initializers.random_normal(0.0, 0.01)
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
            bottlenck = tf.nn.relu(fc2, name='bottleneck')
            logits = tf.layers.dense(inputs=bottlenck,
                                     units=params['n_classes'],
                                     kernel_regularizer=regularizer,
                                     kernel_initializer=initializer,name='logit')
    return logits, bottlenck


def network_standard(input_tensor, params, is_training=True):
    initializer = tf.initializers.random_normal(0.0, 0.01)
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
            bottlenck = tf.layers.dense(inputs=fc1,
                                  units=params['hidden_units'][1],
                                  kernel_initializer=initializer, name='bottleneck',
                                  kernel_regularizer=regularizer,
                                  activation=tf.nn.relu)
            logits = tf.layers.dense(inputs=bottlenck,
                                  units=params['n_classes'],
                                  kernel_regularizer=regularizer,
                                  kernel_initializer=initializer, name='logit')
    return logits, bottlenck


