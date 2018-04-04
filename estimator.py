import os
from network import network_batchnorm, network_standard
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


class EmbeddingSessionRunHook(tf.train.SessionRunHook):

    def __init__(self, embedding_update_op):
        self.embedding_update_op = embedding_update_op


    def before_run(self, run_context):
        print('Before calling session.run().')
        return tf.train.SessionRunArgs(self.embedding_update_op)

    def after_run(self, run_context, run_values):
        print('Done running one step. The values of my embedding[:,0]: ',
            run_values.results[::60,0])


class EmbeddingCheckpointSaverHook(tf.train.CheckpointSaverHook):

    def before_run(self, run_context):
        pass

    def after_run(self, run_context, run_values):
        pass

    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        self._save(last_step, session)

    def _save(self, step, session):
        """Saves the latest checkpoint."""
        self._get_saver().save(session, self._save_path, global_step=step)
        self._summary_writer.add_session_log(
            tf.SessionLog(
                status=tf.SessionLog.CHECKPOINT, checkpoint_path=self._save_path),
            step)



class Classifier(object):

    def __init__(self, config):
        self.params = config.params
        self.model_dir = config.model_dir

    def get_model_fn(self):
        loss_fn = self.loss_fn
        score_fn = self.score_fn
        pred_fn = self.pred_fn
        false_negatives_fn = self.false_negatives_fn
        auc_fn = self.auc_fn

        def model_fn(features, labels, mode, params):

            if mode == tf.estimator.ModeKeys.TRAIN:
                is_training = True
            else:
                is_training = True # Why?

            input_tensor = features['x']
            idx = features['idx']

            if params['batch_norm']:
                logits, bottleneck = network_batchnorm(input_tensor, params, is_training)
            else:
                logits, bottleneck = network_standard(input_tensor, params, is_training)

            loss = loss_fn(logits, labels)
            score = score_fn(logits)
            preds = pred_fn(logits)
            sparse_labels = pred_fn(labels)
            false_negatives, fn_update_op = false_negatives_fn(labels, score)
            auc, auc_update_op = auc_fn(labels, score)
            embedding_train = tf.Variable(tf.zeros([params['train_data_num'], params['hidden_units'][1]]),
                                          name='embedding_train')
            embedding_train_update_op = tf.scatter_update(embedding_train,
                                                          indices=idx,
                                                          updates=bottleneck)
            embedding_valid = tf.Variable(tf.zeros([params['valid_data_num'], params['hidden_units'][1]]),
                                          name='embedding_valid')
            embedding_valid_update_op = tf.scatter_update(embedding_valid,
                                                          indices=idx,
                                                          updates=bottleneck)


            if mode == tf.estimator.ModeKeys.EVAL:

                saver = tf.train.Saver(var_list=[embedding_valid])
                eval_model_dir = os.path.join(params['model_dir'], 'eval')
                metrics = {'false_negatives': (false_negatives, fn_update_op),
                       'auc': (auc, auc_update_op)}

                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics,
                    evaluation_hooks=[EmbeddingSessionRunHook(embedding_valid_update_op),
                                       EmbeddingCheckpointSaverHook(eval_model_dir,
                                                                    save_steps=100,
                                                                    saver=saver)])


            elif mode == tf.estimator.ModeKeys.TRAIN:

                optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
                gradients = optimizer.compute_gradients(loss)
                clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
                #train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                train_op = optimizer.apply_gradients(clipped_gradients, global_step=tf.train.get_global_step())

                logging_hook = tf.train.LoggingTensorHook({'loss':loss,
                                                           'false_negatives':false_negatives,
                                                           'auc':auc,
                                                           'predictions':preds[0:5],
                                                           'labels':sparse_labels[0:5]
                                                            },
                                                          every_n_iter=100)

                train_ops = tf.group(train_op, fn_update_op, auc_update_op, embedding_train_update_op)

                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_ops,
                                                  predictions=preds,
                                                  training_hooks=[logging_hook])
            elif mode == tf.estimator.ModeKeys.PREDICT:
                raise NotImplementedError


        return model_fn


    def loss_fn(self, logits, labels):
        raise NotImplementedError()


    def score_fn(self, logits):
        raise NotImplementedError()


    def pred_fn(self, logits):
        raise NotImplementedError()


    def false_negatives_fn(self, labels, preds):
        false_negatives, update_op = tf.metrics.false_negatives(labels=labels,
                                        predictions=preds,
                                        name='false_negatives')
        return tf.convert_to_tensor(false_negatives), update_op


    def auc_fn(self, labels, preds):
        auc, update_op = tf.metrics.auc(labels=labels,
                                        predictions=preds,
                                        name='auc')
        return tf.convert_to_tensor(auc), update_op


    def get_estimator(self):

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True

        estimator = tf.estimator.Estimator(
            model_fn=self.get_model_fn(),
            params=self.params,
            model_dir=self.model_dir,
            config=tf.contrib.learn.RunConfig(session_config=sess_config))
        return estimator



class MultilabelClassifier(Classifier):


    def loss_fn(self, logits, labels):
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
        return loss


    def score_fn(self, logits):
        score = tf.sigmoid(logits)
        return score


    def pred_fn(self, logits):
        values, indices = tf.nn.top_k(logits, k=7, sorted=True)
        return indices


class SinglelabelClassifier(Classifier):


    def loss_fn(self, logits, labels):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        return loss


    def score_fn(self, logits):
        score = tf.nn.softmax(logits)
        return score


    def pred_fn(self, logits):
        index = tf.argmax(logits)
        return index




class CenterlossClassifier(Classifier):

    def __init__(self, config):
        self.params = config.params
        self.model_dir = config.model_dir
        self.batch_size = config.batch_size

    def get_model_fn(self):
        loss_fn = self.loss_fn
        score_fn = self.score_fn
        pred_fn = self.pred_fn
        false_negatives_fn = self.false_negatives_fn
        auc_fn = self.auc_fn

        def model_fn(features, labels, mode, params):

            if mode == tf.estimator.ModeKeys.TRAIN:
                is_training = True
            else:
                is_training = True # Why?

            input_tensor = features['x']
            idx = features['idx']

            if params['batch_norm']:
                logits, bottleneck = network_batchnorm(input_tensor, params, is_training)
            else:
                logits, bottleneck = network_standard(input_tensor, params, is_training)

            centers = tf.get_variable('centers',
                                      [self.params['n_classes'], params['hidden_units'][1]],
                                      dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(), trainable=False)

            loss, c_update_op = loss_fn(logits, labels, centers, bottleneck)
            score = score_fn(logits)
            preds = pred_fn(logits)
            sparse_labels = pred_fn(labels)
            false_negatives, fn_update_op = false_negatives_fn(labels, score)
            auc, auc_update_op = auc_fn(labels, score)
            embedding_train = tf.Variable(tf.zeros([params['train_data_num'], params['hidden_units'][1]]),
                                          name='embedding_train')
            embedding_train_update_op = tf.scatter_update(embedding_train,
                                                          indices=idx,
                                                          updates=bottleneck)
            embedding_valid = tf.Variable(tf.zeros([params['valid_data_num'], params['hidden_units'][1]]),
                                          name='embedding_valid')
            embedding_valid_update_op = tf.scatter_update(embedding_valid,
                                                          indices=idx,
                                                          updates=bottleneck)


            if mode == tf.estimator.ModeKeys.EVAL:

                saver = tf.train.Saver(var_list=[embedding_valid])
                eval_model_dir = os.path.join(params['model_dir'], 'eval')
                metrics = {'false_negatives': (false_negatives, fn_update_op),
                       'auc': (auc, auc_update_op)}

                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics,
                    evaluation_hooks=[EmbeddingSessionRunHook(embedding_valid_update_op),
                                       EmbeddingCheckpointSaverHook(eval_model_dir,
                                                                    save_steps=100,
                                                                    saver=saver)])


            elif mode == tf.estimator.ModeKeys.TRAIN:

                optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                logging_hook = tf.train.LoggingTensorHook({'loss':loss,
                                                           'false_negatives':false_negatives,
                                                           'auc':auc,
                                                           'predictions':preds[0:5],
                                                           'labels':sparse_labels[0:5]
                                                            },
                                                          every_n_iter=100)

                train_ops = tf.group(train_op, fn_update_op, auc_update_op,
                                     embedding_train_update_op, c_update_op)

                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_ops,
                                                  predictions=preds,
                                                  training_hooks=[logging_hook])
            elif mode == tf.estimator.ModeKeys.PREDICT:
                raise NotImplementedError


        return model_fn


    def loss_fn(self, logits, labels, centers, bottleneck):
        with tf.variable_scope('loss_definition'):
            with tf.variable_scope('cross_entropy_loss'):
                xentropy_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
            with tf.variable_scope('center_loss'):
                center_loss, c_update_op = self.get_center_loss(labels, centers, bottleneck)
            with tf.variable_scope('regularizer'):
                l2_loss = tf.losses.get_regularization_loss()
            with tf.variable_scope('total_loss'):
                loss = xentropy_loss + center_loss + 0.001 * l2_loss
        #    loss = tf.Print(loss, [loss])
        return loss, c_update_op


    def get_center_loss(self, labels, centers, bottleneck):
        raise NotImplementedError()


class MultilabelCenterlossClassifier(CenterlossClassifier):


    def score_fn(self, logits):
        score = tf.sigmoid(logits)
        return score


    def pred_fn(self, logits):
        values, indices = tf.nn.top_k(logits, k=7, sorted=True)
        return indices


    def get_center_loss(self, labels, centers, bottleneck, alpha=0.9):
        """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
        (http://ydwen.github.io/papers/WenECCV16.pdf)
        """
        labels_float = tf.cast(labels, tf.float32)
        one_num = tf.reduce_sum(labels_float, axis=1)
        batch_size = tf.shape(bottleneck)[0]
        one_num = tf.reshape(one_num, (batch_size, 1))
        centers_batch = self.get_centers_batch(labels_float, centers, one_num)
        diff = self.get_diff(alpha, centers_batch, bottleneck, one_num)
        diff_matrix = tf.matmul(labels_float, diff, transpose_a=True) # shape : label_num * bottleneck_size
        update_op = tf.assign(centers, (centers - diff_matrix) / tf.norm(centers - diff_matrix))
        loss = tf.reduce_mean(tf.square(diff))
        #l2_loss = tf.reduce_mean(tf.nn.l2_loss(centers))
        #loss += l2_loss
        loss = tf.Print(loss, [loss, centers_batch, diff, diff_matrix])

        return loss, update_op


    def get_centers_batch(self, labels_float, centers, one_num):
        raise NotImplementedError()


    def get_diff(self, alpha, centers_batch, bottleneck, one_num):
        raise NotImplementedError()



class MultilabelMeanCenterlossClassifier(MultilabelCenterlossClassifier):


    def get_centers_batch(self, labels_float, centers, one_num):
        # centers : (num_labels, bottleneck_size) label : (batch, num_labels)
        centers_batch = tf.divide(tf.matmul(labels_float, centers), one_num)
        return centers_batch

    def get_diff(self, alpha, centers_batch, bottleneck, one_num):
        diff = (1 - alpha) * (centers_batch - bottleneck) / one_num # shape : batch_size * bottleneck_size
        return diff



class MultilabelAddCenterlossClassifier(MultilabelCenterlossClassifier):


    def get_centers_batch(self, labels_float, centers, one_num):
        centers_batch = tf.matmul(labels_float, centers)
        return centers_batch

    def get_diff(self, alpha, centers_batch, bottleneck, one_num):
        diff = (1 - alpha) * (centers_batch - bottleneck)
        return diff


