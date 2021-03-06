import os
from network import network_batchnorm, network_standard
from network import encoder_label_real, encoder_z_type
from network import decoder_label_fake, decoder_v

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


class UpdateSessionRunHook(tf.train.SessionRunHook):

    def __init__(self, update_ops):
        self.update_ops = update_ops


    def before_run(self, run_context):
        print('Before calling session.run().')
        return tf.train.SessionRunArgs(self.update_ops)


    def after_run(self, run_context, run_values):
        print('Done running one step. The values of my embedding[:,0]: ',
            run_values.results[::10,0])


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
        accuracy_fn = self.accuracy_fn
        get_rep_label = self.get_rep_label

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
            rep_label = get_rep_label(features, labels)
            accuracy, acc_update_op = accuracy_fn(rep_label, score)
            tf.summary.scalar('auc', auc)
            tf.summary.scalar('accuracy', accuracy)

            embedding_train = tf.Variable(tf.zeros([params['train_data_num'], params['hidden_units'][1]]),
                                          name='embedding_train', trainable=False)
            embedding_train_update_op = tf.scatter_update(embedding_train,
                                                          indices=idx,
                                                          updates=bottleneck)
            sigmoid_train = tf.Variable(tf.zeros([params['train_data_num'], params['n_classes']]),
                                          name='sigmoid_train', trainable=False)
            sigmoid_train_update_op = tf.scatter_update(sigmoid_train,
                                                          indices=idx,
                                                          updates=score)
            embedding_valid = tf.Variable(tf.zeros([params['valid_data_num'], params['hidden_units'][1]]),
                                          name='embedding_valid', trainable=False)
            embedding_valid_update_op = tf.scatter_update(embedding_valid,
                                                          indices=idx,
                                                          updates=bottleneck)
            sigmoid_valid = tf.Variable(tf.zeros([params['valid_data_num'], params['n_classes']]),
                                          name='sigmoid_valid', trainable=False)
            sigmoid_valid_update_op = tf.scatter_update(sigmoid_valid,
                                                          indices=idx,
                                                          updates=score)



            if mode == tf.estimator.ModeKeys.EVAL:

                saver = tf.train.Saver(var_list=[embedding_valid, sigmoid_valid])
                eval_model_dir = os.path.join(params['model_dir'], 'eval')
                metrics = {'false_negatives': (false_negatives, fn_update_op),
                           'auc': (auc, auc_update_op), 'accuracy': (accuracy, acc_update_op)}

                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics,
                    evaluation_hooks=[EmbeddingSessionRunHook(embedding_valid_update_op),
                                      UpdateSessionRunHook(sigmoid_valid_update_op),
                                       EmbeddingCheckpointSaverHook(eval_model_dir,
                                                                    save_steps=100,
                                                                    saver=saver)])


            elif mode == tf.estimator.ModeKeys.TRAIN:

                optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

                logging_hook = tf.train.LoggingTensorHook({'loss':loss,
                                                           'false_negatives':false_negatives,
                                                           'auc':auc,
                                                           'accuracy':accuracy,
                                                           'predictions':preds[0:5],
                                                           'labels':sparse_labels[0:5]
                                                            },
                                                          every_n_iter=100)
                summary_hook = tf.train.SummarySaverHook(
                    save_secs=60,
                    output_dir=params['model_dir'],
                    summary_op=tf.summary.merge_all()
                )

                train_ops = tf.group(train_op, fn_update_op, auc_update_op, acc_update_op,
                                     auc, accuracy, embedding_train_update_op, sigmoid_train_update_op)

                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_ops,
                                                  predictions=preds,
                                                  training_hooks=[logging_hook, summary_hook])
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


    def accuracy_fn(self, labels, scores):
        preds = tf.round(scores)[:, :self.params['n_rep_classes']]

        accuracy, update_op = tf.metrics.accuracy(labels=labels,
                                        predictions=preds,
                                        name='accuracy')
        return tf.convert_to_tensor(accuracy), update_op


    def get_rep_label(self, features, labels):
        if 'rep_label' in features:
            return features['rep_label']
        else:
            return labels


    def get_estimator(self):

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True

        estimator = tf.estimator.Estimator(
            model_fn=self.get_model_fn(),
            params=self.params,
            model_dir=self.model_dir,
            config=tf.contrib.learn.RunConfig(session_config=sess_config, save_checkpoints_secs=60))
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
        l2_loss = tf.losses.get_regularization_loss()
        loss += 0.001 * l2_loss
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
        accuracy_fn = self.accuracy_fn
        get_rep_label = self.get_rep_label

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
                                      initializer=tf.random_normal_initializer(),
                                      trainable=False)

            loss, c_update_op = loss_fn(logits, labels, centers, bottleneck)
            score = score_fn(logits)
            preds = pred_fn(logits)
            sparse_labels = pred_fn(labels)
            rep_label = get_rep_label(features, labels)
            false_negatives, fn_update_op = false_negatives_fn(labels, score)
            auc, auc_update_op = auc_fn(labels, score)
            accuracy, acc_update_op = accuracy_fn(rep_label, score)
            tf.summary.scalar('auc', auc)
            tf.summary.scalar('accuracy', accuracy)
            idx = tf.Print(idx, [idx])

            embedding_train = tf.Variable(tf.zeros([params['train_data_num'], params['hidden_units'][1]]),
                                          name='embedding_train', trainable=False)
            embedding_train_update_op = tf.scatter_update(embedding_train,
                                                          indices=idx,
                                                          updates=bottleneck)
            sigmoid_train = tf.Variable(tf.zeros([params['train_data_num'], params['n_classes']]),
                                          name='sigmoid_train', trainable=False)
            sigmoid_train_update_op = tf.scatter_update(sigmoid_train,
                                                          indices=idx,
                                                          updates=score)
            embedding_valid = tf.Variable(tf.zeros([params['valid_data_num'], params['hidden_units'][1]]),
                                          name='embedding_valid', trainable=False)
            embedding_valid_update_op = tf.scatter_update(embedding_valid,
                                                          indices=idx,
                                                          updates=bottleneck)
            sigmoid_valid = tf.Variable(tf.zeros([params['valid_data_num'], params['n_classes']]),
                                          name='sigmoid_valid', trainable=False)
            sigmoid_valid_update_op = tf.scatter_update(sigmoid_valid,
                                                          indices=idx,
                                                          updates=score)


            if mode == tf.estimator.ModeKeys.EVAL:

                saver = tf.train.Saver(var_list=[embedding_valid, sigmoid_valid])
                eval_model_dir = os.path.join(params['model_dir'], 'eval')
                metrics = {'false_negatives': (false_negatives, fn_update_op),
                           'auc': (auc, auc_update_op), 'accuracy': (accuracy, acc_update_op)}

                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics,
                    evaluation_hooks=[EmbeddingSessionRunHook(embedding_valid_update_op),
                                      UpdateSessionRunHook(sigmoid_valid_update_op),
                                      EmbeddingCheckpointSaverHook(eval_model_dir,
                                                                   save_steps=100,
                                                                   saver=saver)])


            elif mode == tf.estimator.ModeKeys.TRAIN:
                bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
                gradients = optimizer.compute_gradients(loss)
                with tf.control_dependencies(bn_update_ops+[c_update_op]):
                    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                train_ops = tf.group(train_op, fn_update_op, auc_update_op, acc_update_op,
                                     auc, accuracy, embedding_train_update_op,
                                     sigmoid_train_update_op)


                logging_hook = tf.train.LoggingTensorHook({'loss':loss,
                                                           'false_negatives':false_negatives,
                                                           'auc':auc,
                                                           'accuracy':accuracy,
                                                           'predictions':preds[0:5],
                                                           'labels':sparse_labels[0:5]
                                                            },
                                                          every_n_iter=100)
                summary_hook = tf.train.SummarySaverHook(
                    save_secs=60,
                    output_dir=params['model_dir'],
                    summary_op=tf.summary.merge_all()
                )

                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_ops,
                                                  predictions=preds,
                                                  training_hooks=[logging_hook, summary_hook])
            elif mode == tf.estimator.ModeKeys.PREDICT:
                raise NotImplementedError


        return model_fn


    def loss_fn(self, logits, labels, centers, bottleneck):
        with tf.variable_scope('loss_definition'):
            with tf.variable_scope('cross_entropy_loss'):
                xentropy_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
            with tf.variable_scope('center_loss'):
                center_loss, c_update_op = self.get_center_loss(labels, centers, bottleneck, alpha=0.1)
            with tf.variable_scope('regularizer'):
                l2_loss = tf.losses.get_regularization_loss()
            with tf.variable_scope('total_loss'):
                loss = xentropy_loss  + self.params['centerloss_coef'] * center_loss  + self.params['reg_coef'] * l2_loss
        #    loss = tf.Print(loss, [loss])
        tf.summary.scalar("center_loss", center_loss)
        tf.summary.scalar("cross_entropy_loss", xentropy_loss)
        tf.summary.scalar("regularize_loss", l2_loss)

        return loss, c_update_op


    def get_center_loss(self, labels, centers, bottleneck):
        raise NotImplementedError()


class SinglelabelCenterlossClassifier(CenterlossClassifier):


    def score_fn(self, logits):
        score = tf.nn.softmax(logits)
        return score


    def pred_fn(self, logits):
        values, indices = tf.nn.top_k(logits, k=7, sorted=True)
        return indices


    def get_center_loss2(self, labels, centers, bottleneck, alpha=0.9):
        """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
        (http://ydwen.github.io/papers/WenECCV16.pdf)
        """
        labels_float = tf.cast(labels, tf.float32)
        batch_size = tf.shape(bottleneck)[0]
        centers_batch = self.get_centers_batch(labels_float, centers)
        diff = self.get_diff(alpha, centers_batch, bottleneck)
        diff_matrix = tf.matmul(labels_float, diff, transpose_a=True) # shape : label_num * bottleneck_size
        update_op = tf.assign(centers, (centers - diff_matrix)) #/ tf.norm(centers - diff_matrix))
        loss = tf.reduce_mean(tf.square(diff))
        #l2_loss = tf.reduce_mean(tf.nn.l2_loss(centers))
        #loss += l2_loss
        loss = tf.Print(loss, [loss, centers_batch, diff, diff_matrix])

        return loss, update_op


    def get_centers_batch(self, labels_float, centers):
        centers_batch = tf.matmul(labels_float, centers)
        return centers_batch

    def get_diff(self, alpha, centers_batch, bottleneck):
        diff = (1 - alpha) * (centers_batch - bottleneck)
        return diff


    def get_center_loss3(self, labels, centers, bottleneck, alpha=0.5):
        """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
        (http://ydwen.github.io/papers/WenECCV16.pdf)
        """
        labels = tf.where(tf.equal(labels, 1.0))[:,1] # onehot to dense
        labels = tf.cast(labels, tf.int64)
        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)
        loss = tf.nn.l2_loss(bottleneck - centers_batch)
        diff = centers_batch - bottleneck
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff

        update_op = tf.scatter_sub(centers, labels, diff)

        return loss, update_op


    def get_center_loss(self, labels, centers, bottleneck, alpha=0.5):
        """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
        (http://ydwen.github.io/papers/WenECCV16.pdf)
        """
        labels = tf.cast(labels, tf.float32)
        center_batch = tf.matmul(labels, centers)
        unique_count = tf.reduce_sum(labels, axis=0)
        unique_count = tf.reshape(unique_count, [-1, 1])

        appear_times = tf.matmul(labels, unique_count)
        diff = center_batch - bottleneck
        loss = tf.nn.l2_loss(diff)
        diff = alpha * diff / tf.cast((1 + appear_times), tf.float32)
        diff_matrix = tf.matmul(labels, diff, transpose_a=True)
        update_op = tf.assign_sub(centers, diff_matrix)

        return loss, update_op


    def loss_fn(self, logits, labels, centers, bottleneck):
        with tf.variable_scope('loss_definition'):
            with tf.variable_scope('cross_entropy_loss'):
                xentropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            with tf.variable_scope('center_loss'):
                center_loss, c_update_op = self.get_center_loss(labels, centers, bottleneck)
            with tf.variable_scope('regularizer'):
                l2_loss = tf.losses.get_regularization_loss()
            with tf.variable_scope('total_loss'):
                loss = xentropy_loss  + 0.1* center_loss + 0.001 * l2_loss
        #    loss = tf.Print(loss, [loss])
        tf.summary.scalar("center_loss", center_loss)
        tf.summary.scalar("cross_entropy_loss", xentropy_loss)
        tf.summary.scalar("regularize_loss", l2_loss)

        return loss, c_update_op



class MultilabelCenterlossClassifier(CenterlossClassifier):


    def score_fn(self, logits):
        score = tf.sigmoid(logits)
        return score


    def pred_fn(self, logits):
        values, indices = tf.nn.top_k(logits, k=7, sorted=True)
        return indices


    def get_center_loss2(self, labels, centers, bottleneck, alpha=0.9):
        """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
        (http://ydwen.github.io/papers/WenECCV16.pdf)
        """
        norm_tensor = tf.constant(self.params['tag_weights'], dtype=tf.float32, name='tag_weights')
        norm_tensor = tf.reshape(norm_tensor, [self.params['n_classes'],1])
        labels_float = tf.cast(labels, tf.float32)
        one_num = tf.reduce_sum(labels_float, axis=1)
        batch_size = tf.shape(bottleneck)[0]
        one_num = tf.reshape(one_num, (batch_size, 1))
        centers_batch = self.get_centers_batch(labels_float, centers, one_num)
        diff = self.get_diff(alpha, centers_batch, bottleneck, one_num)
        diff_matrix = tf.matmul(labels_float, diff, transpose_a=True) # shape : label_num * bottleneck_size
        new_centers = centers - diff_matrix
        norms = tf.reshape(tf.norm(new_centers, axis=1), [-1, 1])
        update_op = tf.assign(centers, new_centers / norms * norm_tensor)
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=1))
        # l2_loss = tf.reduce_mean(tf.nn.l2_loss(centers))
        # loss += 0.01 * l2_loss
        # loss += 0.01 * tf.reduce_mean(tf.square(tf.norm(centers, axis=1) - norm_tensor))
        loss = tf.Print(loss, [loss, tf.norm(centers[0]), tf.norm(centers[1]), diff, diff_matrix])

        return loss, update_op


    def get_center_loss(self, labels, centers, bottleneck, alpha=0.5):
        """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
        (http://ydwen.github.io/papers/WenECCV16.pdf)
        """
        labels = tf.cast(labels, tf.float32)
        one_num = tf.reduce_sum(labels, axis=1)
        one_num = tf.stop_gradient(tf.reshape(one_num, [-1,1]))
        center_batch = self.get_centers_batch(labels, centers, one_num)
        unique_count = tf.reduce_sum(labels, axis=0)
        unique_count = tf.reshape(unique_count, [-1, 1])

        appear_times = tf.matmul(labels, unique_count)
        diff = center_batch - bottleneck
        loss = tf.nn.l2_loss(diff)
        diff = alpha * diff / tf.cast((1 + appear_times), tf.float32)
        diff_matrix = tf.matmul(labels, diff, transpose_a=True)
        update_op = tf.assign_sub(centers, diff_matrix)

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
        diff = (1 - alpha) * (centers_batch - bottleneck) / one_num
        return diff


class MultilabelMeanWeightedCenterlossClassifier(MultilabelMeanCenterlossClassifier):


    def get_center_loss(self, labels, centers, bottleneck, alpha=0.5):
        """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
        (http://ydwen.github.io/papers/WenECCV16.pdf)
        """
        norm_tensor = tf.constant(self.params['tag_weights'], dtype=tf.float32, name='tag_weights')
        norm_tensor = tf.reshape(norm_tensor, [self.params['n_classes'],1])

        labels = tf.cast(labels, tf.float32)
        one_num = tf.reduce_sum(labels, axis=1)
        one_num = tf.stop_gradient(tf.reshape(one_num, [-1,1]))
        center_batch = self.get_centers_batch(labels, centers, one_num)
        unique_count = tf.reduce_sum(labels, axis=0)
        unique_count = tf.reshape(unique_count, [-1, 1])

        appear_times = tf.matmul(labels, unique_count)
        diff = center_batch - bottleneck
        loss = tf.nn.l2_loss(diff)
        #loss += tf.reduce_mean(tf.square(tf.norm(centers, axis=1) - norm_tensor))
        diff = alpha * diff / tf.cast((1 + appear_times), tf.float32)
        diff_matrix = tf.matmul(labels, diff, transpose_a=True)
        #update_op = tf.assign_sub(centers, diff_matrix)

        new_centers = centers - diff_matrix
        norms = tf.reshape(tf.norm(new_centers, axis=1), [-1, 1])
        norms = tf.stop_gradient(norms)
        update_op = tf.assign(centers, new_centers / norms * norm_tensor)

        return loss, update_op



class MultilabelAddWeightedCenterlossClassifier(MultilabelAddCenterlossClassifier):


    def get_center_loss(self, labels, centers, bottleneck, alpha=0.5):
        """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
        (http://ydwen.github.io/papers/WenECCV16.pdf)
        """
        norm_tensor = tf.constant(self.params['tag_weights'], dtype=tf.float32, name='tag_weights')
        norm_tensor = tf.reshape(norm_tensor, [self.params['n_classes'],1])

        labels = tf.cast(labels, tf.float32)
        one_num = tf.reduce_sum(labels, axis=1)
        one_num = tf.stop_gradient(tf.reshape(one_num, [-1,1]))
        center_batch = self.get_centers_batch(labels, centers, one_num)
        unique_count = tf.reduce_sum(labels, axis=0)
        unique_count = tf.reshape(unique_count, [-1, 1])

        appear_times = tf.matmul(labels, unique_count)
        diff = center_batch - bottleneck
        loss = tf.nn.l2_loss(diff)
        #loss += tf.reduce_mean(tf.square(tf.norm(centers, axis=1) - norm_tensor))
        diff = alpha * diff / tf.cast((1 + appear_times) / one_num, tf.float32)
        diff_matrix = tf.matmul(labels, diff, transpose_a=True)
        #update_op = tf.assign_sub(centers, diff_matrix)

        new_centers = centers - diff_matrix
        norms = tf.reshape(tf.norm(new_centers, axis=1), [-1, 1])
        norms = tf.stop_gradient(norms)
        update_op = tf.assign(centers, new_centers / norms * norm_tensor)

        return loss, update_op




class VBCenterlossClassifier(MultilabelCenterlossClassifier):

    def __init__(self, config):
        self.params = config.params
        self.model_dir = config.model_dir
        self.batch_size = config.batch_size

    def get_model_fn(self):
        loss_fn = self.loss_fn
        score_fn = self.score_fn
        pred_fn = self.pred_fn
        auc_fn = self.auc_fn
        accuracy_fn = self.accuracy_fn
        get_rep_label = self.get_rep_label

        def model_fn(features, labels, mode, params):

            if mode == tf.estimator.ModeKeys.TRAIN:
                is_training = True
            else:
                is_training = True # Why?

            centers = tf.get_variable('centers',
                                      [self.params['n_classes'], params['hidden_units'][1]],
                                      dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(),
                                      trainable=False)

            input_tensor = features['x']
            idx = features['idx']
            label_fake = tf.cast(labels, tf.float32)

            p_label_real, bottleneck = encoder_label_real(input_tensor, label_fake, params, is_training)
            #sampled_label_real = tf.cast(tf.contrib.distributions.Bernoulli(p_label_real).sample(), tf.float32)
            #sampled_label_real = 0.5 * label_fake - 0.5 * tf.nn.sigmoid(p_label_real) + 0.5
            sampled_label_real = tf.nn.sigmoid(p_label_real)

            one_num = tf.reduce_sum(sampled_label_real, axis=1)
            batch_size = tf.shape(bottleneck)[0]
            one_num = tf.reshape(one_num, (batch_size, 1))
            centers_batch = tf.matmul(sampled_label_real, centers) / one_num # mean
            center_diff = centers_batch - bottleneck
            diff_for_update = (1 - params['alpha']) * center_diff / one_num
            diff_matrix = tf.matmul(sampled_label_real, diff_for_update, transpose_a=True) # shape : label_num * bottleneck_size
            new_centers = centers - diff_matrix
            #norms = tf.reshape(tf.norm(new_centers, axis=1), [-1, 1])
            #update_op = tf.assign(centers, new_centers / norms * norm_tensor)
            center_update_op = tf.assign(centers, new_centers)
            center_loss = tf.reduce_mean(tf.reduce_sum(tf.square(center_diff), axis=1))
            center_l2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(centers), axis=1))

            z_type_mean, z_type_std = encoder_z_type(center_diff, params, is_training)
            sampled_z_type = z_type_mean + z_type_std * tf.random_normal(tf.shape(z_type_mean), 0, 1, dtype=tf.float32)

            decoded_label_fake_logit = decoder_label_fake(sampled_z_type, sampled_label_real, params, is_training)
            decoded_v = decoder_v(sampled_z_type, sampled_label_real, params, is_training)

            marginal_likelihood_lf = -tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,
                                                                  logits=decoded_label_fake_logit)
            marginal_likelihood_v = -tf.losses.mean_squared_error(labels=input_tensor,
                                                                predictions=decoded_v)
            marginal_likelihood = marginal_likelihood_lf + marginal_likelihood_v
            kl_divergence_z = tf.reduce_mean(0.5* tf.reduce_sum(tf.square(z_type_mean) + tf.square(z_type_std) - tf.log(1e-8 + tf.square(z_type_std)) -1, 1))
            kl_divergence_lr = tf.reduce_mean(tf.reduce_sum( sampled_label_real * tf.log(sampled_label_real) - sampled_label_real * tf.log(0.1) + \
                                                             (1-sampled_label_real) *tf.log(1-sampled_label_real) -(1-sampled_label_real)*tf.log(0.9) , 1))

            kl_divergence = kl_divergence_z #+ kl_divergence_lr

            elbo_loss = marginal_likelihood - kl_divergence
            l2_loss = tf.losses.get_regularization_loss()
            #auxiliary_loss = -tf.reduce_mean(tf.reduce_sum( label_fake*tf.log(sampled_label_real) + (1-label_fake)*tf.log(1-sampled_label_real), 1))
            auxiliary_loss = -tf.reduce_mean(tf.reduce_sum( sampled_label_real*tf.log(sampled_label_real) + (1-sampled_label_real)*tf.log(1-sampled_label_real), 1))
            loss = -elbo_loss + 0.01*center_loss + 0.01*center_l2_loss + 0.01*l2_loss + 0.01*auxiliary_loss

            loss = tf.Print(loss, [loss, marginal_likelihood, kl_divergence, center_loss, center_l2_loss, auxiliary_loss, l2_loss])
            sampled_label_real = tf.Print(sampled_label_real, [sampled_label_real[0, :5], label_fake[0, :5]])

            preds = pred_fn(sampled_label_real)
            sparse_labels = pred_fn(labels)
            rep_label = get_rep_label(features, labels)
            auc, auc_update_op = auc_fn(labels, sampled_label_real)
            accuracy, acc_update_op = accuracy_fn(rep_label, sampled_label_real)
            tf.summary.scalar('auc', auc)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('kl_divergence_lr', kl_divergence_lr)
            tf.summary.scalar('kl_divergence_z', kl_divergence_z)
            tf.summary.scalar('marginal_likelihood_lf', marginal_likelihood_lf)
            tf.summary.scalar('marginal_likelihood_v', marginal_likelihood_v)
            tf.summary.scalar('l2_loss', l2_loss)
            tf.summary.scalar('center_loss', center_loss)
            tf.summary.scalar('center_l2_loss', center_l2_loss)
            tf.summary.scalar('auxiliary_loss', auxiliary_loss)

            embedding_train = tf.Variable(tf.zeros([params['train_data_num'], params['hidden_units'][1]]),
                                          name='embedding_train', trainable=False)
            embedding_train_update_op = tf.scatter_update(embedding_train,
                                                          indices=idx,
                                                          updates=bottleneck)
            embedding_valid = tf.Variable(tf.zeros([params['valid_data_num'], params['hidden_units'][1]]),
                                          name='embedding_valid', trainable=False)
            embedding_valid_update_op = tf.scatter_update(embedding_valid,
                                                          indices=idx,
                                                          updates=bottleneck)
            z_type_valid = tf.Variable(tf.zeros([params['valid_data_num'], params['z_type_dim']]),
                                          name='z_type_valid', trainable=False)
            z_type_valid_update_op = tf.scatter_update(z_type_valid,
                                                       indices=idx,
                                                       updates=sampled_z_type)
            sampled_label_real_valid = tf.Variable(tf.zeros([params['valid_data_num'], params['n_classes']]),
                                          name='sampled_label_real_valid', trainable=False)
            sampled_label_real_valid_update_op = tf.scatter_update(sampled_label_real_valid,
                                                       indices=idx,
                                                       updates=sampled_label_real)




            if mode == tf.estimator.ModeKeys.EVAL:

                saver = tf.train.Saver(var_list=[embedding_valid, z_type_valid, sampled_label_real_valid])
                eval_model_dir = os.path.join(params['model_dir'], 'eval')
                metrics = { 'auc': (auc, auc_update_op), 'accuracy': (accuracy, acc_update_op)}

                # sprite image for MNIST
                #config = projector.ProjectorConfig()
                #embedding = config.embeddings.add()
                #embedding.tensor_name = embedding_valid.name
                #embedding.metadata_path = 'metadata_valid.tsv' # os.path.join(params['model_dir'], 'metadata_valid.tsv')
                #metadata_sprite_path = os.path.join(params['model_dir'], 'metadata_sprite_valid.png')
                #embedding.sprite.image_path = metadata_sprite_path
                #embedding.sprite.single_image_dim.extend([28,28])
                #projector.visualize_embeddings(tf.summary.FileWriter(params['model_dir']), config)

                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics,
                    evaluation_hooks=[EmbeddingSessionRunHook(embedding_valid_update_op),
                                      UpdateSessionRunHook(z_type_valid_update_op),
                                      UpdateSessionRunHook(sampled_label_real_valid_update_op),
                                      EmbeddingCheckpointSaverHook(eval_model_dir,
                                                                   save_steps=100,
                                                                   saver=saver)])


            elif mode == tf.estimator.ModeKeys.TRAIN:
                bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
                gradients = optimizer.compute_gradients(loss)
                with tf.control_dependencies(bn_update_ops):
                    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                train_ops = tf.group(train_op, auc_update_op, acc_update_op,
                                     auc, accuracy, embedding_train_update_op, center_update_op)


                logging_hook = tf.train.LoggingTensorHook({'loss':loss,
                                                           'auc':auc,
                                                           'accuracy':accuracy,
                                                           'kl_divergence':kl_divergence,
                                                           'marginal_likelihood':marginal_likelihood,
                                                           'predictions':preds[0:5],
                                                           'labels':sparse_labels[0:5]
                                                            },
                                                          every_n_iter=100)
                summary_hook = tf.train.SummarySaverHook(
                    save_secs=60,
                    output_dir=params['model_dir'],
                    summary_op=tf.summary.merge_all()
                )

                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_ops,
                                                  predictions=preds,
                                                  training_hooks=[logging_hook, summary_hook])
            elif mode == tf.estimator.ModeKeys.PREDICT:
                raise NotImplementedError


        return model_fn


