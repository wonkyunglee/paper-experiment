import os
import gzip
import urllib
import shutil
import cv2
import numpy as np
import tensorflow as tf
from db_manager import DBManager
from tensorflow.examples.tutorials.mnist import input_data


def preprocess(filepath, length=224):
    content = tf.read_file(filepath)
    raw = tf.decode_raw(content, tf.uint8)
    zero_padding_size = tf.constant(length) - tf.mod(tf.shape(raw), tf.constant(length))
    zero_padding = tf.zeros(zero_padding_size, dtype=tf.uint8)
    raw_padded = tf.concat(axis=0, values=[raw, zero_padding])
    image = tf.reshape(raw_padded, [-1, length, 1])
    thumbnail = image_preprocess(image, length)
    return thumbnail


def image_preprocess(image, length=224):
    thumbnail = tf.image.resize_images(image, [length, length],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    subval = 127.5
    divval = 127.5
    normalize = lambda x: tf.div(tf.subtract(tf.cast(x, tf.float32), subval), divval)
    thumbnail = normalize(thumbnail)
    return thumbnail

def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding
       ref : https://github.com/anujshah1003/Tensorboard-own-image-data-image-features-embedding-visualization
    Args:
        data: NxHxW[x3] tensor containing the images.

    Returns:
        data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    #data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data

class ThumbnailLoader(object):

    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.shuffle_buffer = config.shuffle_buffer
        self.db_manager = DBManager(config)
        self.make_dicts_for_labels()


    def make_dicts_for_labels(self):
        raise NotImplementedError()


    def get_train_feature_dict(self):
        sql = self.config.db_query_train
        metadata_path = os.path.join(self.config.model_dir, 'metadata_train.tsv')
        return self.get_feature_dict(sql, metadata_path)


    def get_valid_feature_dict(self):
        sql = self.config.db_query_valid
        metadata_path = os.path.join(self.config.model_dir, 'metadata_valid.tsv')
        return self.get_feature_dict(sql, metadata_path)


    def get_feature_dict(self, sql, metadata_path):
        cursor = self.db_manager.select_query(sql)
        labels_arr = []
        rep_label_arr = []
        filepath_arr = []
        index_arr = []
        label_num = self.label_num

        with open(metadata_path, 'w') as f:
            f.write('Index\tRepLabel\tLabels\tFilepath\n')

            for index, row in enumerate(cursor):
                path = row[self.config.db_path_idx]
                labels = row[self.config.db_labels_idx]
                rep_label = row[self.config.db_rep_label_idx]
                f.write('%s\t%s\t%s\t%s\n'%(index, rep_label, labels, path))

                labels, rep_label = self.get_label_for_purpose(labels, rep_label)

                labels_arr.append(labels)
                rep_label_arr.append(rep_label)
                filepath_arr.append(path)
                index_arr.append(index)

        filepath_arr = np.array(filepath_arr)
        labels_arr = np.array(labels_arr)
        rep_label_arr = np.array(rep_label_arr)
        index_arr = np.array(index_arr)
        print('data num : %d'%len(index_arr))

        return {'index_arr':index_arr, 'filepath_arr':filepath_arr,
                'labels_arr':labels_arr, 'rep_label_arr':rep_label_arr}


    def get_label_for_purpose(self, labels, rep_label):
        raise NotImplementedError()


    def sparse_to_dense(self, sparse, label_num):
        dense = np.zeros(label_num)
        for index in sparse:
            dense[index] += 1
        return dense


    def get_dataset(self, dictionary):
        index_arr = dictionary['index_arr']
        filepath_arr = dictionary['filepath_arr']
        labels_arr = dictionary['labels_arr']
        rep_label_arr = dictionary['rep_label_arr']

        thumbnails = tf.data.Dataset.from_tensor_slices(filepath_arr).map(
            preprocess, num_parallel_calls=10)
        indices = tf.data.Dataset.from_tensor_slices(index_arr)
        rep_label = tf.data.Dataset.from_tensor_slices(rep_label_arr)
        labels = tf.data.Dataset.from_tensor_slices(labels_arr)
        features, labels = self.select_features_and_labels(thumbnails, indices, rep_label, labels)
        dataset = tf.data.Dataset.zip((features, labels))
        return dataset


    def select_features_and_labels(self):
        raise NotImplementedError()


    def get_train_iterator(self, dataset):
        dataset = dataset.prefetch(buffer_size=self.batch_size)
        dataset = dataset.repeat(self.num_epochs)
        dataset = dataset.shuffle(buffer_size=self.shuffle_buffer)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator


    def get_valid_iterator(self, dataset):
        dataset = dataset.prefetch(buffer_size=self.batch_size)
        dataset = dataset.repeat(1)
        dataset = dataset.shuffle(buffer_size=self.shuffle_buffer)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator


    def train_input_fn(self):
        return self.get_input_fn(self.get_train_feature_dict(), self.get_train_iterator)


    def valid_input_fn(self):
        return self.get_input_fn(self.get_valid_feature_dict(), self.get_valid_iterator)


    def get_input_fn(self, dictionary, get_iterator):
        dataset = self.get_dataset(dictionary)
        iterator = get_iterator(dataset)
        features, labels = iterator.get_next()
        return features, labels


    def predict_input_fn(self):
        raise NotImplementedError


    def get_dense_label(self, labels, label_num):
        labels_splitted = labels.split(',')
        labels_sparse = [self.label_dict[label] for label in labels_splitted]
        labels_dense = self.sparse_to_dense(labels_sparse, label_num)
        return labels_dense


    def get_label_for_purpose(self, labels, rep_label):
        raise NotImplementedError()



class ThumbnailMultilabelLoader(ThumbnailLoader):


    def make_dicts_for_labels(self):
        label_metadata_path = os.path.join(self.config.model_dir, 'metadata_label.tsv')
        sql = self.config.db_query_label
        cursor = self.db_manager.select_query(sql)
        label_arr = []
        rep_label_arr = []
        for row in cursor:
            rep_label = row[0]
            labels = row[1]
            [label_arr.append(label) for label in labels.split(',')]
            rep_label_arr.append(rep_label)
        labels, counts = np.unique(label_arr, return_counts=True)
        sorted_labels = list(map(lambda x:x[0], sorted(zip(labels, counts), key=lambda x:x[1], reverse=True)))

        rep_label_set = set(rep_label_arr)
        ref = 0
        for i, label in enumerate(sorted_labels):
            if label in rep_label_set:
                tmp = sorted_labels[ref]
                sorted_labels[ref] = label
                sorted_labels[i] = tmp
                ref += 1

        indexes = range(len(sorted_labels))

        with open(label_metadata_path, 'w') as f:
            f.write('Index\tLabel\n')
            for i, label in enumerate(sorted_labels):
                f.write('%s\t%s\n'%(i, label))

        self.label_dict = {label:index for label, index in zip(sorted_labels, indexes)}
        self.index_dict = {index:label for label, index in zip(sorted_labels, indexes)}
        self.label_num = len(self.label_dict)
        print('label_num : %d'%self.label_num)



    def select_features_and_labels(self, thumbnails, indices, rep_label, labels):
        features = {'x':thumbnails, 'idx':indices, 'rep_label':rep_label}
        labels = labels
        return features, labels


    def get_label_for_purpose(self, labels, rep_label):
        labels_dense = self.get_dense_label(labels, self.config.params['n_classes'])
        rep_label_dense = self.get_dense_label(rep_label, self.config.params['n_rep_classes'])
        return labels_dense, rep_label_dense



class ThumbnailSinglelabelLoader(ThumbnailLoader):


    def make_dicts_for_labels(self):
        label_metadata_path = os.path.join(self.config.model_dir, 'metadata_label.tsv')
        sql = self.config.db_query_label
        cursor = self.db_manager.select_query(sql)
        label_arr = []
        rep_label_arr = []
        for row in cursor:
            rep_label = row[0]
            labels = row[1]
            [label_arr.append(label) for label in labels.split(',')]
            rep_label_arr.append(rep_label)

        labels, counts = np.unique(rep_label_arr, return_counts=True)
        sorted_labels = list(map(lambda x:x[0], sorted(zip(labels, counts), key=lambda x:x[1], reverse=True)))

        indexes = range(len(sorted_labels))

        with open(label_metadata_path, 'w') as f:
            f.write('Index\tLabel\n')
            for i, label in enumerate(sorted_labels):
                f.write('%s\t%s\n'%(i, label))

        self.label_dict = {label:index for label, index in zip(sorted_labels, indexes)}
        self.index_dict = {index:label for label, index in zip(sorted_labels, indexes)}
        self.label_num = len(self.label_dict)
        print('label_num : %d'%self.label_num)


    def select_features_and_labels(self, thumbnails, indices, rep_label, labels):
        features = {'x':thumbnails, 'idx':indices, 'labels':labels}
        labels = rep_label
        return features, labels


    def get_label_for_purpose(self, labels, rep_label):
        rep_label_dense = self.get_dense_label(rep_label, self.config.params['n_rep_classes'])
        return labels, rep_label_dense



class MnistLoader(object):
    """download, decode_image, decod_label methods are copied from
    https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py
    """
    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.shuffle_buffer = config.shuffle_buffer
        self.data_dir = config.data_dir
        self.make_dicts_for_labels()


    def get_dataset(self, purpose):
        data = input_data.read_data_sets(self.data_dir, one_hot=True)
        if purpose == 'train':
            data = data.train
        elif purpose == 'valid':
            data = data.test

        np.random.seed(2018)
        perm = np.arange(len(data.labels))
        np.random.shuffle(perm)

        images_np = data.images[perm].reshape(-1, 28, 28, 1)
        rep_label_np = data.labels[perm]
        indexes_np = np.arange(len(rep_label_np))
        labels_np = self.get_labels_from_rep_label(rep_label_np)
        self.save_tsv_file(indexes_np, rep_label_np, labels_np, purpose)
        self.save_sprite_image(images_np, purpose)
        self.save_images(images_np, purpose)

        images = tf.data.Dataset.from_tensor_slices(images_np).map(
            image_preprocess, num_parallel_calls=2)
        rep_label = tf.data.Dataset.from_tensor_slices(rep_label_np)
        indexes = tf.data.Dataset.from_tensor_slices(indexes_np)
        labels = tf.data.Dataset.from_tensor_slices(labels_np)

        features, labels = self.select_features_and_labels(images, labels, rep_label, indexes)
        dataset = tf.data.Dataset.zip((features, labels))
        return dataset


    def save_images(self, images_np, purpose):
        images_path = os.path.join(self.config.model_dir, 'images_' + purpose + '.png')
        np.save(images_path, images_np)


    def save_sprite_image(self, images_np, purpose):
        metadata_sprite_path = os.path.join(self.config.model_dir, 'metadata_sprite_' + purpose + '.png')
        sprite = images_to_sprite(images_np)
        cv2.imwrite(metadata_sprite_path, sprite)


    def save_tsv_file(self, indexes_np, rep_label_np, labels_np, purpose):
        metadata_path = os.path.join(self.config.model_dir, 'metadata_' + purpose + '.tsv')

        with open(metadata_path, 'w') as f:
            f.write('Index\tRepLabel\tLabels\n')
            count = 0
            for index, rep_label, labels in zip(indexes_np, rep_label_np, labels_np):
                if count >= self.config.params[purpose + '_data_num']:
                    break

                rep_label = np.where(rep_label == 1)[0][0]
                rep_label = self.index_dict[rep_label]
                multilabels = ''
                for i, label in enumerate(labels):
                    if label == 1:
                        multilabels += self.index_dict[i] + ','
                multilabels = multilabels[:-1]
                index = str(index)

                f.write('%s\t%s\t%s\n'%(index, rep_label, multilabels))
                count += 1


    def get_labels_from_rep_label(self, rep_label):
        multilabels = np.zeros([len(rep_label), 13])
        # 10 : multiple of 2, 11: multiple of 3, 12: multiple of 5
        for i, label in enumerate(rep_label):
            label = np.where(label == 1)[0][0]
            multilabels[i][label] = 1
            if label % 2 == 0:
                multilabels[i][10] = 1
            if label % 3 == 0:
                multilabels[i][11] = 1
            if label % 5 == 0:
                multilabels[i][12] = 1
        return multilabels


    def make_dicts_for_labels(self):
        label_metadata_path = os.path.join(self.config.model_dir, 'metadata_label.tsv')
        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  'multiple_of_2', 'multiple_of_3', 'multiple_of_5']
        indexes = range(len(labels))

        with open(label_metadata_path, 'w') as f:
            f.write('Index\tLabel\n')
            for i, label in enumerate(labels):
                f.write('%s\t%s\n'%(i, label))

        self.label_dict = {label:index for label, index in zip(labels, indexes)}
        self.index_dict = {index:label for label, index in zip(labels, indexes)}
        self.label_num = len(self.label_dict)
        print('label_num : %d'%self.label_num)


    def get_train_iterator(self, dataset):
        dataset = dataset.prefetch(buffer_size=self.batch_size)
        dataset = dataset.repeat(self.num_epochs)
        dataset = dataset.shuffle(buffer_size=self.shuffle_buffer)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator


    def get_test_iterator(self, dataset):
        dataset = dataset.prefetch(buffer_size=self.batch_size)
        dataset = dataset.repeat(1)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator


    def train_input_fn(self):
        dataset = self.get_dataset('train')
        iterator = self.get_train_iterator(dataset)
        images, labels = iterator.get_next()
        return images, labels


    def valid_input_fn(self):
        dataset = self.get_dataset('valid')
        iterator = self.get_test_iterator(dataset)
        images, labels = iterator.get_next()
        return images, labels

    def predict_input_fn(self):
        raise NotImplementedError


class MnistSinglelabelLoader(MnistLoader):


    def select_features_and_labels(self, images, labels, rep_label, indexes):
        features = {'x':images, 'idx':indexes, 'labels':labels}
        labels = rep_label
        return features, labels



class MnistMultilabelLoader(MnistLoader):


    def select_features_and_labels(self, images, labels, rep_label, indexes):
        features = {'x':images, 'idx':indexes, 'rep_label':rep_label}
        return features, labels



class MnistNoisyMultilabelLoader(MnistMultilabelLoader):


    def select_features_and_labels(self, images, labels, rep_label, indexes):
        features = {'x':images, 'idx':indexes, 'rep_label':rep_label}
        return features, labels


    def get_dataset(self, purpose):
        if purpose == 'train':
            images = np.load(os.path.join(self.config.data_dir, 'noisy/train_images.npy'))
            labels = np.load(os.path.join(self.config.data_dir, 'noisy/train_labels.npy'))
            real_labels = np.load(os.path.join(self.config.data_dir, 'noisy/train_real_labels.npy'))

        elif purpose == 'valid':
            images = np.load(os.path.join(self.config.data_dir, 'noisy/test_images.npy'))
            labels = np.load(os.path.join(self.config.data_dir, 'noisy/test_labels.npy'))
            real_labels = np.load(os.path.join(self.config.data_dir, 'noisy/test_real_labels.npy'))

        np.random.seed(2018)
        perm = np.arange(len(labels))
        np.random.shuffle(perm)

        images_np = images[perm].reshape(-1, 28, 28, 1)
        rep_label_np = labels[perm]
        real_label_np = real_labels[perm]
        indexes_np = np.arange(len(rep_label_np))
        labels_np = self.get_labels_from_rep_label(rep_label_np)
        self.save_tsv_file(indexes_np, rep_label_np, labels_np, real_label_np, purpose)
        self.save_sprite_image(images_np, purpose)
        self.save_images(images_np, purpose)

        images = tf.data.Dataset.from_tensor_slices(images_np).map(
            image_preprocess, num_parallel_calls=2)
        rep_label = tf.data.Dataset.from_tensor_slices(rep_label_np)
        indexes = tf.data.Dataset.from_tensor_slices(indexes_np)
        labels = tf.data.Dataset.from_tensor_slices(labels_np)

        features, labels = self.select_features_and_labels(images, labels, rep_label, indexes)
        dataset = tf.data.Dataset.zip((features, labels))
        return dataset


    def save_tsv_file(self, indexes_np, rep_label_np, labels_np, real_labels_np, purpose):
        metadata_path = os.path.join(self.config.model_dir, 'metadata_' + purpose + '.tsv')

        with open(metadata_path, 'w') as f:
            f.write('Index\tRepLabel\tLabels\tRealLabel\n')
            count = 0
            for index, rep_label, labels, real_label in zip(indexes_np, rep_label_np,
                                                            labels_np, real_labels_np):
                if count >= self.config.params[purpose + '_data_num']:
                    break

                rep_label = np.where(rep_label == 1)[0][0]
                real_label = np.where(real_label == 1)[0][0]
                rep_label = self.index_dict[rep_label]
                real_label = self.index_dict[real_label]
                multilabels = ''
                for i, label in enumerate(labels):
                    if label == 1:
                        multilabels += self.index_dict[i] + ','
                multilabels = multilabels[:-1]
                index = str(index)

                f.write('%s\t%s\t%s\t%s\n'%(index, rep_label, multilabels, real_label))
                count += 1


