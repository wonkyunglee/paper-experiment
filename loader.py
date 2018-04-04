import os
import numpy as np
import tensorflow as tf
from db_manager import DBManager




def preprocess(filepath, length=224):
    content = tf.read_file(filepath)
    raw = tf.decode_raw(content, tf.uint8)
    zero_padding_size = tf.constant(length) - tf.mod(tf.shape(raw), tf.constant(length))
    zero_padding = tf.zeros(zero_padding_size, dtype=tf.uint8)
    raw_padded = tf.concat(axis=0, values=[raw, zero_padding])
    image = tf.reshape(raw_padded, [-1, length, 1])
    thumbnail = tf.image.resize_images(image, [length, length],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    subval = 127.5
    divval = 127.5
    normalize = lambda x: tf.div(tf.subtract(tf.cast(x, tf.float32), subval), divval)
    thumbnail = normalize(thumbnail)
    return thumbnail


class ThumbnailLoader(object):

    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.shuffle_buffer = config.shuffle_buffer
        self.db_manager = DBManager(config)


    def make_dicts_for_labels(self):
        label_metadata_path = os.path.join(self.config.model_dir, 'metadata_label.tsv')
        sql = self.config.db_query_label
        cursor = self.db_manager.select_query(sql)
        label_arr = []
        for row in cursor:
            [label_arr.append(label) for label in row[0].split(',')]
        labels, counts = np.unique(label_arr, return_counts=True)
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
        dataset = self.select_features(thumbnails, indices, rep_label, labels)
        return dataset


    def select_features(self):
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



class ThumbnailMultilabelLoader(ThumbnailLoader):


    def get_dense_label(self, labels):
        labels_splitted = labels.split(',')
        labels_sparse = [self.label_dict[label] for label in labels_splitted]
        labels_dense = self.sparse_to_dense(labels_sparse, self.label_num)
        return labels_dense


    def get_label_for_purpose(self, labels, rep_label):
        labels_dense = self.get_dense_label(labels)
        return labels_dense, rep_label


    def select_features(self, thumbnails, indices, rep_label, labels):
        dataset = tf.data.Dataset.zip(({'x':thumbnails, 'idx':indices, 'rep_label':rep_label}, labels))
        return dataset



class ThumbnailClassificationLoader(ThumbnailLoader):

    def get_dense_label(self, label):
        rep_label_sparse = [self.label_dict[label]]
        rep_label_dense = self.sparse_to_dense(rep_label_sparse, self.label_num)
        return rep_label_dense


    def get_label_for_purpose(self, labels, rep_label):
        rep_label_dense = self.get_dense_label(rep_label)
        return labels, rep_label_dense


    def select_features(self, thumbnails, indices, rep_label, labels):
        dataset = tf.data.Dataset.zip(({'x':thumbnails, 'idx':indices, 'labels':labels}, rep_label))
        return dataset




class MNISTLoader(object):
    """download, decode_image, decod_label methods are copied from
    https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py
    """
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.shuffle_buffer = config.shuffle_buffer
        self.data_dir = config.data_dir


    def get_dataset(self, images_filename, labels_filename):
        images_filepath = self.download(self.data_dir, images_filename)
        labels_filepath = self.download(self.data_dir, labels_filename)

        def decode_image(image):
            # Normalize from [0, 255] to [0.0, 1.0]
            image = tf.decode_raw(image, tf.uint8)
            image = tf.cast(image, tf.float32)
            image = tf.reshape(image, [784])
            return image / 255.0

        def decode_label(label):
            label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
            label = tf.reshape(label, [])  # label is a scalar
            return tf.to_int32(label)

        images = tf.data.FixedLengthRecordDataset(
            images_filepath, 28 * 28, header_bytes=16).map(decode_image)
        labels = tf.data.FixedLengthRecordDataset(
            labels_filepath, 1, header_bytes=8).map(decode_label)
        dataset = tf.data.Dataset.zip(({'x':images}, labels))
        return dataset


    def download(self, directory, filename):
        """Download (and unzip) a file from the MNIST dataset if not already done."""
        filepath = os.path.join(directory, filename)
        if tf.gfile.Exists(filepath):
            return filepath
        if not tf.gfile.Exists(directory):
            tf.gfile.MakeDirs(directory)
        # CVDF mirror of http://yann.lecun.com/exdb/mnist/
        url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
        zipped_filepath = filepath + '.gz'
        print('Downloading %s to %s' % (url, zipped_filepath))
        urllib.request.urlretrieve(url, zipped_filepath)
        with gzip.open(zipped_filepath, 'rb') as f_in, open(filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(zipped_filepath)
        return filepath



    def get_train_iterator(self, dataset):
        dataset = dataset.prefetch(buffer_size=self.batch_size)
        dataset = dataset.repeat(self.num_epochs)
        dataset = dataset.shuffle(buffer_size=self.shuffle_buffer)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator


    def get_test_iterator(self, dataset):
        TEST_NUM = 10000  # the number of mnist-testset
        dataset = dataset.prefetch(buffer_size=self.batch_size)
        dataset = dataset.repeat(1)
        dataset = dataset.batch(TEST_NUM)
        iterator = dataset.make_one_shot_iterator()
        return iterator


    def train_input_fn(self):
        dataset = self.get_dataset('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
        iterator = self.get_train_iterator(dataset)
        images, labels = iterator.get_next()
        return images, labels


    def test_input_fn(self):
        dataset = self.get_dataset('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
        iterator = self.get_test_iterator(dataset)
        images, labels = iterator.get_next()
        return images, labels

    def predict_input_fn(self):
        raise NotImplementedError
