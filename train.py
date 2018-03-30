
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from loader import ThumbnailClassificationLoader, ThumbnailMultilabelLoader
from estimator import Classifier, MultilabelClassifier, SinglelabelClassifier, MultilabelCenterlossClassifier
import os

from config import TempConfig


def main():

    tf.logging.set_verbosity(tf.logging.INFO)
    config = TempConfig()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_list)
    #loader = ThumbnailClassificationLoader(config)
    loader = ThumbnailMultilabelLoader(config)
    loader.make_dicts_for_labels()
    train_input_fn = loader.train_input_fn
    valid_input_fn = loader.valid_input_fn
    #estimator = SinglelabelClassifier(config).get_estimator()
    #estimator = MultilabelClassifier(config).get_estimator()
    estimator = MultilabelCenterlossClassifier(config).get_estimator()

    estimator.train(input_fn=train_input_fn)

    eval_result = estimator.evaluate(input_fn=valid_input_fn)
    print(eval_result)


if __name__ == '__main__':
    main()
