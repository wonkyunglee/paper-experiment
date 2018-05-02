import tensorflow as tf
from tensorflow.python import debug as tf_debug
from utils import convert_path_to_module_name
import importlib
import os


flags = tf.app.flags
flags.DEFINE_string('config_path', 'config.py', 'A config file path')
FLAGS = flags.FLAGS
config_module_name = convert_path_to_module_name(FLAGS.config_path)

config_module = importlib.import_module(config_module_name)
config = config_module.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_list)

def main():

    tf.logging.set_verbosity(tf.logging.INFO)
    loader = config.get_loader()
    estimator = config.get_estimator()

    eval_result = estimator.evaluate(input_fn=loader.valid_input_fn)
    print(eval_result)


if __name__ == '__main__':
    main()
