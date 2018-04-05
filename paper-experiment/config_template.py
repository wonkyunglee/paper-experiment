import os


class TempConfig(object):

    def __init__(self):

        self.gpu_list = 0

        self.batch_size = 60
        self.num_epochs = 100
        self.shuffle_buffer = 180
        self.model_dir = 'model_dir/multi-label/centerloss/add/'
        self.check_exist_and_mkdir(self.model_dir)


        self.host='host'
        self.port=port
        self.user='user'
        self.password='password'
        self.db='db'
        self.db_query_label = 'select tags from table'
        self.db_query_train = 'select sha, path, tags, rep_tag from table where fold !=7 and fold !=8 and fold !=9'
        self.db_query_valid = 'select sha, path, tags, rep_tag from table where fold =7 or fold =8 or fold =9'
        self.db_sha_idx = 0
        self.db_path_idx = 1
        self.db_labels_idx = 2
        self.db_rep_label_idx = 3


        self.params = dict()
        self.params['hidden_units'] = [1000, 100]
        self.params['filter_num'] = [32, 32, 32, 32, 32]
        self.params['n_classes'] = n_classes
        self.params['learning_rate'] = 0.001
        self.params['model_dir'] = self.model_dir
        self.params['train_data_num'] = 13300
        self.params['valid_data_num'] = 19000-13300
        self.params['batch_norm'] = True


    def check_exist_and_mkdir(self, directory):
        if not os.path.exists(directory):
            print('make dir', directory)
            os.makedirs(directory)




