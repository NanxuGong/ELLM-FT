task_dict = {'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary':
    'cls', 'bike_share': 'reg', 'german_credit': 'cls', 'higgs': 'cls',
             'housing_boston': 'reg', 'ionosphere': 'cls', 'lymphography': 'cls',
             'messidor_features': 'cls', 'openml_620': 'reg', 'pima_indian': 'cls',
             'spam_base': 'cls', 'spectf': 'cls', 'svmguide3': 'cls',
             'uci_credit_card': 'cls', 'wine_red': 'cls', 'wine_white': 'cls',
             'openml_586': 'reg', 'openml_589': 'reg', 'openml_607': 'reg',
             'openml_616': 'reg', 'openml_618': 'reg', 'openml_637': 'reg', 'smtp':
                 'det', 'thyroid': 'det', 'yeast': 'det', 'wbc': 'det', 'mammography': 'det',
                 'npha': 'cls',
             }
task_type = {'reg', 'cls', 'det', 'rank'}
task_measure = {'reg': 'rae', 'cls': 'f1', 'det': 'ROC AUC Score', 'rank':
    'auprc'}
state_rep = {'mds', 'gcn', 'ae', 'mds+ae', 'mds+ae+gcn'}
support_rl_method = {'dqn', 'ddqn', 'dueling_dqn', 'dueling_ddqn'}
base_path = 'data/'
config_map = {'airfoil': ('airfoil.json', 'airfoil_config.yml', True),
              'amazon_employee': ('amazon_employee.json', 'amazon_employee.yml',
                                  False), 'ap_omentum_ovary': ('ap.json', 'ap_config.yml', True),
              'bike_share': ('bike_share.json', 'bike_share_config.yml', False),
              'german_credit': ('german_credit.json', 'german_credit_config.yml',
                                False), 'higgs': ('higgs.json', 'higgs_config.yml', False),
              'housing_boston': ('house_boston.json', 'house_boston_config.yml', True
                                 ), 'ionosphere': ('ionosphere.json', 'ionosphere_config.yml', False),
              'lymphography': ('lymphography.json', 'lymphography.yml', True),
              'messidor_features': ('messidor_features.json',
                                    'messidor_features_config.yml', False), 'openml_620': (
        'openml_620.json', 'openml_620_config.yml', True), 'pima_indian': (
        'pima.json', 'pima_config.yml', True), 'spam_base': ('spambase.json',
                                                             'spambase_config.yml', True), 'spectf': ('spectf.json',
                                                                                                      'spectf_config.yml',
                                                                                                      False),
              'svmguide3': ('svmguide3.json',
                            'svmguide3_config.yml', True), 'uci_credit_card': ('credit_card.json',
                                                                               'credit_card_config.yml', True),
              'wine_red': ('wine_red.json',
                           'wine_red_config.yml', False), 'wine_white': ('wine_white.json',
                                                                         'wine_white_config.yml', False),
              'openml_586': ('openml_586.json',
                             'openml_586_config.yml', True), 'openml_589': ('openml_589.json',
                                                                            'openml_589_config.yml', False),
              'openml_607': ('openml_607.json',
                             'openml_607_config.yml', False), 'openml_616': ('openml_616.json',
                                                                             'openml_616_config.yml', True),
              'openml_618': ('config.json',
                             'openml_618_config.yml', False), 'openml_637': ('openml_637.json',
                                                                             'openml_637_config.yml', True),
              'default': ('search_space.json',
                          'config.yml', True)}
yml_config = {'trial.default_script':
                  '~/miniconda3/envs/nni/bin/python -u '
                  '~/jupyter_base/RL-TKDE/_code/_script/GRFG_with_nni.py '
    , 'maxTrialNum': 100, 'trialConcurrency': 25}

if __name__ == '__main__':
    import os

    print(os.curdir)
    base = './_dataset/RC_dataset/processed/'
    for name, task in task_dict.items():
        if not os.path.exists(base + name + '.hdf'):
            print(name)
