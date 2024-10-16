LogRegClient_params = {
    'centralized': {
        'device': 'cuda:0',
        'lr_patience': 5,
        'es_patience': 15,
        'n_rounds': 100,
        'eval_every': 10,
        'batch_size': 128,
        'optimizer': 'SGD',
        'optimizer_params': {
            'lr': 0.08507714201331565,
            'momentum': 0.8343372940844335,
            'weight_decay': 0.0015318920671180095,
            'dampening': 0.6260300336301439,
        },
        'criterion': 'CrossEntropyLoss',
        'criterion_params': {},
    },
    'federated': {
        'device': 'cuda:0',
        'lr_patience': 5,
        'es_patience': 15,
        'n_rounds': 100,
        'eval_every': 10,
        'batch_size': 128,
        'optimizer': 'SGD',
        'optimizer_params': {
            'lr': 0.029577125072356544,
            'momentum': 0.6954937697844237,
            'weight_decay': 0.0016911232487892836,
            'dampening': 0.007628568457341833,
        },
        'criterion': 'CrossEntropyLoss',
        'criterion_params': {}
    },
    'isolated': {
        'clients': {
            'c0': {
                'device': 'cuda:0',
                'lr_patience': 5,
                'es_patience': 15,
                'n_rounds': 100,
                'eval_every': 10,
                'batch_size': 128,
                'optimizer': 'Adam',
                'optimizer_params': {
                    'lr': 0.03937729543826933,
                    'weight_decay': 0.0005450625188668391,
                    'amsgrad': True
                },
                'criterion': 'CrossEntropyLoss', 
                'criterion_params': {}
            },
            'c1': {
                'device': 'cuda:0',
                'lr_patience': 5,
                'es_patience': 15,
                'n_rounds': 100,
                'eval_every': 10,
                'batch_size': 128,
                'optimizer': 'SGD',
                'optimizer_params': {
                    'lr': 0.0924277036106537,
                    'momentum': 0.8913437725949631,
                    'weight_decay': 0.0006195976833673726,
                    'dampening': 0.12132015743487427,
                },
                'criterion': 'CrossEntropyLoss',
                'criterion_params': {},
            },
            'c2': {
                'device': 'cuda:0',
                'lr_patience': 5,
                'es_patience': 15,
                'n_rounds': 100,
                'eval_every': 10,
                'batch_size': 128,
                'optimizer': 'Adam',
                'optimizer_params': {
                    'lr': 0.06800830792699754,
                    'weight_decay': 0.00012351437711755064,
                    'amsgrad': False
                },
                'criterion': 'CrossEntropyLoss',
                'criterion_params': {},
            }
        }
    }
}

GradientBoostingClient_params = {
    'centralized': {
        'loss': 'log_loss',
        'learning_rate': 0.08700569202616437,
        'n_estimators': 122,
        'criterion': 'friedman_mse',
        'max_depth': 8,
        'random_state': 42
    },
    'isolated': {
        'clients': {
            'c0': {
                'loss': 'exponential',
                'learning_rate': 0.05818935091522384,
                'n_estimators': 194,
                'criterion': 'squared_error',
                'max_depth': 6,
                'random_state': 42
            },
            'c1': {
                'loss': 'log_loss',
                'learning_rate': 0.08657514443468428,
                'n_estimators': 85,
                'criterion': 'friedman_mse',
                'max_depth': 7,
                'random_state': 42
            },
            'c2': {
                'loss': 'log_loss',
                'learning_rate': 0.2106783453669997,
                'n_estimators': 85,
                'criterion': 'squared_error',
                'max_depth': 6,
                'random_state': 42
            }
        }
    }
}

SVMClient_params = {
    'centralized': {
        'C': 3.561624539963823,
        'kernel': 'sigmoid',
        'degree': 2,
        'gamma': 'auto',
        'coef0': 0.9904719650185255,
        'shrinking': True,
        'probability': True,
        'class_weight': 'balanced',
        'cache_size': 7000,
        'max_iter': 1000, 
        'random_state': 42
    },
    'isolated': {
        'clients': {
            'c0': {
                'C': 3.9430543054526375,
                'kernel': 'rbf',
                'degree': 4,
                'gamma': 'scale',
                'coef0': 0.4318228869212707,
                'shrinking': True,
                'probability': True,
                'class_weight': None,
                'cache_size': 7000,
                'max_iter': 1000, 
                'random_state': 42
            },
            'c1': {
                'C': 8.463920117068124,
                'kernel': 'linear',
                'degree': 5,
                'gamma': 'auto',
                'coef0': 0.16800771550250437,
                'shrinking': True,
                'probability': True,
                'class_weight': 'balanced',
                'cache_size': 7000,
                'max_iter': 1000, 
                'random_state': 42
            },
            'c2': {
                'C': 8.960930759300453,
                'kernel': 'linear',
                'degree': 4,
                'gamma': 'auto',
                'coef0': 0.9593958036426028,
                'shrinking': True,
                'probability': True,
                'class_weight': None,
                'cache_size': 7000,
                'max_iter': 1000, 
                'random_state': 42
            }
        }
    }
}

KNNClient_params = {
    'centralized': {
        'n_neighbors': 78,
        'weights': 'distance',
        'algorithm': 'ball_tree',
        'leaf_size': 25,
        'p': 2,
        'metric': 'minkowski',
        'n_jobs': -1
    },
    'isolated': {
        'clients': {
            'c0': {
                'n_neighbors': 68,
                'weights': 'distance',
                'algorithm': 'auto',
                'leaf_size': 16,
                'p': 2,
                'metric': 'minkowski',
                'n_jobs': -1
            },
            'c1': {
                'n_neighbors': 87,
                'weights': 'distance',
                'algorithm': 'auto',
                'leaf_size': 41,
                'p': 2,
                'metric': 'minkowski',
                'n_jobs': -1
            },
            'c2': {
                'n_neighbors': 90,
                'weights': 'distance',
                'algorithm': 'kd_tree',
                'leaf_size': 38,
                'p': 1,
                'metric': 'minkowski',
                'n_jobs': -1
            }
        }
    }
}

DecisionTreeClient_params = {
    'centralized': {
        'criterion': 'entropy',
        'splitter': 'random',
        'max_depth': 9,
        'class_weight': None,
        'random_state': 42,
    },
    'isolated': {
        'clients': {
            'c0': {
                'criterion': 'log_loss',
                'splitter': 'best',
                'max_depth': 7,
                'class_weight': None,
                'random_state': 42,
            },
            'c1': {
                'criterion': 'gini',
                'splitter': 'random',
                'max_depth': 8,
                'class_weight': 'balanced',
                'random_state': 42,
            },
            'c2': {
                'criterion': 'entropy',
                'splitter': 'random',
                'max_depth': 8,
                'class_weight': 'balanced',
                'random_state': 42,
            }
        }
    }
}

RandomForestClient_params = {
    'centralized': {
        'n_estimators': 989,
        'criterion': 'log_loss',
        'max_depth': 31,
        'class_weight': 'balanced',
        'random_state': 42,
    },
    'isolated': {
        'clients': {
            'c0': {
                'n_estimators': 808,
                'criterion': 'log_loss',
                'max_depth': 51,
                'class_weight': 'balanced',
                'random_state': 42,
            },
            'c1': {
                'n_estimators': 971,
                'criterion': 'entropy',
                'max_depth': 57,
                'class_weight': None,
                'random_state': 42,
            },
            'c2': {
                'n_estimators': 795,
                'criterion': 'entropy',
                'max_depth': 28,
                'class_weight': 'balanced',
                'random_state': 42,
            }
        }
    }
}

MLPClient_params = {
    'centralized': {
        'device': 'cuda:0',
        'lr_patience': 5,
        'es_patience': 15,
        'n_rounds': 100,
        'eval_every': 10,
        'batch_size': 128,
        'optimizer': 'Adam',
        'optimizer_params': {
            'lr': 0.01,
            'weight_decay': 0.0,
            'amsgrad': False
        },
        'criterion': 'CrossEntropyLoss',
        'criterion_params': {},
        'n_hidden_layers': 2,
        'hidden_dim': 64,
    },
    'federated': {
        'device': 'cuda:0',
        'lr_patience': 5,
        'es_patience': 15,
        'n_rounds': 100,
        'eval_every': 10,
        'batch_size': 128,
        'optimizer': 'Adam',
        'optimizer_params': {
            'lr': 0.01,
            'weight_decay': 0.0,
            'amsgrad': False
        },
        'criterion': 'CrossEntropyLoss',
        'criterion_params': {},
        'n_hidden_layers': 2,
        'hidden_dim': 64,
    },
    'isolated': {
        'clients': {
            'c0': {
                'device': 'cuda:0',
                'lr_patience': 5,
                'es_patience': 15,
                'n_rounds': 100,
                'eval_every': 10,
                'batch_size': 128,
                'optimizer': 'Adam',
                'optimizer_params': {
                    'lr': 0.01,
                    'weight_decay': 0.0,
                    'amsgrad': False
                },
                'criterion': 'CrossEntropyLoss',
                'criterion_params': {},
                'n_hidden_layers': 2,
                'hidden_dim': 64,
            },
            'c1': {
                'device': 'cuda:0',
                'lr_patience': 5,
                'es_patience': 15,
                'n_rounds': 100,
                'eval_every': 10,
                'batch_size': 128,
                'optimizer': 'Adam',
                'optimizer_params': {
                    'lr': 0.01,
                    'weight_decay': 0.0,
                    'amsgrad': False
                },
                'criterion': 'CrossEntropyLoss',
                'criterion_params': {},
                'n_hidden_layers': 2,
                'hidden_dim': 64,
            },
            'c2': {
                'device': 'cuda:0',
                'lr_patience': 5,
                'es_patience': 15,
                'n_rounds': 100,
                'eval_every': 10,
                'batch_size': 128,
                'optimizer': 'Adam',
                'optimizer_params': {
                    'lr': 0.01,
                    'weight_decay': 0.0,
                    'amsgrad': False
                },
                'criterion': 'CrossEntropyLoss',
                'criterion_params': {},
                'n_hidden_layers': 2,
                'hidden_dim': 64,
            }
        }
    }
}