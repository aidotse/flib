
LogRegClient_params = {
    'default': {
        'lr_patience': 100,
        'es_patience': 100,
        'n_rounds': 50,
        'eval_every': 10,
        'batch_size': 512,
        'optimizer': {
            'SGD': {
                'lr': 0.01,
                'momentum': 0.0,
                'weight_decay': 0.0,
                'dampening': 0.0,
            }
        },
        'criterion': {
            'ClassBalancedLoss': {
                'gamma': 0.9
            }
        }
    },
    'search_space': {
        'batch_size': [512], #[128, 256, 512],
        'optimizer': {
            'SGD': {
                'lr': (0.001, 0.1),
                'momentum': (0.0, 1.0),
                'weight_decay': (0.0, 1.0),
                'dampening': (0.0, 1.0),
            },
            'Adam': {
                'lr': (0.0001, 0.1),
                'weight_decay': (0.0, 1.0),
                'amsgrad': [True, False]
            }
        },
        'criterion': {
            'ClassBalancedLoss': {
                'gamma': (0.5, 0.99999)
            },
            #'CrossEntropyLoss': {}
        }
    }
}

DecisionTreeClient_params = {
    'default': {
        'criterion': 'gini',
        'splitter': 'best',
        'max_depth': None,
        'random_state': 42,
        'class_weight': 'balanced'
    },
    'search_space': {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_depth': (1, 100),
        'class_weight': ['balanced', None]
    }
}

RandomForestClient_params = {
    'default': {
        'n_estimators': 100,
        'criterion': 'gini',
        'max_depth': None,
        'random_state': 42,
        'class_weight': 'balanced'
    },
    'search_space': {
        'n_estimators': (10, 1000),
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': (1, 100),
        'class_weight': ['balanced', None]
    }
}

GradientBoostingClient_params = {
    'default': {
        'loss': 'log_loss',
        'learning_rate': 0.1,
        'n_estimators': 100,
        'criterion': 'friedman_mse',
        'max_depth': 3,
        'random_state': 42
    },
    'search_space': {
        'loss': ['log_loss', 'exponential'],
        'learning_rate': (0.01, 1.0),
        'n_estimators': (10, 200),
        'criterion': ['friedman_mse', 'squared_error'],
        'max_depth': (2, 100),
        'random_state': 42
    }
}

SVMClient_params = {
    'default': {
        'C': 1.0,
        'kernel': 'rbf',
        'degree': 3,
        'gamma': 'scale',
        'coef0': 0.0,
        'shrinking': True,
        'probability': False,
        'class_weight': 'balanced',
        'cache_size': 7000,
        'max_iter': 1000, 
        'random_state': 42
    },
    'search_space': {
        'C': (0.1, 10.0),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': (2, 5),
        'gamma': ['scale', 'auto'],
        'coef0': (0.0, 1.0),
        'shrinking': [True, False],
        'probability': [False, True],
        'class_weight': ['balanced', None],
        'random_state': 42
    }
}

KNNClient_params = {
    'default': {
        'n_neighbors': 5,
        'weights': 'uniform',
        'algorithm': 'auto',
        'leaf_size': 30,
        'p': 2,
        'metric': 'minkowski',
        'n_jobs': -1
    },
    'search_space': {
        'n_neighbors': (3, 100),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': (10, 50),
        'p': [1, 2],
        'metric': 'minkowski'
    }
}

MLPClient_params = {
    'default': {
        'lr_patience': 100,
        'es_patience': 100,
        'n_rounds': 50,
        'eval_every': 10,
        'batch_size': 512,
        'optimizer': {
            'SGD': {
                'lr': 0.01,
                'momentum': 0.0,
                'weight_decay': 0.0,
                'dampening': 0.0,
            }
        },
        'criterion': {
            'ClassBalancedLoss': {
                'gamma': 0.9
            }
        },
        'n_hidden_layers': 2,
        'hidden_dim': 64
    },
    'search_space': {
        'batch_size': [128, 256, 512],
        'optimizer': {
            'SGD': {
                'lr': (0.001, 0.1),
                'momentum': (0.0, 1.0),
                'weight_decay': (0.0, 1.0),
                'dampening': (0.0, 1.0),
            },
            'Adam': {
                'lr': (0.001, 0.1),
                'weight_decay': (0.0, 1.0),
                'amsgrad': [True, False]
            }
        },
        'criterion': {
            'ClassBalancedLoss': {
                'gamma': (0.5, 0.9999)
            },
            'CrossEntropyLoss': {}
        },
        'n_hidden_layers': (1, 5),
        'hidden_dim': (32, 256)
    }
}

GraphSAGEClient_params = {
    'default': {
        'n_rounds': 200,
        'eval_every': 10,
        'lr_patience': 200,
        'es_patience': 200,
        'hidden_dim': 64,
        'optimizer': 'Adam',
        'optimizer_params': {
            'lr': 0.01,
            'weight_decay': 0.0,
            'amsgrad': False,
        },
        'criterion': 'CrossEntropyLoss',
        'criterion_params': {}
    },
    'search_space': {
        'hidden_dim': [32, 64, 128, 256, 512],
        'optimizer': {
            'Adam': {
                'lr': (0.001, 0.1),
                'weight_decay': (0.0, 1.0),
                'amsgrad': [False, True],
            }, 
            'SGD': {
                'lr': (0.001, 0.1),
                'momentum': (0.0, 1.0),
                'weight_decay': (0.0, 1.0),
                'dampening': (0.0, 1.0),
            }
        },
        'criterion': {
            'CrossEntropyLoss': {},
            'ClassBalancedLoss': {
                'gamma': (0.5, 0.9999)
            },
            'NLLLoss': {
                'weight': (0.0, 1.0),
            }
        }
    }
}
