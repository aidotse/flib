LogRegClient_params = {
    'centralized': {
        'device': 'cuda:0',
        'lr_patience': 100,
        'es_patience': 100,
        'n_rounds': 100,
        'eval_every': 10,
        'batch_size': 512, #128,
        'optimizer': 'Adam', #'SGD',
        'optimizer_params': {
            'lr': 0.01, #0.09973026277406209,
            #'momentum': 0.9699303775123841,
            #'weight_decay': 0.7980000343573007,
            #'dampening': 0.15500715715253313,
        },
        'criterion': 'ClassBalancedLoss', #'CrossEntropyLoss',
        'criterion_params': {
            'gamma': 0.9999
        },
    },
    'federated': {
        'device': 'cuda:0',
        'lr_patience': 100,
        'es_patience': 100,
        'n_rounds': 100,
        'eval_every': 10,
        'batch_size': 512, #128,
        'optimizer': 'Adam', #'SGD',
        'optimizer_params': {
            'lr': 0.01, #0.03456591494454122,
            #'momentum': 0.0, #0.5752335830497186,
            #'weight_decay': 0.0, #0.9751077466806092,
            #'dampening': 0.0, #0.19067255025350016,
        },
        'criterion': 'ClassBalancedLoss',
        'criterion_params': {
            'gamma': 0.9999, #0.9994980560757293
        },
        'clients': {
            'c0': {
                'device': 'cuda:0',
                'lr_patience': 100,
                'es_patience': 100,
                'n_rounds': 100,
                'eval_every': 10,
                'batch_size': 512, #128,
                'optimizer': 'Adam',
                'optimizer_params': {
                    'lr': 0.01, #0.059010854331331665,
                    #'weight_decay': 0.4938202384268264,
                    #'amsgrad': True,
                },
                'criterion': 'ClassBalanceLoss', #'CrossEntropyLoss',
                'criterion_params': {
                    'gamma': 0.9999,
                },
            }, 
            'c1': {
                'device': 'cuda:0',
                'lr_patience': 100,
                'es_patience': 100,
                'n_rounds': 100,
                'eval_every': 10,
                'batch_size': 512, #256,
                'optimizer': 'Adam',
                'optimizer_params': {
                    'lr': 0.01, #0.06285523620101933,
                    #'weight_decay': 0.9177014947271461,
                    #'amsgrad': False,
                },
                'criterion': 'ClassBalancedLoss',
                'criterion_params': {
                    'gamma': 0.9999 #0.6432920180664913
                },
            },
            'c2': {
                'device': 'cuda:0',
                'lr_patience': 100,
                'es_patience': 100,
                'n_rounds': 100,
                'eval_every': 10,
                'batch_size': 512,
                'optimizer': 'Adam',
                'optimizer_params': {
                    'lr': 0.01, #0.015842359744286286,
                    #'weight_decay': 0.6321060179101374,
                    #'amsgrad': False,
                },
                'criterion': 'ClassBalancedLoss',
                'criterion_params': {
                    'gamma': 0.9999 #0.7996909689446341
                },
            }
        }
    },
    'isolated': {
        'device': 'cuda:0',
        'lr_patience': 100,
        'es_patience': 100,
        'n_rounds': 100,
        'eval_every': 10,
        'batch_size': 256,
        'optimizer': 'Adam',
        'optimizer_params': {
            'lr': 0.01,
            'weight_decay': 0.0,
            'amsgrad': False
        },
        'criterion': 'ClassBalancedLoss', 
        'criterion_params': {
            'gamma': 0.6
        },
        'clients': {
            'c0': {
                'device': 'cuda:0',
                'lr_patience': 5,
                'es_patience': 15,
                'n_rounds': 100,
                'eval_every': 10,
                'batch_size': 256,
                'optimizer': 'Adam',
                'optimizer_params': {
                    'lr': 0.04,
                    'weight_decay': 0.0,
                    'amsgrad': False
                },
                'criterion': 'ClassBalancedLoss', 
                'criterion_params': {
                    'gamma': 0.6
                }
            },
            'c1': {
                'device': 'cuda:0',
                'lr_patience': 5,
                'es_patience': 15,
                'n_rounds': 100,
                'eval_every': 10,
                'batch_size': 256,
                'optimizer': 'Adam',
                'optimizer_params': {
                    'lr': 0.04,
                    'weight_decay': 0.0,
                    'amsgrad': False
                },
                'criterion': 'ClassBalancedLoss', 
                'criterion_params': {
                    'gamma': 0.6
                }
            },
            'c2': {
                'device': 'cuda:0',
                'lr_patience': 5,
                'es_patience': 15,
                'n_rounds': 100,
                'eval_every': 10,
                'batch_size': 256,
                'optimizer': 'Adam',
                'optimizer_params': {
                    'lr': 0.04,
                    'weight_decay': 0.0,
                    'amsgrad': False
                },
                'criterion': 'ClassBalancedLoss', 
                'criterion_params': {
                    'gamma': 0.6
                }
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
        'criterion': 'log_loss',
        'splitter': 'best',
        'max_depth': 7,
        'class_weight': 'balanced',
        'random_state': 42,
    },
    'isolated': {
        'clients': {
            'c0': {
                'criterion': 'gini',
                'splitter': 'best',
                'max_depth': 4,
                'class_weight': 'balanced',
                'random_state': 42,
            },
            'c1': {
                'criterion': 'entropy',
                'splitter': 'random',
                'max_depth': 5,
                'class_weight': 'balanced',
                'random_state': 42,
            },
            'c2': {
                'criterion': 'entropy',
                'splitter': 'random',
                'max_depth': 4,
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

GraphSAGEClient_params = {
    'centralized': {
        'device': 'cuda:0',
        'n_rounds': 100,
        'eval_every': 10,
        'lr_patience': 100,
        'es_patience': 100,
        'hidden_dim': 128,
        'optimizer': 'Adam',
        'optimizer_params': {
            'lr': 0.001994010234629029,
            'weight_decay': 0.018678778429870604,
            'amsgrad': True,
        },
        'criterion': 'ClassBalancedLoss',
        'criterion_params': {
            'gamma': 0.9936137007812603
        }
    },
    'federated': {
        'device': 'cuda:0',
        'n_rounds': 100,
        'eval_every': 10,
        'lr_patience': 100,
        'es_patience': 100,
        'device': 'cuda:0',
        'hidden_dim': 256,
        'optimizer': 'SGD',
        'optimizer_params': {
            'lr': 0.048630848021750224,
            'momentum': 0.3482146849308458,
            'weight_decay': 0.25496952117003857,
            'dampening': 0.2581674844412982
        },
        'criterion': 'NLLLoss',
        'criterion_params': {
            'weight': 0.8529780681438437
        }
    },
    'isolated': {
        'device': 'cuda:0',
        'n_rounds': 100,
        'eval_every': 10,
        'lr_patience': 100,
        'es_patience': 100,
        'batch_size': 128,
        'hidden_dim': 64,
        'optimizer': 'Adam',
        'optimizer_params': {
            'lr': 0.01,
            'weight_decay': 0.0,
            'amsgrad': False,
        },
        'criterion': 'CrossEntropyLoss',
        'criterion_params': {},
        'clients': {
            'c0': {
                'device': 'cuda:0',
                'n_rounds': 100,
                'eval_every': 10,
                'lr_patience': 100,
                'es_patience': 100,
                'hidden_dim': 512,
                'optimizer': 'Adam',
                'optimizer_params': {
                    'lr': 0.0010042395371481477,
                    'weight_decay': 0.14417159251403777,
                    'amsgrad': False,
                },
                'criterion': 'CrossEntropyLoss',
                'criterion_params': {}
            },
            'c1': {
                'device': 'cuda:0',
                'n_rounds': 100,
                'eval_every': 10,
                'lr_patience': 100,
                'es_patience': 100,
                'hidden_dim': 512,
                'optimizer': 'SGD',
                'optimizer_params': {
                    'lr': 0.07331504183228132,
                    'momentum': 0.23873608202007257,
                    'weight_decay': 0.14324551236448438,
                    'dampening': 0.662142529500039,
                },
                'criterion': 'NLLLoss',
                'criterion_params': {
                    'weight': 0.7676710189846737
                }
            },
            'c2': {
                'device': 'cuda:0',
                'n_rounds': 100,
                'eval_every': 10,
                'lr_patience': 100,
                'es_patience': 100,
                'hidden_dim': 32, #32,
                'optimizer': 'SGD',
                'optimizer_params': {
                    'lr': 0.009297617869520835,
                    'momentum': 0.10013231108969067,
                    'weight_decay': 0.12498499034883515,
                    'dampening': 0.4599952706941705,
                },
                'criterion': 'NLLLoss',
                'criterion_params': {
                    'weight': 0.6950743876647961,
                }
            }
        }
    }
}