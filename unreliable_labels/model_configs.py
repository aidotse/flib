from modules import GraphSAGE, GCN, GAT, GAT2, GAT2NO_NORM, GraphSAGE2, GCN2
# Define model configurations in a dictionary

configs = {
    "100K_accts_EASY25_NEW_NEW": {
        "GraphSAGE": {
            "model_class": GraphSAGE2,
            "epochs": 1000,
            "learning_rate": 0.001765536234432861,
            "input_dim": None,  # to be filled in dynamically
            "hidden_dim": 45,
            "output_dim": 2,
            "dropout": 0.18769996929143462,
            "plot_name": "GraphSAGE_precision_recall",
        },
        "GCN": {
            "model_class": GCN2,
            "epochs": 1000,
            "learning_rate":  0.004331639628376353,
            "input_dim": None,
            "hidden_dim": 128,
            "output_dim": 2,
            "dropout": 0.427320205112153,
            "plot_name": "GCN_precision_recall",
        },
        "GAT": {
            "model_class": GAT2NO_NORM,
            "epochs": 1000,
            "learning_rate": 0.009024128239127212,
            "in_channels": None,
            "hidden_channels": 32,
            "out_channels": 2,
            "dropout": 0.10015263490284221,
            "plot_name": "GAT_precision_recall",
            "num_heads": 8,
        }
    },
    "100K_accts_MID5_NEW_NEW": {
        "GraphSAGE": {
            "model_class": GraphSAGE2,
            "epochs": 1000,
            "learning_rate": 0.013878573173920782,
            "input_dim": None,
            "hidden_dim": 74,
            "output_dim": 2,
            "dropout": 0.34427484527577384,
            "plot_name": "GraphSAGE_precision_recall",
        },
        "GCN": {
            "model_class": GCN2,
            "epochs": 1000,
            "learning_rate": 0.011023326391573367,
            "input_dim": None,
            "hidden_dim": 128,
            "output_dim": 2,
            "dropout": 0.23113192713936842,
            "plot_name": "GCN_precision_recall",
            
        },
        "GAT": {
            "model_class": GAT2NO_NORM,
            "epochs": 1000,
            "learning_rate": 0.024424403802276087,
            "in_channels": None,
            "hidden_channels": 32,
            "out_channels": 2,
            "dropout": 0.14016326857972194,
            "plot_name": "GAT_precision_recall",
            "num_heads": 4,
        }
    }
    ,
    "100K_accts_HARD1_NEW_NEW": {
        "GraphSAGE": {
            "model_class": GraphSAGE2,
            "epochs": 1000,
            "learning_rate": 0.0005933457573893414,
            "input_dim": None,
            "hidden_dim": 61,
            "output_dim": 2,
            "dropout": 0.4275271124641962,
            "plot_name": "GraphSAGE_precision_recall",
        },
        "GCN": {
            "model_class": GCN2,
            "epochs": 1000,
            "learning_rate": 0.02272154701505407,
            "input_dim": None,
            "hidden_dim": 128,
            "output_dim": 2,
            "dropout": 0.1484173142138741,
            "plot_name": "GCN_precision_recall"
        },
        "GAT": {
            "model_class": GAT2NO_NORM,
            "epochs": 1000,
            "learning_rate": 0.06988707124912724,
            "in_channels": None,
            "hidden_channels": 32,
            "out_channels": 2,
            "dropout": 0.16235140144297927,
            "plot_name": "GAT_precision_recall",
            "num_heads": 1,
        }
    }
    

}
