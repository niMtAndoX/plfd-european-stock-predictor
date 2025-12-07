"""
Hyperparameter search spaces for each model.

Each dict maps parameter name -> list of candidate values.
We keep ~5 tunable parameters per model to avoid combinatorial explosion.
"""

# CNN ------------------------------------------------------------------------

CNN_SEARCH = {
    "seq_len": [20, 30, 40],
    "hidden_channels": [16, 32, 64],
    "kernel_size": [3, 5],
    "dropout": [0.1, 0.2],
    "lr": [1e-4, 1e-3],
}

# SACLSTM --------------------------------------------------------------------

SACLSTM_SEARCH = {
    "seq_len": [20, 30],
    "conv_channels": [16, 32],
    "kernel_size": [3, 5],
    "lstm_hidden": [32, 64],
    "lr": [1e-4, 1e-3],
}

# SCINet ---------------------------------------------------------------------

SCINET_SEARCH = {
    "seq_len": [20, 30],
    "stacks": [1, 2],
    "levels": [2, 3],
    "hidden_channels": [16, 32],
    "kernel_size": [3, 5],
}

# Random Forest --------------------------------------------------------------

RF_SEARCH = {
    "n_estimators": [200, 500],
    "max_depth": [5, 8, None],
    "min_samples_leaf": [1, 3, 5],
    "min_samples_split": [2, 6],
}

# XGBoost --------------------------------------------------------------------

XGB_SEARCH = {
    "n_estimators": [300, 600],
    "learning_rate": [0.01, 0.05],
    "max_depth": [3, 4, 5],
    "subsample": [0.7, 0.9],
}

# RNN ------------------------------------------------------------------------

RNN_SEARCH = {
    "seq_len": [20, 30],
    "hidden_size": [32, 64],
    "num_layers": [1, 2],
    "lr": [1e-4, 1e-3],
}

# MTSMFF ---------------------------------------------------------------------

MTSMFF_SEARCH = {
    "seq_len": [20, 30],
    "out_len": [1, 3],
    "enc_hidden": [32, 64],
    "num_layers": [1, 2],
    "dropout": [0.1, 0.2],
}

# Dilated RNN ----------------------------------------------------------------

DILATED_RNN_SEARCH = {
    "seq_len": [20, 30],
    "hidden_size": [32, 64],
    "dilations": [
        [1, 2, 4],
        [1, 3, 9],
    ],
    "lr": [1e-4, 1e-3],
}

# Transformer ---------------------------------------------------------------

TRANSFORMER_SEARCH = {
    "seq_len": [20, 30],
    "d_model": [32, 64],
    "nhead": [2, 4],
    "num_layers": [1, 2],
}

# Temporal Fusion Transformer (TFT) -----------------------------------------

TFT_SEARCH = {
    "seq_len": [20, 30],
    "out_len": [1, 3],
    "d_model": [32, 64],
    "lstm_hidden": [32, 64],
    "nhead": [2, 4],
}

# Pyraformer ----------------------------------------------------------------

PYRAFORMER_SEARCH = {
    "seq_len": [20, 30],
    "out_len": [1, 3],
    "d_model": [32, 64],
    "n_heads": [2, 4],
    "num_layers": [2, 3],
}

# Preformer -----------------------------------------------------------------

PREFORMER_SEARCH = {
    "seq_len": [20, 30],
    "out_len": [1, 3],
    "d_model": [32, 64],
    "seg_lens": [[4, 8], [8, 16]],
    "num_layers": [1, 2],
}

# Autoformer ----------------------------------------------------------------

AUTOFORMER_SEARCH = {
    "seq_len": [20, 30],
    "out_len": [1, 3],
    "d_model": [32, 64],
    "moving_avg": [12, 25, 50],
    "e_layers": [1, 2],
}
