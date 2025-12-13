
"""
Central registry that maps model names -> (class, search space).
Import this everywhere to stay consistent.
"""
from src.models.cnn.cnn import CNNModel
from src.models.cnn.saclstm import SACLSTMModel
from src.models.cnn.scinet import SCINetModel

from src.models.decision_trees.random_forest import RandomForestModel
from src.models.decision_trees.xg_boost import XGBoostModel

from src.models.rnn.rnn import RNNModel
from src.models.rnn.mtsmff import MTSMFFModel
from src.models.rnn.dilated_rnn import DilatedRNNModel

from src.models.transformer.transformer import TransformerModel
from src.models.transformer.temporal_fusion_transformer import TFTModel
from src.models.transformer.pyraformer import PyraformerModel
from src.models.transformer.preformer import PreformerModel
from src.models.transformer.autoformer import AutoformerModel

from .search_spaces import (
    CNN_SEARCH,
    SACLSTM_SEARCH,
    SCINET_SEARCH,
    RF_SEARCH,
    XGB_SEARCH,
    RNN_SEARCH,
    MTSMFF_SEARCH,
    DILATED_RNN_SEARCH,
    TRANSFORMER_SEARCH,
    TFT_SEARCH,
    PYRAFORMER_SEARCH,
    PREFORMER_SEARCH,
    AUTOFORMER_SEARCH,
)


MODEL_REGISTRY = {
    "CNN": {"class": CNNModel, "search": CNN_SEARCH},
    "SACLSTM": {"class": SACLSTMModel, "search": SACLSTM_SEARCH},
    "SCINET": {"class": SCINetModel, "search": SCINET_SEARCH},
    "RANDOM_FOREST": {"class": RandomForestModel, "search": RF_SEARCH},
    "XGBOOST": {"class": XGBoostModel, "search": XGB_SEARCH},
    "RNN": {"class": RNNModel, "search": RNN_SEARCH},
    "MTSMFF": {"class": MTSMFFModel, "search": MTSMFF_SEARCH},
    "DILATED_RNN": {"class": DilatedRNNModel, "search": DILATED_RNN_SEARCH},
    "TRANSFORMER": {"class": TransformerModel, "search": TRANSFORMER_SEARCH},
    "TFT": {"class": TFTModel, "search": TFT_SEARCH},
    "PYRAFORMER": {"class": PyraformerModel, "search": PYRAFORMER_SEARCH},
    "PREFORMER": {"class": PreformerModel, "search": PREFORMER_SEARCH},
    "AUTOFORMER": {"class": AutoformerModel, "search": AUTOFORMER_SEARCH},
}
