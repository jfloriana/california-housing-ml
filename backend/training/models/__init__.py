from .classic_1_mlp import build_mlp
from .classic_2_deep_mlp import build_deep_mlp
from .classic_3_resnet import build_residual_mlp
from .hybrid_4_cnn_lstm import build_cnn_lstm
from .hybrid_5_autoencoder_mlp import build_autoencoder_mlp

__all__ = [
    "build_mlp",
    "build_deep_mlp",
    "build_residual_mlp",
    "build_cnn_lstm",
    "build_autoencoder_mlp",
]
