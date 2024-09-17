REGISTRY = {}

from .rnn_agent import RNNAgent
from .LINDA_agent import LINDAAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["linda"] = LINDAAgent