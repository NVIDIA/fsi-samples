from greenflow.dataframe_flow import Node
from .nemoBaseNode import NeMoBase
import nemo
import nemo.backends.pytorch.common



class BCEWithLogitsLossNMNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.common.losses.BCEWithLogitsLossNM)



class CrossEntropyLossNMNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.common.losses.CrossEntropyLossNM)



class LossAggregatorNMNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.common.losses.LossAggregatorNM)



class MSELossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.common.losses.MSELoss)



class SequenceLossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.common.losses.SequenceLoss)



class SequenceEmbeddingNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.common.other.SequenceEmbedding)



class ZerosLikeNMNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.common.other.ZerosLikeNM)



class DecoderRNNNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.common.rnn.DecoderRNN)



class EncoderRNNNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.common.rnn.EncoderRNN)



class NeMoModelNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.core.nemo_model.NeMoModel)



class NeuralModuleNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.core.neural_modules.NeuralModule)



class BeamSearchNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.common.search.BeamSearch)



class GreedySearchNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.common.search.GreedySearch)



class ZerosDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.common.zero_data.ZerosDataLayer)
