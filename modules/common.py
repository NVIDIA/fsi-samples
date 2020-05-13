from gquant.plugin_nodes.nemo_util.nemoBaseNode import NeMoBase
import nemo
import nemo.backends.pytorch.common



class BCEWithLogitsLossNMNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.common.losses.BCEWithLogitsLossNM)



class CrossEntropyLossNMNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.common.losses.CrossEntropyLossNM)



class LossAggregatorNMNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.common.losses.LossAggregatorNM)



class MSELossNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.common.losses.MSELoss)



class SequenceLossNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.common.losses.SequenceLoss)



class SequenceEmbeddingNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.common.other.SequenceEmbedding)



class ZerosLikeNMNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.common.other.ZerosLikeNM)



class DecoderRNNNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.common.rnn.DecoderRNN)



class EncoderRNNNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.common.rnn.EncoderRNN)



class NeMoModelNode(NeMoBase):
    def init(self):
        super().init(nemo.core.nemo_model.NeMoModel)



class NeuralModuleNode(NeMoBase):
    def init(self):
        super().init(nemo.core.neural_modules.NeuralModule)



class BeamSearchNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.common.search.BeamSearch)



class GreedySearchNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.common.search.GreedySearch)



class ZerosDataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.common.zero_data.ZerosDataLayer)
