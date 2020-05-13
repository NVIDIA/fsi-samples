from gquant.plugin_nodes.nemo_util.nemoBaseNode import NeMoBase
import nemo
import nemo.backends.pytorch.tutorials



class DialogDataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.tutorials.chatbot.modules.DialogDataLayer)



class EncoderRNNNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.tutorials.chatbot.modules.EncoderRNN)



class GreedyLuongAttnDecoderRNNNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.tutorials.chatbot.modules.GreedyLuongAttnDecoderRNN)



class LuongAttnDecoderRNNNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.tutorials.chatbot.modules.LuongAttnDecoderRNN)



class MaskedXEntropyLossNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.tutorials.chatbot.modules.MaskedXEntropyLoss)



class DialogDataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.tutorials.chatbot.modules.DialogDataLayer)



class EncoderRNNNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.tutorials.chatbot.modules.EncoderRNN)



class GreedyLuongAttnDecoderRNNNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.tutorials.chatbot.modules.GreedyLuongAttnDecoderRNN)



class LuongAttnDecoderRNNNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.tutorials.chatbot.modules.LuongAttnDecoderRNN)



class MaskedXEntropyLossNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.tutorials.chatbot.modules.MaskedXEntropyLoss)



class CrossEntropyLossNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.tutorials.toys.CrossEntropyLoss)



class L1LossNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.tutorials.toys.L1Loss)



class MSELossNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.tutorials.toys.MSELoss)



class RealFunctionDataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.tutorials.toys.RealFunctionDataLayer)



class TaylorNetNode(NeMoBase):
    def init(self):
        super().init(nemo.backends.pytorch.tutorials.toys.TaylorNet)
