from greenflow.dataframe_flow import Node
from .nemoBaseNode import NeMoBase
import nemo
import nemo.backends.pytorch.tutorials



class DialogDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.tutorials.chatbot.modules.DialogDataLayer)



class EncoderRNNNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.tutorials.chatbot.modules.EncoderRNN)



class GreedyLuongAttnDecoderRNNNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.tutorials.chatbot.modules.GreedyLuongAttnDecoderRNN)



class LuongAttnDecoderRNNNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.tutorials.chatbot.modules.LuongAttnDecoderRNN)



class MaskedXEntropyLossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.tutorials.chatbot.modules.MaskedXEntropyLoss)



class CrossEntropyLossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.tutorials.toys.CrossEntropyLoss)



class L1LossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.tutorials.toys.L1Loss)



class MSELossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.tutorials.toys.MSELoss)



class RealFunctionDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.tutorials.toys.RealFunctionDataLayer)



class TaylorNetNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.backends.pytorch.tutorials.toys.TaylorNet)
