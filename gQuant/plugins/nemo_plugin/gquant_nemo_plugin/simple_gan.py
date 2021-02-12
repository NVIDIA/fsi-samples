from gquant.dataframe_flow import Node
from .nemoBaseNode import NeMoBase
import nemo
import nemo.collections.simple_gan



class DiscriminatorLossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.simple_gan.gan.DiscriminatorLoss)



class GradientPenaltyNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.simple_gan.gan.GradientPenalty)



class InterpolateImageNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.simple_gan.gan.InterpolateImage)



class MnistGanDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.simple_gan.gan.MnistGanDataLayer)



class RandomDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.simple_gan.gan.RandomDataLayer)



class SimpleDiscriminatorNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.simple_gan.gan.SimpleDiscriminator)



class SimpleGeneratorNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.simple_gan.gan.SimpleGenerator)
