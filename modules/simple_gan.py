from gquant.plugin_nodes.nemo_util.nemoBaseNode import NeMoBase
import nemo
import nemo.collections.simple_gan



class DiscriminatorLossNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.simple_gan.gan.DiscriminatorLoss)



class GradientPenaltyNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.simple_gan.gan.GradientPenalty)



class InterpolateImageNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.simple_gan.gan.InterpolateImage)



class MnistGanDataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.simple_gan.gan.MnistGanDataLayer)



class RandomDataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.simple_gan.gan.RandomDataLayer)



class SimpleDiscriminatorNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.simple_gan.gan.SimpleDiscriminator)



class SimpleGeneratorNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.simple_gan.gan.SimpleGenerator)
