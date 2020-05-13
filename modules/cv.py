from gquant.plugin_nodes.nemo_util.nemoBaseNode import NeMoBase
import nemo
import nemo.collections.cv



class CIFAR100DataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.cv.modules.data_layers.cifar100_datalayer.CIFAR100DataLayer)



class CIFAR10DataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.cv.modules.data_layers.cifar10_datalayer.CIFAR10DataLayer)



class MNISTDataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.cv.modules.data_layers.mnist_datalayer.MNISTDataLayer)



class STL10DataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.cv.modules.data_layers.stl10_datalayer.STL10DataLayer)



class NLLLossNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.cv.modules.losses.nll_loss.NLLLoss)



class NonLinearityNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.cv.modules.non_trainables.non_linearity.NonLinearity)



class ReshapeTensorNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.cv.modules.non_trainables.reshape_tensor.ReshapeTensor)



class ConvNetEncoderNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.cv.modules.trainables.convnet_encoder.ConvNetEncoder)



class FeedForwardNetworkNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.cv.modules.trainables.feed_forward_network.FeedForwardNetwork)



class ImageEncoderNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.cv.modules.trainables.image_encoder.ImageEncoder)



class LeNet5Node(NeMoBase):
    def init(self):
        super().init(nemo.collections.cv.modules.trainables.lenet5.LeNet5)
