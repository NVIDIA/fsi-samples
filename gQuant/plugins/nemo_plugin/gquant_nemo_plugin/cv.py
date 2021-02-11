from greenflow.dataframe_flow import Node
from .nemoBaseNode import NeMoBase
import nemo
import nemo.collections.cv



class CIFAR100DataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.cv.modules.data_layers.cifar100_datalayer.CIFAR100DataLayer)



class CIFAR10DataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.cv.modules.data_layers.cifar10_datalayer.CIFAR10DataLayer)



class MNISTDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.cv.modules.data_layers.mnist_datalayer.MNISTDataLayer)



class STL10DataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.cv.modules.data_layers.stl10_datalayer.STL10DataLayer)



class NLLLossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.cv.modules.losses.nll_loss.NLLLoss)



class NonLinearityNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.cv.modules.non_trainables.non_linearity.NonLinearity)



class ReshapeTensorNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.cv.modules.non_trainables.reshape_tensor.ReshapeTensor)



class ConvNetEncoderNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.cv.modules.trainables.convnet_encoder.ConvNetEncoder)



class FeedForwardNetworkNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.cv.modules.trainables.feed_forward_network.FeedForwardNetwork)



class ImageEncoderNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.cv.modules.trainables.image_encoder.ImageEncoder)



class LeNet5Node(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.cv.modules.trainables.lenet5.LeNet5)
