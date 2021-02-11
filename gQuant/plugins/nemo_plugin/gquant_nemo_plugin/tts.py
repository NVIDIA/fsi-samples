from greenflow.dataframe_flow import Node
from .nemoBaseNode import NeMoBase
import nemo
import nemo.collections.tts



class AudioDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.data_layers.AudioDataLayer)



class FastSpeechNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.fastspeech_modules.FastSpeech)



class FastSpeechDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.fastspeech_modules.FastSpeechDataLayer)



class FastSpeechLossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.fastspeech_modules.FastSpeechLoss)



class MakeGateNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.tacotron2_modules.MakeGate)



class Tacotron2DecoderNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.tacotron2_modules.Tacotron2Decoder)



class Tacotron2DecoderInferNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.tacotron2_modules.Tacotron2DecoderInfer)



class Tacotron2EncoderNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.tacotron2_modules.Tacotron2Encoder)



class Tacotron2LossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.tacotron2_modules.Tacotron2Loss)



class Tacotron2PostnetNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.tacotron2_modules.Tacotron2Postnet)



class TextEmbeddingNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.tacotron2_modules.TextEmbedding)



class LenSamplerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.talknet_modules.LenSampler)



class TalkNetNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.talknet_modules.TalkNet)



class TalkNetDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.talknet_modules.TalkNetDataLayer)



class TalkNetDursLossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.talknet_modules.TalkNetDursLoss)



class TalkNetMelsLossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.talknet_modules.TalkNetMelsLoss)



class WaveGlowInferNMNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.waveglow_modules.WaveGlowInferNM)



class WaveGlowLossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.waveglow_modules.WaveGlowLoss)



class WaveGlowNMNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.tts.waveglow_modules.WaveGlowNM)
