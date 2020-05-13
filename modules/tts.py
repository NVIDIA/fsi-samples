from gquant.plugin_nodes.nemo_util.nemoBaseNode import NeMoBase
import nemo
import nemo.collections.tts



class AudioDataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.data_layers.AudioDataLayer)



class FastSpeechNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.fastspeech_modules.FastSpeech)



class FastSpeechDataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.fastspeech_modules.FastSpeechDataLayer)



class FastSpeechLossNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.fastspeech_modules.FastSpeechLoss)



class MakeGateNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.tacotron2_modules.MakeGate)



class Tacotron2DecoderNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.tacotron2_modules.Tacotron2Decoder)



class Tacotron2DecoderInferNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.tacotron2_modules.Tacotron2DecoderInfer)



class Tacotron2EncoderNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.tacotron2_modules.Tacotron2Encoder)



class Tacotron2LossNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.tacotron2_modules.Tacotron2Loss)



class Tacotron2PostnetNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.tacotron2_modules.Tacotron2Postnet)



class TextEmbeddingNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.tacotron2_modules.TextEmbedding)



class LenSamplerNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.talknet_modules.LenSampler)



class TalkNetNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.talknet_modules.TalkNet)



class TalkNetDataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.talknet_modules.TalkNetDataLayer)



class TalkNetDursLossNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.talknet_modules.TalkNetDursLoss)



class TalkNetMelsLossNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.talknet_modules.TalkNetMelsLoss)



class WaveGlowInferNMNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.waveglow_modules.WaveGlowInferNM)



class WaveGlowLossNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.waveglow_modules.WaveGlowLoss)



class WaveGlowNMNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.tts.waveglow_modules.WaveGlowNM)
