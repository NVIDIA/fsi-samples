from gquant.dataframe_flow import Node
from .nemoBaseNode import NeMoBase
import nemo
import nemo.collections.asr



class AudioPreprocessorNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.audio_preprocessing.AudioPreprocessor)



class AudioToMFCCPreprocessorNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.audio_preprocessing.AudioToMFCCPreprocessor)



class AudioToMelSpectrogramPreprocessorNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.audio_preprocessing.AudioToMelSpectrogramPreprocessor)



class AudioToSpectrogramPreprocessorNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.audio_preprocessing.AudioToSpectrogramPreprocessor)



class CropOrPadSpectrogramAugmentationNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.audio_preprocessing.CropOrPadSpectrogramAugmentation)



class MultiplyBatchNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.audio_preprocessing.MultiplyBatch)



class SpectrogramAugmentationNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.audio_preprocessing.SpectrogramAugmentation)



class TimeStretchAugmentationNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.audio_preprocessing.TimeStretchAugmentation)



class BeamSearchDecoderWithLMNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.beam_search_decoder.BeamSearchDecoderWithLM)



class ContextNetDecoderForCTCNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.contextnet.ContextNetDecoderForCTC)



class ContextNetEncoderNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.contextnet.ContextNetEncoder)



class JasperEncoderNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.jasper.JasperEncoder)



class AudioToSpeechLabelDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.data_layer.AudioToSpeechLabelDataLayer)



class AudioToTextDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.data_layer.AudioToTextDataLayer)



class KaldiFeatureDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.data_layer.KaldiFeatureDataLayer)



class TarredAudioToTextDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.data_layer.TarredAudioToTextDataLayer)



class TranscriptDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.data_layer.TranscriptDataLayer)



class GreedyCTCDecoderNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.greedy_ctc_decoder.GreedyCTCDecoder)



class JasperDecoderForCTCNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.jasper.JasperDecoderForCTC)



class JasperDecoderForClassificationNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.jasper.JasperDecoderForClassification)



class JasperDecoderForSpkrClassNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.jasper.JasperDecoderForSpkrClass)



class CTCLossNMNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.losses.CTCLossNM)



class ASRConvCTCModelNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.models.asrconvctcmodel.ASRConvCTCModel)



class JasperNetNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.models.asrconvctcmodel.JasperNet)



class QuartzNetNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.asr.models.asrconvctcmodel.QuartzNet)
