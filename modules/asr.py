from gquant.plugin_nodes.nemo_util.nemoBaseNode import NeMoBase
import nemo
import nemo.collections.asr



class AudioPreprocessorNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.audio_preprocessing.AudioPreprocessor)



class AudioToMFCCPreprocessorNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.audio_preprocessing.AudioToMFCCPreprocessor)



class AudioToMelSpectrogramPreprocessorNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.audio_preprocessing.AudioToMelSpectrogramPreprocessor)



class AudioToSpectrogramPreprocessorNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.audio_preprocessing.AudioToSpectrogramPreprocessor)



class CropOrPadSpectrogramAugmentationNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.audio_preprocessing.CropOrPadSpectrogramAugmentation)



class MultiplyBatchNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.audio_preprocessing.MultiplyBatch)



class SpectrogramAugmentationNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.audio_preprocessing.SpectrogramAugmentation)



class TimeStretchAugmentationNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.audio_preprocessing.TimeStretchAugmentation)



class BeamSearchDecoderWithLMNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.beam_search_decoder.BeamSearchDecoderWithLM)



class ContextNetDecoderForCTCNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.contextnet.ContextNetDecoderForCTC)



class ContextNetEncoderNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.contextnet.ContextNetEncoder)



class JasperEncoderNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.jasper.JasperEncoder)



class AudioToSpeechLabelDataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.data_layer.AudioToSpeechLabelDataLayer)



class AudioToTextDataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.data_layer.AudioToTextDataLayer)



class KaldiFeatureDataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.data_layer.KaldiFeatureDataLayer)



class TarredAudioToTextDataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.data_layer.TarredAudioToTextDataLayer)



class TranscriptDataLayerNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.data_layer.TranscriptDataLayer)



class GreedyCTCDecoderNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.greedy_ctc_decoder.GreedyCTCDecoder)



class JasperDecoderForCTCNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.jasper.JasperDecoderForCTC)



class JasperDecoderForClassificationNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.jasper.JasperDecoderForClassification)



class JasperDecoderForSpkrClassNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.jasper.JasperDecoderForSpkrClass)



class JasperEncoderNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.jasper.JasperEncoder)



class CTCLossNMNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.losses.CTCLossNM)



class ASRConvCTCModelNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.models.asrconvctcmodel.ASRConvCTCModel)



class JasperNetNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.models.asrconvctcmodel.JasperNet)



class QuartzNetNode(NeMoBase):
    def init(self):
        super().init(nemo.collections.asr.models.asrconvctcmodel.QuartzNet)
