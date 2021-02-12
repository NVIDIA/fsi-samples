from greenflow.dataframe_flow import Node
from .nemoBaseNode import NeMoBase
import nemo
import nemo.collections.nlp.nm



class BertInferDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.data_layers.bert_inference_datalayer.BertInferDataLayer)



class BertPretrainingDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.data_layers.lm_bert_datalayer.BertPretrainingDataLayer)



class BertPretrainingPreprocessedDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.data_layers.lm_bert_datalayer.BertPretrainingPreprocessedDataLayer)



class TextDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.data_layers.text_datalayer.TextDataLayer)



class MaskedLogLossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.losses.masked_xentropy_loss.MaskedLogLoss)



class SGDDialogueStateLossNMNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.losses.sgd_loss.SGDDialogueStateLossNM)



class SmoothedCrossEntropyLossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.losses.smoothed_cross_entropy_loss.SmoothedCrossEntropyLoss)



class SpanningLossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.losses.spanning_loss.SpanningLoss)



class AlbertNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.common.huggingface.albert_nm.Albert)



class BERTNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.common.huggingface.bert_nm.BERT)



class BeamSearchTranslatorNMNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.common.transformer.transformer_nm.BeamSearchTranslatorNM)



class BertTokenClassifierNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.common.token_classification_nm.BertTokenClassifier)



class EncoderRNNNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.common.encoder_rnn.EncoderRNN)



class GreedyLanguageGeneratorNMNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.common.transformer.transformer_nm.GreedyLanguageGeneratorNM)



class JointIntentSlotClassifierNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.joint_intent_slot.joint_intent_slot_classifier_nm.JointIntentSlotClassifier)



class PunctCapitTokenClassifierNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.punctuation_capitalization.punctuation_capitalization_classifier_nm.PunctCapitTokenClassifier)



class RobertaNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.common.huggingface.roberta_nm.Roberta)



class SGDDecoderNMNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.dialogue_state_tracking.sgd.sgd_decoder_nm.SGDDecoderNM)



class SGDEncoderNMNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.dialogue_state_tracking.sgd.sgd_encoder_nm.SGDEncoderNM)



class SequenceClassifierNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.common.sequence_classification_nm.SequenceClassifier)



class SequenceRegressionNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.common.sequence_regression_nm.SequenceRegression)



class TRADEGeneratorNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.dialogue_state_tracking.trade_generator_nm.TRADEGenerator)



class TokenClassifierNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.common.token_classification_nm.TokenClassifier)



class TransformerDecoderNMNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.common.transformer.transformer_nm.TransformerDecoderNM)



class TransformerEncoderNMNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, nemo.collections.nlp.nm.trainables.common.transformer.transformer_nm.TransformerEncoderNM)
