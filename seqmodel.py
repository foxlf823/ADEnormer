

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from wordsequence import WordSequence
from crf import CRF
import logging

class SeqModel(nn.Module):
    def __init__(self, data, opt):
        super(SeqModel, self).__init__()

        self.gpu = opt.gpu

        ## add two more label for downlayer lstm, use original label size for CRF
        self.word_hidden = WordSequence(data, opt)
        self.crf = CRF(data.label_alphabet.size(), self.gpu)


    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask, feature_inputs, text_inputs):

        outs = self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, feature_inputs, text_inputs)
        batch_size = word_inputs.size(0)

        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)

        scores, tag_seq = self.crf._viterbi_decode(outs, mask)

        total_loss = total_loss / batch_size

        return total_loss, tag_seq


    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, feature_inputs, text_inputs):
        outs = self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, feature_inputs, text_inputs)

        scores, tag_seq = self.crf._viterbi_decode(outs, mask)

        return tag_seq


    def decode_nbest(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, nbest, feature_inputs, text_inputs):

        outs = self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, feature_inputs, text_inputs)

        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq

        