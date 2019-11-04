
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from wordrep import WordRep

class WordSequence(nn.Module):
    def __init__(self, data, opt):
        super(WordSequence, self).__init__()

        self.gpu = opt.gpu

        self.droplstm = nn.Dropout(opt.dropout)

        self.wordrep = WordRep(data, opt)
        self.input_size = data.word_emb_dim

        self.input_size += opt.char_hidden_dim

        if data.feat_config is not None:
            for idx in range(len(data.feature_emb_dims)):
                self.input_size += data.feature_emb_dims[idx]

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        lstm_hidden = opt.hidden_dim // 2

        self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(opt.hidden_dim, data.label_alphabet.size()+2)

        if opt.gpu >= 0 and torch.cuda.is_available():
            self.droplstm = self.droplstm.cuda(self.gpu)
            self.hidden2tag = self.hidden2tag.cuda(self.gpu)
            self.lstm = self.lstm.cuda(self.gpu)


    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, feature_inputs, text_inputs):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output: 
                Variable(batch_size, sent_len, hidden_dim)
        """
        word_represent = self.wordrep(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, feature_inputs, text_inputs)
        ## word_embs (batch_size, seq_len, embed_size)

        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        ## lstm_out (seq_len, seq_len, hidden_size)
        feature_out = self.droplstm(lstm_out.transpose(1,0))
        ## feature_out (batch_size, seq_len, hidden_size)
        outputs = self.hidden2tag(feature_out)
        return outputs
