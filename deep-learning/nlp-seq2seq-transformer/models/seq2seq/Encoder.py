"""
S2S Encoder model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the encoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer based on the "model_type" argument.            #
        #          Supported types (strings): "RNN", "LSTM". Instantiate the        #
        #          appropriate layer for the specified model_type.                  #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden states of the Encoder(namely, Linear - ReLU - Linear).    #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #          HINT: the size of the output of the second linear layer must     #
        #          satisfy certain constraint relevant to the decoder.              #
        #       4) A dropout layer                                                  #
        #############################################################################

        # 1) Embedding layer
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=emb_size)

        # 2) Recurrent layer
        if self.model_type == "RNN":
            self.rnn = nn.RNN(input_size=emb_size,
                              hidden_size=self.encoder_hidden_size,
                              batch_first=True)
        elif self.model_type == "LSTM":
            self.rnn = nn.LSTM(input_size=emb_size,
                               hidden_size=self.encoder_hidden_size,
                               batch_first=True)
        else:
            raise ValueError("Unsupported model_type. Use 'RNN' or 'LSTM'.")

        # 3) Two linear layers with a ReLU in between
        #    The first linear layer keeps dimension = encoder_hidden_size
        #    The second linear layer must match decoder_hidden_size
        self.linear1 = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)

        # 4) Dropout
        self.dropout = nn.Dropout(dropout)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)
            Returns:
                output (tensor): the output of the Encoder;
                hidden (tensor): the state coming out of the last hidden unit
        """
        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer.                                                    #
        #                                                                           #
        #       Apply tanh activation to the hidden tensor before returning it.     #
        #                                                                           #
        #       Do not apply any linear layers/Relu for the cell state when         #
        #       model_type is LSTM before returning it.                             #
        #                                                                           #
        #       If model_type is LSTM, the hidden variable returns a tuple          #
        #       containing both the hidden state and the cell state of the LSTM.    #
        #############################################################################

        embedded = self.embedding(input)              # shape: (batch, seq_len, emb_size)
        embedded = self.dropout(embedded)             # apply dropout to embeddings

        if self.model_type == "RNN":
            output, hidden = self.rnn(embedded)       # hidden shape: (1, batch, enc_hidden_size)
            # Take the last hidden state and transform it
            hidden = hidden[-1]                       # shape: (batch, enc_hidden_size)
            hidden = self.linear1(hidden)
            hidden = self.relu(hidden)
            hidden = self.linear2(hidden)             # shape: (batch, decoder_hidden_size)
            hidden = torch.tanh(hidden)               # apply tanh before returning

        elif self.model_type == "LSTM":
            output, (h_n, c_n) = self.rnn(embedded)
            # h_n shape: (1, batch, enc_hidden_size)
            # c_n shape: (1, batch, enc_hidden_size)
            # Only transform h_n for the next stage
            h_n_2d = h_n[-1]                          # shape: (batch, enc_hidden_size)
            h_n_trans = self.linear1(h_n_2d)
            h_n_trans = self.relu(h_n_trans)
            h_n_trans = self.linear2(h_n_trans)       # shape: (batch, decoder_hidden_size)
            h_n_trans = torch.tanh(h_n_trans)
            # Unsqueeze back to 3-D: (1, batch, decoder_hidden_size)
            h_n_final = h_n_trans.unsqueeze(0)
            # Return both h_n_final and c_n as a tuple
            hidden = (h_n_final, c_n)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden