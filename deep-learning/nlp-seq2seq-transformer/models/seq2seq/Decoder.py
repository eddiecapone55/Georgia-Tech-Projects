"""
S2S Decoder model.  (c) 2021 Georgia Tech

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
import torch.nn.functional as F


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size,
                 dropout=0.2, model_type="RNN", attention=False):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type
        self.attention = attention

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer based on the "model_type" argument.            #
        #          Supported types (strings): "RNN", "LSTM".                        #
        #       3) A single linear layer with a (log)softmax layer for output        #
        #       4) A dropout layer                                                  #
        #       5) If attention is True, a linear layer to downsize concatenation    #
        #          of context vector and input. (Used after computing attention)     #
        #############################################################################

        # 1) Embedding layer
        self.embedding = nn.Embedding(num_embeddings=self.output_size, embedding_dim=self.emb_size)

        # 2) Recurrent layer
        if self.model_type == "RNN":
            self.rnn = nn.RNN(input_size=self.emb_size, hidden_size=self.decoder_hidden_size,
                              batch_first=True)
        elif self.model_type == "LSTM":
            self.rnn = nn.LSTM(input_size=self.emb_size, hidden_size=self.decoder_hidden_size,
                               batch_first=True)
        else:
            raise ValueError("Unsupported model_type. Use 'RNN' or 'LSTM'.")

        # 3) A single linear layer + log softmax
        self.fc_out = nn.Linear(self.decoder_hidden_size, self.output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # 4) Dropout
        self.dropout = nn.Dropout(dropout)

        # 5) Attention-based linear layer (for context vector + input) if attention = True
        if self.attention:
            self.attn_combine = nn.Linear(self.decoder_hidden_size + self.emb_size, self.emb_size)
            # New: Projection layer to map encoder outputs from encoder_hidden_size to decoder_hidden_size
            self.encoder_projection = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def compute_attention(self, hidden, encoder_outputs):
        """ Compute attention probabilities given a controller state (hidden) and encoder_outputs 
            using cosine similarity as your attention function.
            hidden: shape (1, N, hidden_dim)
            encoder_outputs: shape (N, T, hidden_dim)
            Returns: attention (N, 1, T)
        """
        # Reshape hidden from (1, N, hidden_dim) to (N, hidden_dim)
        hidden_2d = hidden.squeeze(0)

        # Expand hidden_2d to match encoder_outputs shape and compute cosine similarity
        hidden_expanded = hidden_2d.unsqueeze(1).expand_as(encoder_outputs)
        sim = F.cosine_similarity(encoder_outputs, hidden_expanded, dim=-1)

        # Apply softmax over the time dimension (T)
        attn_weights = torch.softmax(sim, dim=1)
        # Unsqueeze to get shape (N, 1, T)
        attn_weights = attn_weights.unsqueeze(1)
        return attn_weights

    def forward(self, input, hidden, encoder_outputs=None):
        """ The forward pass of the decoder.
            Args:
                input (tensor): shape (N, 1)  # a single time-step's input token index
                hidden (tensor): shape (1,N,decoder_hidden_size) for RNN 
                                 or a tuple ( (1,N,decoder_hidden_size), (1,N,decoder_hidden_size) ) for LSTM
                encoder_outputs (tensor): shape (N,T,encoder_hidden_size) for attention
            Returns:
                output (tensor): shape (N, output_size)
                hidden (tensor): updated hidden state
        """
        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       1) Apply dropout to the embedding layer.                            #
        #       2) If attention is true, compute attention probabilities, do a        #
        #          weighted sum on encoder_outputs to get a context vector,          #
        #          concatenate it with the embedding (with context on the left and     #
        #          input on the right), and then feed into attn_combine before        #
        #          passing to the RNN.                                               #
        #       3) If attention is false, just feed the embedding into the RNN.      #
        #       4) The final output is fc_out -> log_softmax.                        #
        # NOTE: For LSTM, hidden is a tuple (h, c) and should be passed accordingly.#
        #############################################################################

        # 1) Embedding + dropout
        embedded = self.embedding(input)  # (N, 1, emb_size)
        embedded = self.dropout(embedded)

        if self.attention and encoder_outputs is not None:
            # Project encoder_outputs to decoder hidden size
            projected_encoder_outputs = self.encoder_projection(encoder_outputs)  # (N, T, decoder_hidden_size)
            if self.model_type == "LSTM":
                h, c = hidden
                attn_weights = self.compute_attention(h, projected_encoder_outputs)
            else:
                attn_weights = self.compute_attention(hidden, projected_encoder_outputs)

            # Compute context vector as weighted sum of projected encoder_outputs
            context = torch.bmm(attn_weights, projected_encoder_outputs)  # (N, 1, decoder_hidden_size)
            # Concatenate context with embedding: context (left) and embedded (right)
            concat = torch.cat((context, embedded), dim=2)  # (N, 1, decoder_hidden_size + emb_size)
            # Down-project concatenated vector to emb_size using attn_combine
            attn_combined = torch.tanh(self.attn_combine(concat.squeeze(1)))  # (N, emb_size)
            attn_combined = attn_combined.unsqueeze(1)  # (N, 1, emb_size)

            if self.model_type == "LSTM":
                output, (h_out, c_out) = self.rnn(attn_combined, hidden)
                hidden = (h_out, c_out)
            else:
                output, hidden = self.rnn(attn_combined, hidden)
        else:
            if self.model_type == "LSTM":
                output, (h_out, c_out) = self.rnn(embedded, hidden)
                hidden = (h_out, c_out)
            else:
                output, hidden = self.rnn(embedded, hidden)

        # 3) Squeeze output: (N, 1, decoder_hidden_size) -> (N, decoder_hidden_size)
        output = output.squeeze(1)
        # 4) Final linear layer and log softmax
        output = self.fc_out(output)
        output = self.log_softmax(output)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden
