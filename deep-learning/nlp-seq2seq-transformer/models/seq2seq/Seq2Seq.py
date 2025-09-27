""" 			  
Seq2Seq model.  (c) 2021 Georgia Tech

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


class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the Seq2Seq model. You should use .to(device) to ensure     #
        #    the models are on the same device (CPU/GPU). This should take no more  #
        #    than 2 lines of code.                                                  #
        #############################################################################
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, source):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
            Returns:
                outputs (tensor): shape (batch_size, seq_len, decoder_output_size)
        """

        batch_size = source.shape[0]
        seq_len = source.shape[1]
        decoder_output_size = self.decoder.output_size

        #############################################################################
        # 1) Pass source into encoder to get (encoder_outputs, hidden)
        #############################################################################
        encoder_outputs, hidden = self.encoder(source)

        # If using an RNN (not LSTM) and hidden is 2D, unsqueeze to add the layer dimension
        if self.decoder.model_type == "RNN" and hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)  # Now hidden is (1, batch_size, hidden_size)

        #############################################################################
        # 2) The first input for the decoder is the <sos> token, assumed to be source[:, 0]
        #############################################################################
        input_token = source[:, 0].unsqueeze(1)  # shape (batch_size, 1)

        outputs = torch.zeros(batch_size, seq_len, decoder_output_size, device=self.device)

        #############################################################################
        # 3) Decode step by step
        #############################################################################
        for t in range(seq_len):
            if self.decoder.attention:
                output, hidden = self.decoder(input_token, hidden, encoder_outputs=encoder_outputs)
            else:
                output, hidden = self.decoder(input_token, hidden)

            outputs[:, t, :] = output

            top1 = output.argmax(dim=1)  # shape (batch_size,)
            input_token = top1.unsqueeze(1)  # shape (batch_size, 1)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs