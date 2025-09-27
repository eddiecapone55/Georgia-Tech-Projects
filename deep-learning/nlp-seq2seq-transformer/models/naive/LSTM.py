"""
LSTM model.  (c) 2021 Georgia Tech

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

-----do not edit anything above this line---
"""

import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # You will need to complete the class init function, and forward function

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns:
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes in order specified below to pass GS.   #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate (tilde cell)                                         #
        #       4) o_t: output gate                                                    #
        #   for each equation above, initialize the weights,biases for input prior     #
        #   to weights, biases for hidden.                                             #
        #   when initializing the weights consider that in forward method you          #
        #   should NOT transpose the weights.                                          #
        #   You also need to include correct activation functions                      #
        ################################################################################

        # i_t: input gate
        self.Wii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.bii = nn.Parameter(torch.Tensor(hidden_size))
        self.Whi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bhi = nn.Parameter(torch.Tensor(hidden_size))

        # f_t: forget gate
        self.Wif = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.bif = nn.Parameter(torch.Tensor(hidden_size))
        self.Whf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bhf = nn.Parameter(torch.Tensor(hidden_size))

        # g_t: the cell gate
        self.Wig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.big = nn.Parameter(torch.Tensor(hidden_size))
        self.Whg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bhg = nn.Parameter(torch.Tensor(hidden_size))

        # o_t: the output gate
        self.Wio = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.bio = nn.Parameter(torch.Tensor(hidden_size))
        self.Who = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bho = nn.Parameter(torch.Tensor(hidden_size))

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        self.init_hidden()

    def init_hidden(self):
        # Initialize parameters (e.g., using xavier for weights, zeros for biases)
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_size, seq_len, _ = x.shape

        # Initialize h_t, c_t to zeros
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of the jupyter notebook. Iterate through all time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              #
        #   h_t and c_t should be updated at every time step.                          #
        ################################################################################
        for t in range(seq_len):
            x_t = x[:, t, :]  # shape (batch, input_size)

            i_t = torch.sigmoid(x_t @ self.Wii + self.bii + h_t @ self.Whi + self.bhi)
            f_t = torch.sigmoid(x_t @ self.Wif + self.bif + h_t @ self.Whf + self.bhf)
            g_t = torch.tanh(x_t @ self.Wig + self.big + h_t @ self.Whg + self.bhg)
            o_t = torch.sigmoid(x_t @ self.Wio + self.bio + h_t @ self.Who + self.bho)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)