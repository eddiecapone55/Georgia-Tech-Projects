"""
Transformer model.  (c) 2021 Georgia Tech

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
from torch import nn
import random

##############################################################################
# If you want reproducible initialization, call seed_torch(0) once externally
##############################################################################
def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    produces a final output (e.g. probabilities for each token).
    """

    def __init__(self, input_size, output_size, device,
                 hidden_dim=128, num_heads=2, dim_feedforward=2048,
                 dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input vocabulary
        :param output_size: the size of the output vocabulary
        :param hidden_dim: embedding dimension
        :param num_heads: number of heads
        :param dim_feedforward: dimension of feed-forward network
        :param dim_k, dim_v, dim_q: dims for K, V, Q (to be used as provided)
        :param max_length: maximum sequence length
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        # Use the provided dimensions for key, value, and query (do not override)
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q

        ##############################################################################
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Don’t worry about sine/cosine encodings- use positional embeddings.        #
        ##############################################################################
        self.embeddingL = nn.Embedding(self.input_size, self.word_embedding_dim)
        self.posembeddingL = nn.Embedding(self.max_length, self.word_embedding_dim)
        # Initialize position embeddings in a standard way
        nn.init.normal_(self.posembeddingL.weight, mean=0, std=0.02)

        ##############################################################################
        # Deliverable 2: multi-head self-attention layers (already given)            #
        ##############################################################################
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)

        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)

        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        ##############################################################################
        # Deliverable 3: feed-forward layer + layer norm                             #
        ##############################################################################
        self.ff1 = nn.Linear(self.hidden_dim, self.dim_feedforward)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(self.dim_feedforward, self.hidden_dim)
        self.norm_ff = nn.LayerNorm(self.hidden_dim)

        ##############################################################################
        # Deliverable 4: final layer for classification/translation logits           #
        ##############################################################################
        self.output_layer = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, inputs):
        """
        Full forward pass of the single-layer Transformer
        :param inputs: shape (N,T), integer token lookups
        :returns: shape (N,T,output_size)
        """
        # 1) embed
        embeds = self.embed(inputs)            # (N,T,H)
        # 2) multi-head attention
        attn_out = self.multi_head_attention(embeds)   # (N,T,H)
        # 3) feed-forward
        ff_out = self.feedforward_layer(attn_out)      # (N,T,H)
        # 4) final layer
        outputs = self.final_layer(ff_out)             # (N,T,output_size)
        return outputs

    def embed(self, inputs):
        """
        :param inputs: shape (N,T)
        :returns embeddings: shape (N,T,H)
        """
        N, T = inputs.shape
        # Word embeddings (no scaling)
        word_embeds = self.embeddingL(inputs)  # (N,T,H)
        # Positional embeddings: positions = [0..T-1]
        positions = torch.arange(T, device=inputs.device).unsqueeze(0).expand(N, T)
        pos_embeds = self.posembeddingL(positions)  # (N,T,H)

        embeddings = word_embeds + pos_embeds
        return embeddings

    def multi_head_attention(self, inputs):
        """
        :param inputs: shape (N,T,H)
        :returns: shape (N,T,H)
        """
        N, T, H = inputs.shape

        # Head1: compute K1, Q1, V1
        K1 = self.k1(inputs)   # (N,T,dim_k)
        Q1 = self.q1(inputs)   # (N,T,dim_q)
        V1 = self.v1(inputs)   # (N,T,dim_v)

        # Head2: compute K2, Q2, V2
        K2 = self.k2(inputs)   # (N,T,dim_k)
        Q2 = self.q2(inputs)   # (N,T,dim_q)
        V2 = self.v2(inputs)   # (N,T,dim_v)

        # scaled dot product attention for each head
        scores1 = torch.bmm(Q1, K1.transpose(1,2)) / (self.dim_k ** 0.5)  # (N,T,T)
        attn_weights1 = self.softmax(scores1)
        head1 = torch.bmm(attn_weights1, V1)  # (N,T,dim_v)

        scores2 = torch.bmm(Q2, K2.transpose(1,2)) / (self.dim_k ** 0.5)
        attn_weights2 = self.softmax(scores2)
        head2 = torch.bmm(attn_weights2, V2)

        # concat heads
        concat = torch.cat([head1, head2], dim=2)  # (N,T, dim_v*num_heads)
        # project
        proj = self.attention_head_projection(concat)  # (N,T,H)

        # add + norm
        out = self.norm_mh(inputs + proj)  # (N,T,H)
        return out

    def feedforward_layer(self, inputs):
        """
        :param inputs: shape (N,T,H)
        :returns: shape (N,T,H)
        """
        ff = self.ff1(inputs)    # (N,T,dim_feedforward)
        ff = self.relu(ff)
        ff = self.ff2(ff)        # (N,T,H)
        # add + norm
        out = self.norm_ff(inputs + ff)  # (N,T,H)
        return out

    def final_layer(self, inputs):
        """
        :param inputs: shape (N,T,H)
        :returns: shape (N,T,output_size)
        """
        return self.output_layer(inputs)


class FullTransformerTranslator(nn.Module):
    """
    A full encoder-decoder Transformer using PyTorch's nn.Transformer
    (optional portion).
    """
    def __init__(self, input_size, output_size, device,
                 hidden_dim=128, num_heads=2, dim_feedforward=2048,
                 num_layers_enc=2, num_layers_dec=2, dropout=0.2,
                 max_length=43, ignore_index=1):
        super(FullTransformerTranslator, self).__init__()

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.pad_idx = ignore_index  # <pad> token index

        ##############################################################################
        # Deliverable 1: Initialize nn.Transformer for the encoder/decoder           #
        ##############################################################################
        self.transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            num_encoder_layers=num_layers_enc,
            num_decoder_layers=num_layers_dec,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        ##############################################################################
        # Deliverable 2: embeddings for src and tgt, plus positional embeddings      #
        ##############################################################################
        # Per the discussion: srcembedding => input_size, tgtembedding => output_size
        self.srcembeddingL = nn.Embedding(self.input_size, self.hidden_dim)
        self.tgtembeddingL = nn.Embedding(self.output_size, self.hidden_dim)
        self.srcposembeddingL = nn.Embedding(self.max_length, self.hidden_dim)
        self.tgtposembeddingL = nn.Embedding(self.max_length, self.hidden_dim)

        ##############################################################################
        # Deliverable 3: final layer (projection to output vocab)                    #
        ##############################################################################
        self.fc_out = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, src, tgt):
        """
        This function computes the full Transformer forward pass used during training.
        It internally shifts the target sequence so that the decoder receives an input 
        starting with the <sos> token and without the final token.

        :param src: shape (N, T)
        :param tgt: shape (N, T)  (target sequence, not shifted)
        :returns: shape (N, T, output_size)
        """
        N, T = src.shape

        # 1) Shift target to the right by 1:
        #    Prepend the start token (<sos> = index 2) and remove the last token.
        sos_token = 2
        tgt_shifted = torch.cat([
            torch.full((N, 1), sos_token, device=src.device, dtype=tgt.dtype),
            tgt[:, :-1]
        ], dim=1)  # shape (N, T)

        # 2) Embed source with positional embeddings
        positions_src = torch.arange(T, device=src.device).unsqueeze(0).expand(N, T)
        src_emb = self.srcembeddingL(src) + self.srcposembeddingL(positions_src)

        # 3) Embed target (shifted) with positional embeddings
        positions_tgt = torch.arange(T, device=src.device).unsqueeze(0).expand(N, T)
        tgt_emb = self.tgtembeddingL(tgt_shifted) + self.tgtposembeddingL(positions_tgt)

        # 4) Create a causal mask for tgt to avoid looking into the future
        tgt_mask = self.transformer.generate_square_subsequent_mask(T).to(src.device)

        # 5) Create target key padding mask where <pad> is True
        tgt_key_padding_mask = (tgt_shifted == self.pad_idx)
        # Optionally, create source key padding mask if needed
        src_key_padding_mask = (src == self.pad_idx)

        # 6) Pass through the transformer
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_key_padding_mask=src_key_padding_mask
        )  # shape (N, T, H)

        # 7) Project to the output vocabulary
        out = self.fc_out(out)  # shape (N, T, output_size)
        return out

    def generate_translation(self, src):
        """
        Generate the output of the transformer taking src as input in an autoregressive manner.
        This version uses TWO <sos> tokens at the start of the target, 
        as described in the discussion thread.

        :param src: shape (N,T)
        :returns: shape (N,T,output_size)
        """
        N, T = src.shape
        # We'll store logits for each time step
        outputs = torch.zeros(N, T, self.output_size, device=self.device)

        # Create a "running" target full of pad tokens
        tgt_tokens = torch.full((N, T), self.pad_idx, dtype=torch.long, device=src.device)

        # We'll slice the first token from src to use as the <sos> token,
        # or simply set it to 2 if your <sos> is always 2.
        # The thread mentions "the function extracts the SOS token by slicing the first token
        # of each source sequence and sets the first TWO positions of tgt to it."
        # We'll do that approach:
        sos_from_src = src[:, 0]  # shape (N,)
        # set the first two tokens of each sequence to <sos> (or sos_from_src)
        # For example, if your <sos> is always index 2, you can do:
        #   tgt_tokens[:, 0] = 2
        #   tgt_tokens[:, 1] = 2
        # We'll do what's in the thread literally:
        tgt_tokens[:, 0] = sos_from_src
        if T > 1:
            tgt_tokens[:, 1] = sos_from_src

        # Precompute the source embeddings (encoder sees full source)
        pos_src = torch.arange(T, device=src.device).unsqueeze(0).expand(N, T)
        src_emb = self.srcembeddingL(src) + self.srcposembeddingL(pos_src)

        # The thread's logic: loop for seq_len-1 steps (since positions 0 and 1 are <sos>)
        # or possibly loop T times. 
        # We'll do seq_len times but only update up to T-1, as the comment says.
        for t in range(T):
            # We'll skip updating the first two tokens, 
            # so effectively we only do real generation from t=0..T-2 
            # because positions 0 and 1 are already <sos>.
            if t >= T - 1:
                # No more tokens to predict
                break

            # forward pass
            # embed the current target
            pos_tgt_len = t + 1  # number of tokens "filled" so far
            if pos_tgt_len < 2:
                pos_tgt_len = 2  # ensure at least 2 positions if T>1

            cur_tgt = tgt_tokens[:, :pos_tgt_len]
            pos_tgt = torch.arange(pos_tgt_len, device=src.device).unsqueeze(0).expand(N, pos_tgt_len)
            tgt_emb = self.tgtembeddingL(cur_tgt) + self.tgtposembeddingL(pos_tgt)

            # create causal mask for current length
            submask = self.transformer.generate_square_subsequent_mask(pos_tgt_len).to(src.device)
            # build target key padding mask
            tgt_kp_mask = (cur_tgt == self.pad_idx)

            # run the transformer
            out = self.transformer(
                src=src_emb,
                tgt=tgt_emb,
                tgt_mask=submask,
                tgt_key_padding_mask=tgt_kp_mask
            )  # shape (N, pos_tgt_len, H)
            out = self.fc_out(out)  # shape (N, pos_tgt_len, output_size)

            # get the logits for the "current" position t 
            # (the thread says "it extracts the logits for position t, because positions 0 & 1 are fixed")
            # If t < 2, that might be weird, but let's do what the thread says literally:
            # "It updates the target tensor at position t+1 with the predicted token"
            # "It updates the outputs at position t with the current logits"
            # So next_token_logits = out[:, t, :]
            # But we only have pos_tgt_len positions in out. If t >= pos_tgt_len, it fails.
            # We'll clamp t to pos_tgt_len-1 if needed:
            clamp_t = min(t, pos_tgt_len - 1)
            next_token_logits = out[:, clamp_t, :]  # shape (N, output_size)

            # store the logits at position t in outputs
            outputs[:, t, :] = next_token_logits
            # pick the next token
            next_token = torch.argmax(next_token_logits, dim=-1)  # shape (N,)

            # "update the target at position t+1 with the predicted token"
            if (t + 1) < T:
                tgt_tokens[:, t + 1] = next_token

        return outputs
