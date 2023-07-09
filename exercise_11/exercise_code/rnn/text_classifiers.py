import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .rnn_nn import Embedding, RNN, LSTM


class RNNClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'use_lstm': use_lstm,
            **additional_kwargs
        }
        # if you do not inherit from lightning module use the following line
        self.hparams = hparams
        
        # if you inherit from lightning module, comment out the previous line and use the following line
        # self.hparams.update(hparams)
        
        ########################################################################
        # TODO: Initialize an RNN network for sentiment classification         #
        # hint: A basic architecture can have an embedding, an rnn             #
        # and an output layer                                                  #
        ########################################################################
        

        self.embedding = Embedding(num_embeddings, embedding_dim, padding_idx=0)
        if use_lstm:
            # self.rnn = LSTM(embedding_dim, hidden_size)
            self.rnn = nn.LSTM(embedding_dim, hidden_size, hparams['n_layers'], bidirectional=hparams['bidirectional'],
                            dropout=hparams['dropout_rate'], batch_first=False)
        else:
            self.rnn = RNN(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, hparams['output_dim'])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, sequence, lengths=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        output = None

        ########################################################################
        # TODO: Apply the forward pass of your network                         #
        # hint: Don't forget to use pack_padded_sequence if lenghts is not None#
        # pack_padded_sequence should be applied to the embedding outputs      #
        ########################################################################

        embedded = self.embedding(sequence)

        if lengths is not None:
            # If sequence lengths are provided, we pack the sequence before passing it to the RNN
            lengths = lengths.view(-1).tolist()
            embedded = pack_padded_sequence(embedded, lengths)

        # RNN outputs
        if self.hparams['use_lstm']:
            rnn_out, (hidden, _) = self.rnn(embedded)
        else:
            rnn_out, hidden = self.rnn(embedded)

        if lengths is not None:
            # Unpack the output sequence if packed
            rnn_out, _ = pad_packed_sequence(rnn_out)

        output = self.fc(hidden.squeeze(0))
        output = torch.sigmoid(output)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return output.squeeze()
