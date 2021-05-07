# Copyright (c) 2020-present, pascalnotsawo@gmail.com.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext import data, datasets
from transformers import BertTokenizer
import spacy
import time
import random
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nlp = spacy.load('en')
   
class RNN(nn.Module):
    """
    RNN class
    This model have three layers : embedding layer, RNN and linear layer. 
    All layers have their parameters initialized to random values, unless explicitly specified.
    """
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        """
        create a new instance of RNN class
        input_dim (int) : dimension of the one-hot vectors, which is equal to the vocabulary size
        embedding_dim (int) : size of the dense word vectors
        hidden_dim (int) : size of the hidden states
        output_dim (int) : usually the number of classes, however in the case of only 2 classes 
                           the output value is between 0 and 1 and thus can be 1-dimensional, 
                           i.e. a single scalar real number.
        """
        
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):

        #text = [sent len, batch size]
        
        embedded = self.embedding(text)
        #embedded = [sent len, batch size, emb dim]
        
        output, hidden = self.rnn(embedded)
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        return self.fc(hidden.squeeze(0))
    
    def getID(self):
        """To save the best model"""
        return 'RNN'
    
class LSTM(nn.Module):
    """LSTM class : embedding layer, LSTM, and a linear layer."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        """
        vocab_size (int) : vocabulary size
        embedding_dim (int) : size of the dense word vectors
        hidden_dim (int) : size of the hidden states
        output_dim (int) : usually the number of classes, however in the case of only 2 classes 
                           the output value is between 0 and 1 and thus can be 1-dimensional, 
                           i.e. a single scalar real number.
        n_layers (int) : Multi-layer RNNs (also called deep RNNs) are another simple concept. 
                         The idea is that we add additional RNNs on top of the initial standard RNN, 
                         where each RNN added is another *layer*
        bidirectional (bool) : The concept behind a bidirectional RNN is simple. 
                              As well as having an RNN processing the words in the sentence from the 
                              first to the last (a forward RNN), we have a second RNN processing the 
                              words in the sentence from the last to the first (a backward RNN)
        dropout (int) : we use a method of regularization called dropout. 
                        Dropout works by randomly dropping out (setting to 0) neurons in a layer during 
                        a forward pass
        pad_idx (int) : index of <pad> token in th vocabulary
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_percent = dropout
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional=bidirectional, dropout = dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)) 
        #hidden = [batch size, hid dim * num directions]
            
        return self.fc(hidden)
    
    def getID(self):
        """To save the best model"""
        return 'LSTM'
    
class CNN(nn.Module):
    """
    Currently the CNN model can only use 3 different sized filters, but we can actually improve 
    the code of our model to make it more generic and take any number of filters.
    We do this by placing all of our convolutional layers in a nn.ModuleList, a function used to 
    hold a list of PyTorch nn.Modules. If we simply used a standard Python list, the modules within 
    the list cannot be "seen" by any modules outside the list which will cause us some errors.

    We can now pass an arbitrary sized list of filter sizes and the list comprehension will create 
    a convolutional layer for each of them. 
    Then, in the forward method we iterate through the list applying each convolutional layer to 
    get a list of convolutional outputs, which we also feed through the max pooling in a list 
    comprehension before concatenating together and passing through the dropout and linear layers.
    """
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,  dropout, pad_idx):
        """
        vocab_size (int) : vocabulary size
        embedding_dim (int) : size of the dense word vectors
        n_filters (int) : number of filter
        filter_sizes (list) : size of the filters or kernel, is going to be [n x emb_dim] where n is the size of the n-grams.
        output_dim (int) : usually the number of classes, however in the case of only 2 classes 
                           the output value is between 0 and 1 and thus can be 1-dimensional, 
                           i.e. a single scalar real number.
        n_layers (int) : number of layers
        bidirectional (bool) : bidirectional or nor
        dropout (int) : we use a method of regularization called dropout. 
                        Dropout works by randomly dropping out (setting to 0) neurons in a layer during 
                        a forward pass
        pad_idx (int) : index of <pad> token in th vocabulary
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.output_dim = output_dim
        self.dropout_percent = dropout
        self.pad_idx = pad_idx 
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
                
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)
    
    def getID(self):
        """To save the best model"""
        return 'CNN'
    
"""
We can also implement the above model using 1-dimensional convolutional layers, 
where the embedding dimension is the "depth" of the  filter and the number of 
tokens in the sentence is the width.

We'll run our tests in this notebook using the 2-dimensional convolutional model, 
but leave the implementation for the 1-dimensional model below for anyone interested.
"""
class CNN1d(nn.Module):
    """
    CNN1d class
    """
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        """
        vocab_size (int) : vocabulary size
        embedding_dim (int) : size of the dense word vectors
        n_filters (int) : number of filter
        filter_sizes (list) : size of the filters or kernel, is going to be [n x emb_dim] where n is the size of the n-grams.
        output_dim (int) : usually the number of classes, however in the case of only 2 classes 
                           the output value is between 0 and 1 and thus can be 1-dimensional, 
                           i.e. a single scalar real number.
        n_layers (int) : numbers of layers
        bidirectional (int) : bidirectional or not
        dropout (int) : we use a method of regularization called dropout. 
                        Dropout works by randomly dropping out (setting to 0) neurons in a layer during 
                        a forward pass
        pad_idx (int) : index of <pad> token in th vocabulary
        """

        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.output_dim = output_dim
        self.dropout_percent = dropout
        self.pad_idx = pad_idx 
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = embedding_dim, out_channels = n_filters, kernel_size = fs)
                                    for fs in filter_sizes])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.permute(0, 2, 1)
        #embedded = [batch size, emb dim, sent len]
        
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)
    
    def getID(self):
        """To save the best model"""
        return 'CNN1d'
    
    
class BERTGRUSentiment(nn.Module):
    """
    Instead of using an embedding layer to get embeddings for our text, we'll be using the 
    pre-trained transformer model. These embeddings will then be fed into a GRU to produce 
    a prediction for the sentiment of the input sentence. We get the embedding dimension size 
    (called the `hidden_size`) from the transformer via its config attribute. The rest of the 
    initialization is standard.
    
    Within the forward pass, we wrap the transformer in a `no_grad` to ensure no gradients are 
    calculated over this part of the model. The transformer actually returns the embeddings for 
    the whole sequence as well as a pooled output. The [documentation]
    The rest of the forward pass is the standard implementation of a recurrent model, 
    where we take the hidden state over the final time-step, and pass it through a linear 
    layer to get our predictions.
    """
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        """
        bert : BERT pretrained model
        embedding_dim (int) : size of the dense word vectors
        output_dim (int) : usually the number of classes, however in the case of only 2 classes 
                           the output value is between 0 and 1 and thus can be 1-dimensional, 
                           i.e. a single scalar real number.
        n_layers (int) : number of layers
        dropout (int) : we use a method of regularization called dropout. 
                        Dropout works by randomly dropping out (setting to 0) neurons in a layer during 
                        a forward pass
        bidirectional (bool) : bidirectional or not
        """
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional, batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.bert(text)[0]
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])       
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        #output = [batch size, out dim]
        
        return output
    
    def getID(self):
        """To save the best model"""
        return 'BERT'

class Trainer():
    """
    This class allows to do these four things :
    - define the model
    - compilation, which will load the data, build the optimizer and the loss function, and update the model parameters if necessary.
    - training and testing the model. 
    - test the model on a few real-life cases.
    """
    def __init__(self, model, dump_path = ""):
        """
        create the new instance of trainer
        model (subclass of nn.Module) : model concerned
        dump_path (string, default="") : folder in which the model will be saved during and after training
        """
        assert any([isinstance(model, className) for className in [RNN, LSTM, CNN, CNN1d, BERTGRUSentiment]]), "Model type not supported"
        self.model = model
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        self.dump_path = dump_path
        
    def compile(self, 
        optimizer = "Adam", 
        criterion = "BCEWithLogitsLoss", 
        seed = 1234, 
        train_n_samples = 25000, 
        split_ratio = 0.8, 
        test_n_samples = 25000,
        batch_size = 64, 
        max_vocab_size = 25000
    ):
        """
        load the data, build the optimizer and the loss function, and update the model parameters if necessary.
        optimizer (torch.optim, default = Adam) : model optimizer (use to update the model parameters)
        criterion (function, default = nn.BCEWithLogitsLoss) : loss function 
        seed (int, default = 1234) : random seeds for reproducibility
        train_n_samples (int, default = 25000) : number of training examples to consider
        split_ratio (float between 0 and 1, default = 0.8) : ratio of training data to use for training, 
                                                             the rest for validation
        test_n_samples (int, default = 25000) : number of test examples to consider
        batch_size (int, default = 64) : number of examples per batch
        max_vocab_size (int, default = 25000) : maximun token in the vocabulary
        """
        assert optimizer in ["SGD", "Adam"], "optimizer type not supported"
        assert criterion in ["BCEWithLogitsLoss"], "criterion type not supported"
        assert train_n_samples > 0
        assert test_n_samples > 0

        if train_n_samples > 25000 :
            train_n_samples = 25000
        train_ratio = train_n_samples/25000

        if test_n_samples > 25000 :
            test_n_samples = 25000
        test_ratio = test_n_samples/25000
        
        """
        Instead of writing a parameterizable function to process the data, we preferred to detail 
        the process to better justify each approach.
        """

        ## RNN
        if isinstance(self.model, RNN) :
            
            # set the random seed
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

            # use spacy as tokenizer
            TEXT = data.Field(tokenize = 'spacy')
            LABEL = data.LabelField(dtype = torch.float)

            """
            The following code automatically downloads the IMDb dataset and splits it into 
            the canonical train/test splits as `torchtext.datasets` objects. 
            It process the data using the `Fields` we have previously defined. 
            The IMDb dataset consists of 50,000 movie reviews, each marked as being a positive 
            or negative review.
            """
            train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
            if 1 > test_ratio :   
                test_data,_ = test_data.split(
                                                random_state = seed, 
                                                split_ratio = test_ratio
                                            ) 
            if 1 > train_ratio :
                train_data,_ = train_data.split(
                                                random_state = seed, 
                                                split_ratio = train_ratio
                                            )   
            """
            The IMDb dataset only has train/test splits, so we need to create a validation set. 
            We can do this with the `.split()` method. By default this splits 70/30, however by 
            passing a split_ratio argument, we can change the ratio of the split, i.e. a
            split_ratio of 0.8 would mean 80% of the examples make up the training set and 20% 
            make up the validation set. We also pass our random seed to the random_state argument, 
            ensuring that we get the same train/validation split each time.
            """
            train_data, valid_data = train_data.split(
                                                    random_state = seed, 
                                                    split_ratio = split_ratio
                                                )
            # We can see how many examples are in each split by checking their length.
            print(f'Number of training examples: {len(train_data)}')
            print(f'Number of validation examples: {len(valid_data)}')
            print(f'Number of testing examples: {len(test_data)}')
            
            """
            Next, we have to build a vocabulary. This is a effectively a look up table where 
            every unique word in your data set has a corresponding index (an integer).
            We do this as our machine learning model cannot operate on strings, only numbers. 
            Each index is used to construct a one-hot vector for each word. 
            A one-hot vector is a vector where all of the elements are 0, except one, which is 1, 
            and dimensionality is the total number of unique words in your vocabulary.
            The number of unique words in our training set is over 100,000, which means that our 
            one-hot vectors will have over 100,000 dimensions! This will make training slow and 
            possibly won't fit onto your GPU (if you're using one). 
            There are two ways effectively cut down our vocabulary, we can either only take the 
            top n most common words or ignore words that appear less than m times. 
            We'll do the former, only keeping the top max_vocab_size words.

            What do we do with words that appear in examples but we have cut from the vocabulary? 
            We replace them with a special unknown or <unk> token. 
            For example, if the sentence was "This film is great and I love it" but the word 
            "love" was not in the vocabulary, it would become "This film is great and I <unk> it".
            The following builds the vocabulary, only keeping the most common max_vocab_size tokens.
            """
            TEXT.build_vocab(train_data, max_size = max_vocab_size)
            LABEL.build_vocab(train_data)
            print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
            print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
            """
            Why do we only build the vocabulary on the training set? 
            When testing any machine learning system you do not want to look at the
            test set in any way. We do not include the validation set as we want it to 
            reflect the test set as much as possible.
            """

            """
            The final step of preparing the data is creating the iterators. 
            We iterate over these in the training/evaluation loop, and they return a batch of 
            examples (indexed and converted into tensors) at each iteration.

            We'll use a `BucketIterator` which is a special type of iterator that will return a batch 
            of examples where each example is of a similar length, minimizing the amount of padding 
            per example.
            
            We also want to place the tensors returned by the iterator on the GPU (if you're using one). 
            PyTorch handles this using `torch.device`, we then pass this device to the iterator.
            """
            train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                (train_data, valid_data, test_data), 
                batch_size = batch_size, device = device
            )

            
            self.dataset = {"TEXT" : TEXT, "LABEL" : LABEL,
                            "train_data" : train_data, "valid_data" : valid_data, "test_data" : test_data,
                            "train_iterator" : train_iterator, "valid_iterator" : valid_iterator, "test_iterator" : test_iterator}
                     
            """
            We now update the input dimention of the RNN model : the input dimension is the 
            dimension of the one-hot vectors, which is equal to the vocabulary size.
            """
            self.model = RNN(
                input_dim = len(self.dataset["TEXT"].vocab), 
                embedding_dim = self.model.embedding_dim, 
                hidden_dim = self.model.hidden_dim, 
                output_dim = self.model.output_dim
            )
         
        ## LSTM   
        elif isinstance(self.model, LSTM) :
          
            # Previous comments are valid here.
            
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            
            """
            As above, we'll set the seed, define the Fields and get the train/valid/test splits.

            We'll be using packed padded sequences, which will make our LSTM only process the 
            non-padded elements of our sequence, and for any padded element the output will be a 
            zero tensor. To use packed padded sequences, we have to tell the LSTM how long the 
            actual sequences are. We do this by setting `include_lengths = True` for our TEXT 
            field. This will cause batch.text to now be a tuple with the first element being 
            our sentence (a numericalized tensor that has been padded) and the second element 
            being the actual lengths of our sentences.
            """
            TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
            LABEL = data.LabelField(dtype = torch.float)
            
            # We then load the IMDb dataset.
            train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
            if 1 > test_ratio :   
                test_data,_ = test_data.split(
                                                random_state = seed, 
                                                split_ratio = test_ratio
                                            )
            if 1 > train_ratio :
                train_data,_ = train_data.split(
                                                random_state = seed, 
                                                split_ratio = train_ratio
                                            )
            train_data, valid_data = train_data.split(
                                                    random_state = seed, 
                                                    split_ratio = split_ratio
                                                )
            print(f'Number of training examples: {len(train_data)}')
            print(f'Number of validation examples: {len(valid_data)}')
            print(f'Number of testing examples: {len(test_data)}')

            """
            Next is the use of pre-trained word embeddings. Now, instead of having our word 
            embeddings initialized randomly, they are initialized with these pre-trained vectors.
            We get these vectors simply by specifying which vectors we want and passing it as an 
            argument to build_vocab. TorchText handles downloading the vectors and associating them 
            with the correct words in our vocabulary.

            Here, we'll be using the glove.6B.100d vectors. glove (https://nlp.stanford.edu/projects/glove/) 
            is the algorithm used to calculate the vectors. 6B indicates these vectors were trained on 6 
            billion tokens and 100d indicates these vectors are 100-dimensional.

            You can see the other available vectors here : https://github.com/pytorch/text/blob/master/torchtext/vocab.py#L113.

            The theory is that these pre-trained vectors already have words with similar semantic meaning close together in vector space, 
            e.g. "terrible", "awful", "dreadful" are nearby. This gives our embedding layer a good initialization as it does not have to 
            learn these relations from scratch.

            These vectors are about 862MB, so watch out if you have a limited internet connection.

            By default, TorchText will initialize words in your vocabulary but not in your pre-trained 
            embeddings to zero. We don't want this, and instead initialize them randomly by setting unk_init to 
            torch.Tensor.normal_. This will now initialize those words via a Gaussian distribution.
            """
            TEXT.build_vocab(train_data, max_size = max_vocab_size, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
            LABEL.build_vocab(train_data)
            print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
            print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
    
            """
            Another thing for packed padded sequences all of the tensors within a batch need to be 
            sorted by their lengths. This is handled in the iterator by setting `sort_within_batch = True`.
            """
            train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                (train_data, valid_data, test_data), 
                batch_size = batch_size, sort_within_batch = True, device = device
            )

            self.dataset = {"TEXT" : TEXT, "LABEL" : LABEL,
                            "train_data" : train_data, "valid_data" : valid_data, "test_data" : test_data,
                            "train_iterator" : train_iterator, "valid_iterator" : valid_iterator, "test_iterator" : test_iterator}
            
            """
            We now update the input dimention of the LSTM model : the input dimension is the 
            dimension of the one-hot vectors, which is equal to the vocabulary size.
            The pad index comes from vocabulary
            """
            self.model = LSTM(
                vocab_size = len(self.dataset["TEXT"].vocab), 
                embedding_dim = self.model.embedding_dim, 
                hidden_dim = self.model.hidden_dim, 
                output_dim = self.model.output_dim, 
                n_layers = self.model.n_layers, 
                bidirectional = self.model.bidirectional, 
                dropout = self.model.dropout_percent, 
                pad_idx = self.dataset["TEXT"].vocab.stoi[self.dataset["TEXT"].pad_token] 
            )
            
            
            """
            The final addition is copying the pre-trained word embeddings we loaded earlier into the 
            embedding layer of our model. We retrieve the embeddings from the field's vocab, and 
            check they're the correct size, [vocab size, embedding dim]
            """
            pretrained_embeddings = self.dataset["TEXT"].vocab.vectors

            """
            We then replace the initial weights of the embedding layer with the pre-trained embeddings.
            This should always be done on the weight.data and not the weight !
            """
            self.model.embedding.weight.data.copy_(pretrained_embeddings)

            """
            As our <unk> and <pad> token aren't in the pre-trained vocabulary they have been 
            initialized using unk_init (an N(0,1) distribution) when building our vocab. 
            It is preferable to initialize them both to all zeros to explicitly tell our model that, 
            initially, they are irrelevant for determining sentiment. 

            We do this by manually setting their row in the embedding weights matrix to zeros. 
            We get their row by finding the index of the tokens, which we have already done for the padding index.

            Like initializing the embeddings, this should be done on the weight.data and not the weight!
            """
            UNK_IDX = self.dataset["TEXT"].vocab.stoi[self.dataset["TEXT"].unk_token]
            self.model.embedding.weight.data[UNK_IDX] = torch.zeros(self.model.embedding_dim)
            self.model.embedding.weight.data[self.model.pad_idx] = torch.zeros(self.model.embedding_dim)
               
        ## CNN/CNN1d
        elif isinstance(self.model, CNN) or isinstance(self.model, CNN1d) :
            
            # Previous comments are valid here.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            
            """ 
            Unlike the above with the FastText model, we no longer explicitly need to create the 
            bi-grams and append them to the end of the sentence.
            As convolutional layers expect the batch dimension to be first we can tell TorchText 
            to return the data already permuted using the `batch_first = True` argument on the field.
            """
            TEXT = data.Field(tokenize = 'spacy', batch_first = True)
            LABEL = data.LabelField(dtype = torch.float)

            train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
            if 1 > test_ratio :   
                test_data,_ = test_data.split(
                                                random_state = seed, 
                                                split_ratio = test_ratio
                                            )
            if 1 > train_ratio :
                train_data,_ = train_data.split(
                                                random_state = seed, 
                                                split_ratio = train_ratio
                                            )
            train_data, valid_data = train_data.split(
                                                    random_state = seed, 
                                                    split_ratio = split_ratio
                                                )
            print(f'Number of training examples: {len(train_data)}')
            print(f'Number of validation examples: {len(valid_data)}')
            print(f'Number of testing examples: {len(test_data)}')

            
            # Build the vocab and load the pre-trained word embeddings.
            TEXT.build_vocab(train_data, max_size = max_vocab_size, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
            LABEL.build_vocab(train_data)
            print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
            print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
            
            # As above, we create the iterators.
            train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                (train_data, valid_data, test_data), 
                batch_size = batch_size, device = device
            )

            self.dataset = {"TEXT" : TEXT, "LABEL" : LABEL,
                            "train_data" : train_data, "valid_data" : valid_data, "test_data" : test_data,
                            "train_iterator" : train_iterator, "valid_iterator" : valid_iterator, "test_iterator" : test_iterator}
    
            """
            We now update the input dimention of the CNN/CNN1d model : the input dimension is the 
            dimension of the one-hot vectors, which is equal to the vocabulary size.
            The pad index comes from vocabulary
            """
            if isinstance(self.model, CNN) :
                self.model = CNN(
                    vocab_size = len(self.dataset["TEXT"].vocab), 
                    embedding_dim = self.model.embedding_dim, 
                    n_filters =  self.model.n_filters, 
                    filter_sizes = self.model.filter_sizes, 
                    output_dim = self.model.output_dim, 
                    dropout = self.model.dropout_percent, 
                    pad_idx = self.dataset["TEXT"].vocab.stoi[self.dataset["TEXT"].pad_token]
                )
                
            else :
                self.model = CNN1d(
                    vocab_size = len(self.dataset["TEXT"].vocab), 
                    embedding_dim = self.model.embedding_dim, 
                    n_filters =  self.model.n_filters, 
                    filter_sizes = self.model.filter_sizes, 
                    output_dim = self.model.output_dim, 
                    dropout = self.model.dropout_percent, 
                    pad_idx = self.dataset["TEXT"].vocab.stoi[self.dataset["TEXT"].pad_token]
                )
            
            # load the pre-trained embeddings
            pretrained_embeddings = self.dataset["TEXT"].vocab.vectors
            self.model.embedding.weight.data.copy_(pretrained_embeddings)
            
            # Then zero the initial weights of the unknown and padding tokens.
            UNK_IDX = self.dataset["TEXT"].vocab.stoi[self.dataset["TEXT"].unk_token]
            self.model.embedding.weight.data[UNK_IDX] = torch.zeros(self.model.embedding_dim)
            self.model.embedding.weight.data[self.model.pad_idx] = torch.zeros(self.model.embedding_dim)
            
        ## BERT
        elif isinstance(self.model, BERTGRUSentiment) :

            # Previous comments are valid here.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            
            """
            The transformer has already been trained with a specific vocabulary, which means we 
            need to train with the exact same vocabulary and also tokenize our data in the same 
            way that the transformer did when it was initially trained.

            Luckily, the transformers library has tokenizers for each of the transformer models 
            provided. In this case we are using the BERT model which ignores casing (i.e. will 
            lower case every word). We get this by loading the pre-trained bert-base-uncased
            tokenizer.
            """
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.tokenizer = tokenizer
            
            """
            Another thing we need to handle is that the model was trained on sequences with a 
            defined maximum length - it does not know how to handle sequences longer than it has 
            been trained on. We can get the maximum length of these input sizes by checking the 
            max_model_input_sizes for the version of the transformer we want to use. In this case, 
            it is 512 tokens.
            """
            max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
            self.max_input_length = max_input_length
            
            """
            Previously we have used the spaCy tokenizer to tokenize our examples. 
            However we now need to define a function that we will pass to our TEXT field that will
            handle all the tokenization for us. It will also cut down the number of tokens to a 
            maximum length. Note that our maximum length is 2 less than the actual maximum length. 
            This is because we need to append two tokens to each sequence, one to the start and one
            to the end.
            """
            def tokenize_and_cut(sentence):
                tokens = tokenizer.tokenize(sentence) 
                tokens = tokens[:max_input_length-2]
                return tokens
                    
            """
            The transformer was also trained with special tokens to mark the beginning and end of 
            the sentence. As well as a standard padding and unknown token. We can also get these 
            from the tokenizer. The tokenizer does have a beginning of sequence and end of sequence 
            attributes (bos_token and eos_token) but these are not set and should not be used for 
            this transformer.
            """
            init_token_idx = tokenizer.cls_token_id
            eos_token_idx = tokenizer.sep_token_id
            pad_token_idx = tokenizer.pad_token_id
            unk_token_idx = tokenizer.unk_token_id

            self.init_token_idx = init_token_idx
            self.eos_token_idx = eos_token_idx
            self.pad_token_idx = pad_token_idx
            self.unk_token_idx = unk_token_idx 
            
            """
            Now we define our fields. The transformer expects the batch dimension to be first, 
            so we set `batch_first = True`. As we already have the vocabulary for our text, 
            provided by the transformer we set `use_vocab = False` to tell torchtext that we'll be 
            handling the vocabulary side of things. We pass our tokenize_and_cut function as the 
            tokenizer. The preprocessing argument is a function that takes in the example after it 
            has been tokenized, this is where we will convert the tokens to their indexes. Finally, 
            we define the special tokens - making note that we are defining them to be their index value 
            and not their string value, i.e. 100 instead of [UNK].
            This is because the sequences will already be converted into indexes.
            """
            TEXT = data.Field(
                            batch_first = True,
                            use_vocab = False,
                            tokenize = tokenize_and_cut,
                            preprocessing = tokenizer.convert_tokens_to_ids,
                            init_token = init_token_idx,
                            eos_token = eos_token_idx,
                            pad_token = pad_token_idx,
                            unk_token = unk_token_idx
                        )

            # We define the label field as before.
            LABEL = data.LabelField(dtype = torch.float)
            
            # We load the data and create the validation splits as before.
            train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
            if 1 > test_ratio :   
                test_data,_ = test_data.split(
                                                random_state = seed, 
                                                split_ratio = test_ratio
                                            )
            if 1 > train_ratio :
                train_data,_ = train_data.split(
                                                random_state = seed, 
                                                split_ratio = train_ratio
                                            )
            train_data, valid_data = train_data.split(
                                                    random_state = seed, 
                                                    split_ratio = split_ratio
                                                )
            print(f"Number of training examples: {len(train_data)}")
            print(f"Number of validation examples: {len(valid_data)}")
            print(f"Number of testing examples: {len(test_data)}")
            
            # Although we've handled the vocabulary for the text, we still need to build the vocabulary for the labels.
            LABEL.build_vocab(train_data)
            print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
            
            """
            As before, we create the iterators. Ideally we want to use the largest batch size 
            that we can as I've found this gives the best results for transformers.
            """
            train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                    (train_data, valid_data, test_data), 
                                    batch_size = batch_size, device = device)
            
            self.dataset = {"TEXT" : TEXT, "LABEL" : LABEL,
                            "train_data" : train_data, "valid_data" : valid_data, "test_data" : test_data,
                            "train_iterator" : train_iterator, "valid_iterator" : valid_iterator, "test_iterator" : test_iterator}            
        
        
        # count the model parameters
        self.count_parameters()
        
        # As is standard, we define our optimizer and criterion (loss function).
        if optimizer == "Adam" :
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        else :
            self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
        
        #if criterion == "BCEWithLogitsLoss" :
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Place the model and criterion onto the GPU (if available)
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)  
        
    def reload_model(self, dump_id=""):
        """
        Preloaded the parameters of an existing model
        dump_id : identifier to distinguish models in the serialization folder, 
                  is by default equal to the name of the base model.
        """

        dump_id = self.model.getID() if dump_id == "" else dump_id
        self.model.load_state_dict(torch.load(self.dump_path+"/"+dump_id+'-best-model.pth'))
        
    # produces rather large files and generates errors during serialization
    """
    def save_dataset(self):
        #pickle.dump(self.dataset, self.dump_path+'/dataset')
        torch.save(self.dataset, self.dump_path+'/dataset')
    
    def load_dataset(self):
        assert os.path.isfile(self.dump_path+'/dataset'), 'File not found'
        #self.dataset = pickle.loard(self.dump_path+'/dataset')
        self.dataset = torch.loard(self.dump_path+'/dataset')
    """
        
    def count_parameters(self):
        """print out the number of parameters in the model"""
        nb_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'The model has {nb_p:,} trainable parameters\n')
        
    def binary_accuracy(self, preds, y):
        """ 
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        
        Our criterion function calculates the loss, however we have to write our function to 
        calculate the accuracy. 
        This function first feeds the predictions through a sigmoid layer, squashing the values 
        between 0 and 1, we then round them to the nearest integer. This rounds any value greater 
        than 0.5 to 1 (a positive sentiment) and the rest to 0 (a negative sentiment).
        We then calculate how many rounded predictions equal the actual labels and average it 
        across the batch.
        """
        
        #round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float() #convert into float for division 
        acc = correct.sum() / len(correct)
        return acc
    
    def train_step(self, iterator):
        """
        One epoch of training : this function iterates over all examples, one batch at a time.
        
        iterator : data iterator
        """

        epoch_loss = 0
        epoch_acc = 0

        # to compute classification report
        y = []
        y_pred = []

        #To ensure the dropout is "turned on" while training
        self.model.train()

        for batch in iterator:
            
            """
            For each batch, we first zero the gradients. 
            Each parameter in a model has a grad attribute which stores the gradient calculated 
            by the criterion.
            """
            self.optimizer.zero_grad()
            
            if isinstance(self.model, LSTM) :
                """
                As we have set `include_lengths = True`, our batch.text is now a tuple with the 
                first element being the numericalized tensor and the second element being the 
                actual lengths of each sequence. We separate these into their own variables, 
                text and text_lengths, before passing them to the model.
                """
                text, text_lengths = batch.text
                predictions = self.model(text, text_lengths).squeeze(1)
            else :
                # We then feed the batch of sentences, batch.text, into the model
                predictions = self.model(batch.text).squeeze(1)

            """
            The loss and accuracy are then calculated using our predictions and the labels, 
            batch.label, with the loss being averaged over all examples in the batch.
            """
            loss = self.criterion(predictions, batch.label)
            acc = self.binary_accuracy(predictions, batch.label)

            # We calculate the gradient of each parameter 
            loss.backward()
            
            # and then update the parameters using the gradients and optimizer algorithm
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            y.append(batch.label)
            y_pred.append(torch.round(torch.sigmoid(predictions)))
                
         
        # flatten
        y = list(itertools.chain.from_iterable(y))
        y_pred = list(itertools.chain.from_iterable(y_pred))

        # convert tensor to integer
        y = [int(y_i) for y_i in y]
        y_pred = [int(y_i) for y_i in y_pred]

        """
        Finally, we return the loss and accuracy, averaged across the epoch. 
        The len of an iterator is the number of batches in the iterator.
        """
        return epoch_loss / len(iterator), epoch_acc / len(iterator), y, y_pred
    
    def evaluate(self, iterator):
        """
        evaluate is similar to train_step, with a few modifications as you don't want to 
        update the parameters when evaluating.

        iterator : data iterator
        """
        
        epoch_loss = 0
        epoch_acc = 0

        """
        use model.eval() puts the model in "evaluation mode" and ensure the dropout is 
        "turned off" while evaluating.
        """
        self.model.eval()

        # to compute classification report
        y = []
        y_pred = []

        """
        No gradients are calculated on PyTorch operations inside the `with no_grad()` block. 
        This causes less memory to be used and speeds up computation.
        
        The rest of the function is the same as `train`, with the removal of optimizer.zero_grad(), 
        loss.backward() and optimizer.step(), as we do not update the model's parameters when evaluating.
        """
        with torch.no_grad():

            for batch in iterator:
                
                if isinstance(self.model, LSTM) :
                    """
                    As we have set `include_lengths = True`, our batch.text is now a tuple with the 
                    first element being the numericalized tensor and the second element being the 
                    actual lengths of each sequence. We separate these into their own variables, 
                    text and text_lengths, before passing them to the model.
                    """
                    text, text_lengths = batch.text
                    predictions = self.model(text, text_lengths).squeeze(1)
                else :
                    # We then feed the batch of sentences, batch.text, into the model
                    predictions = self.model(batch.text).squeeze(1)

                """
                The loss and accuracy are then calculated using our predictions and the labels, 
                batch.label, with the loss being averaged over all examples in the batch.
                """
                loss = self.criterion(predictions, batch.label)
                acc = self.binary_accuracy(predictions, batch.label)

                y.append(batch.label)
                y_pred.append(torch.round(torch.sigmoid(predictions)))

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        # flatten
        y = list(itertools.chain.from_iterable(y))
        y_pred = list(itertools.chain.from_iterable(y_pred))

        # convert tensor to integer
        y = [int(y_i) for y_i in y]
        y_pred = [int(y_i) for y_i in y_pred]

        """
        Finally, we return the loss and accuracy, averaged across the epoch. 
        The len of an iterator is the number of batches in the iterator.
        """
        return epoch_loss / len(iterator), epoch_acc / len(iterator), y, y_pred
    
    def epoch_time(self, start_time, end_time):
        """
        Tell us how long epochs take, use to compare training times between models.
        start_time : start time of the epoch
        end_time : end time of the epoch
        """
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    
    def train(self, max_epochs = 10, improving_limit = 5, eval_metric='loss', dump_id = ""):
        """
        We then train the model through multiple epochs, an epoch being a complete pass through 
        all examples in the training and validation sets.
        
        max_epochs (int, default = 10 ) : maximun number of epochs
        improving_limit (int, default = 5 ) : If the `eval_metric` of the model does not 
                                              improve during `improving_limit` epoch, we 
                                              stop training and keep the best model.
        eval_metric (string, default = '' ) : evaluation metric
        dump_id : identifier to distinguish models in the serialization folder, 
                  is by default equal to the name of the base model.
    
        At each epoch, if the validation loss is the best we have seen so far, we'll save the 
        parameters of the model and then after training has finished we'll use that model on the test set.
        """

        assert max_epochs > 0
        assert improving_limit >= 0
        assert eval_metric in ['loss', 'binary_accuracy', 'accuracy_score', 'precision', 'recall', 'f1-score']

        def isSatisfy(best_metric, current_metric):
            """check if current metric if best like last one"""
            if eval_metric == "loss" :
                return best_metric > current_metric
            else :
                return best_metric < current_metric

        if eval_metric == "loss" :
            best_valid_metric = float('inf')
        else :
            best_valid_metric = 0
        
        if eval_metric == "accuracy_score":
            key = "accuracy"
        else :
            # loss, binary_accuracy, macro avg : precision, recall, f1-score
            key = eval_metric
            
        no_best_model = 0
        
        dump_id = self.model.getID() if dump_id == "" else dump_id
        
        # store our model evolution during training
        statistics = {}
        statistics["epoch"] = []
        for i in ["train", "valid"] :
            statistics[i] = {}
            for j in ["loss", "binary_accuracy", "accuracy_score", "precision", "recall", "f1-score"] :
                statistics[i][j] = []

        flag = False

        for epoch in range(max_epochs):

            start_time = time.time()

            train_loss, train_acc, train_y, train_y_pred = self.train_step(self.dataset["train_iterator"])
            valid_loss, valid_acc, valid_y, valid_y_pred = self.evaluate(self.dataset["valid_iterator"])
            
            train_score = accuracy_score(train_y, train_y_pred)
            valid_score = accuracy_score(valid_y, valid_y_pred)

            statistics["epoch"].append(epoch)
            statistics['train']["loss"].append(train_loss)
            statistics['train']["binary_accuracy"].append(train_acc)
            statistics['train']["accuracy_score"].append(train_score)
            statistics['valid']["loss"].append(valid_loss)
            statistics['valid']["binary_accuracy"].append(valid_acc)
            statistics['valid']["accuracy_score"].append(valid_score)

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            train_cr = classification_report(train_y, train_y_pred, labels=[0,1], output_dict = True)
            statistics['train']["precision"].append(train_cr['macro avg']['precision'])
            statistics['train']["recall"].append(train_cr['macro avg']['recall'])
            statistics['train']["f1-score"].append(train_cr['macro avg']['f1-score'])

            valid_cr = classification_report(valid_y, valid_y_pred, labels=[0,1], output_dict = True)
            valid_cr['loss']            = valid_loss
            valid_cr['binary_accuracy'] = valid_acc
            valid_cr['accuracy']        = valid_cr['accuracy']
            valid_cr['precision']       = valid_cr['macro avg']['precision']
            valid_cr['recall']          = valid_cr['macro avg']['recall']
            valid_cr['f1-score']        = valid_cr['macro avg']['f1-score'] 

            statistics['valid']["precision"].append(valid_cr['precision'])
            statistics['valid']["recall"].append(valid_cr['recall'])
            statistics['valid']["f1-score"].append(valid_cr['f1-score'])

            if isSatisfy(best_metric = best_valid_metric, current_metric = valid_cr[key]) :
                
                best_valid_metric = valid_cr[key]

                # save the best model parameters
                torch.save(self.model.state_dict(), self.dump_path+"/"+dump_id+'-best-model.pth')
                no_best_model = 0
                flag = True
         
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f"\tTrain Loss: {train_loss:.4f} | Train binary Acc: {train_acc*100:.3f}% | Train score Acc: {train_score*100:.3f}% | "
                +f"Train precision: {train_cr['macro avg']['precision']*100:.3f}% | "
                +f"Train recall: {train_cr['macro avg']['recall']*100:.3f}% | "
                +f"Train f1-score: {train_cr['macro avg']['f1-score']*100:.3f}%")
            print(f"\t Val. Loss: {valid_loss:.4f} |  Val. binary Acc: {valid_acc*100:.3f}% | Val. score Acc: {valid_score*100:.3f}% | "
                +f"Val. precision: {valid_cr['macro avg']['precision']*100:.3f}% | "
                +f"Val. recall: {valid_cr['macro avg']['recall']*100:.3f}% | "
                +f"Val. f1-score: {valid_cr['macro avg']['f1-score']*100:.3f}%")
            
            if flag :
                print("\t====== New best validation score : "+str(best_valid_metric)+"\n")
                flag = False
            else :
                print("\t====== Not a better validation score (%i / %i).\n" % (no_best_model, improving_limit))
                no_best_model = no_best_model + 1
                
            if no_best_model == (improving_limit+1) :
                break
                
        # loard the best model parameters
        self.model.load_state_dict(torch.load(self.dump_path+"/"+dump_id+'-best-model.pth')) 

        return statistics       
                
    def plot_statistics(self, statistics, figsize=(15,3)):
        """
        https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html
        https://stackoverflow.com/questions/14770735/how-do-i-change-the-figure-size-with-subplots
        """
        x = statistics["epoch"]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize = figsize)
        fig.suptitle('')

        ax1.plot(x, statistics['train']["loss"], label='train')
        ax1.plot(x, statistics['valid']["loss"], label='valid')
        ax1.set(xlabel='epoch', ylabel='loss')
        ax1.set_title('loss per epoch')
        ax1.legend()
        ax1.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.

        ax2.plot(x, statistics['train']["binary_accuracy"], label='train')
        ax2.plot(x, statistics['valid']["binary_accuracy"], label='valid')
        ax2.set(xlabel='epoch', ylabel='accuracy')
        ax2.set_title('binary accuracy per epoch')
        ax2.legend()
        ax2.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.

        ax3.plot(x, statistics['train']["accuracy_score"], label='train')
        ax3.plot(x, statistics['valid']["accuracy_score"], label='valid')
        ax3.set(xlabel='epoch', ylabel='accuracy')
        ax3.set_title('accuracy_score per epoch')
        ax3.legend()
        ax3.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize = figsize)
        fig.suptitle('')

        ax1.plot(x, statistics['train']["precision"], label='train')
        ax1.plot(x, statistics['valid']["precision"], label='valid')
        ax1.set(xlabel='epoch', ylabel='precision')
        ax1.set_title('precision per epoch')
        ax1.legend()
        ax1.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.

        ax2.plot(x, statistics['train']["recall"], label='train')
        ax2.plot(x, statistics['valid']["recall"], label='valid')
        ax2.set(xlabel='epoch', ylabel='accuracy')
        ax2.set_title('recall per epoch')
        ax2.legend()
        ax2.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.

        ax3.plot(x, statistics['train']["f1-score"], label='train')
        ax3.plot(x, statistics['valid']["f1-score"], label='valid')
        ax3.set(xlabel='epoch', ylabel='accuracy')
        ax3.set_title('f1-score per epoch')
        ax3.legend()
        ax3.label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.
        

    def test(self, dump_id = ""): 
        """
        The the model : plot the test loss and accuracy
        """
        dump_id = self.model.getID() if dump_id == "" else dump_id
        self.model.load_state_dict(torch.load(self.dump_path+"/"+dump_id+'-best-model.pth')) 
        
        test_loss, test_acc, y, y_pred = self.evaluate(self.dataset["test_iterator"])
        test_score = accuracy_score(y, y_pred)
        cr = classification_report(y, y_pred, labels=[0,1], output_dict = True)

        print(f"Test Loss: {test_loss:.4f} | Test binary Acc: {test_acc*100:.3f}% | Test score Acc: {test_score*100:.3f}% | "
            +f"Test precision: {cr['macro avg']['precision']*100:.3f}% | "
            +f"Test recall: {cr['macro avg']['recall']*100:.3f}% | "
            +f"f1-score: {cr['macro avg']['f1-score']*100:.3f}%")
        print()

        cm = confusion_matrix(y, y_pred)
        print("confusion_matrix --> [ ", cm[0], ", ", cm[1] ," ]")
        print()

        print("classification report : ")
        print("\tnegative(0) --> ", cr["0"])
        print("\tpositive(1) --> ", cr["1"])
        print("\taccuracy    --> ", cr["accuracy"])
        print("\tmacro avg   --> ", cr["macro avg"])
        print("\tweighted avg--> ", cr["weighted avg"])
        
        return y, y_pred
        
    def get_predict_sentiment(self) :
        """
        Return the function that use the model to test the sentiment of some sequences. 
        The returned function use our model to predict the sentiment of any sentence we give it. 
        As it has been trained on movie reviews, the sentences provided should also be movie reviews.
        """
        if isinstance(self.model, RNN) or isinstance(self.model, LSTM) :
        
            def predict(sentence):
                """sentence : sentence whose sentiments must be predicted"""
                self.model.eval()
                tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
                indexed = [self.dataset["TEXT"].vocab.stoi[t] for t in tokenized]
                length = [len(indexed)]
                tensor = torch.LongTensor(indexed).to(device)
                tensor = tensor.unsqueeze(1)
                if isinstance(self.model, RNN) :
                    prediction = torch.sigmoid(self.model(tensor)) 
                else :
                    length_tensor = torch.LongTensor(length)
                    prediction = torch.sigmoid(self.model(tensor, length_tensor)) 
                return prediction.item()
            
        elif isinstance(self.model, CNN) or isinstance(self.model, CNN1d):
            
            def predict(sentence, min_len = 5):
                """
                As mentioned in the implementation details, the input sentence has to be at least as 
                long as the largest filter height used. We modify our predict_sentiment function to 
                also accept a minimum length argument. If the tokenized input sentence is less 
                than min_len tokens, we append padding tokens (<pad>) to make it min_len tokens.

                sentence : sentence whose sentiments must be predicted
                """
                self.model.eval()
                tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
                if len(tokenized) < min_len:
                    tokenized += ['<pad>'] * (min_len - len(tokenized))
                indexed = [self.dataset["TEXT"].vocab.stoi[t] for t in tokenized]
                tensor = torch.LongTensor(indexed).to(device)
                tensor = tensor.unsqueeze(0)
                prediction = torch.sigmoid(self.model(tensor))
                return prediction.item()
            
        elif isinstance(self.model, BERTGRUSentiment) :
            
            def predict(sentence, tokenizer=None):
                """
                We tokenize the input sequence, trim it down to the maximum length, add the 
                special tokens to either side, convert it to a tensor, add a fake batch dimension 
                and then pass it through our model.

                sentence : sentence whose sentiments must be predicted
                """
                self.model.eval()
                tokenizer = tokenizer if tokenizer else self.tokenizer
                tokens = tokenizer.tokenize(sentence)
                tokens = tokens[:self.max_input_length-2]
                indexed = [self.init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [self.eos_token_idx]
                tensor = torch.LongTensor(indexed).to(device)
                tensor = tensor.unsqueeze(0)
                prediction = torch.sigmoid(self.model(tensor))
                return prediction.item()
        else :
            def predict(sentence):
                """sentence : sentence whose sentiments must be predicted"""
                return
        
        return predict
