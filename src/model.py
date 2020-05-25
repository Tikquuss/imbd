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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nlp = spacy.load('en')

def get_dataset(seed = 1234, max_vocab_size = 25000, batch_size = 64, include_lengths = True, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_):
    
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    TEXT = data.Field(tokenize = 'spacy', include_lengths = include_lengths)
    LABEL = data.LabelField(dtype = torch.float)
                 
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split(random_state = random.seed(seed))
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')
    
    TEXT.build_vocab(train_data, max_size = max_vocab_size, vectors = vectors, unk_init = unk_init)
    LABEL.build_vocab(train_data)
    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
    
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = batch_size, sort_within_batch = True, device = device
    )
    
    return {"TEXT" : TEXT, "LABEL" : LABEL,
            "train_data" : train_data, "valid_data" : valid_data, "test_data" : test_data,
            "train_iterator" : train_iterator, "valid_iterator" : valid_iterator, "test_iterator" : test_iterator}
    
class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        
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
        return 'RNN'
    
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_perent = dropout
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
        return 'LSTM'
    
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,  dropout, pad_idx):
        
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
        return 'CNN'
    
class CNN1d(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        
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
        return 'CNN1d'
    
    
class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        
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
        return 'BERT'

class Trainer():
    def __init__(self, model, dump_path = ""):
        assert any([isinstance(model, className) for className in [RNN, LSTM, CNN, CNN1d, BERTGRUSentiment]]), "Model type not supported"
        self.model = model
        self.dump_path = dump_path
        
    def compile(self, optimizer = "Adam", criterion = "BCEWithLogitsLoss", seed = 1234, batch_size = 64, max_vocab_size = 25000):
        
        assert optimizer in ["SGD", "Adam"], "optimizer type not supported"
        assert criterion in ["BCEWithLogitsLoss"], "criterion type not supported"
        
        """
        Instead of writing a parameterizable function to process the data, we preferred to detail the process to better justify each 
        approach.
        """
        
        if isinstance(self.model, RNN) :
            
            """
            One of the main concepts of TorchText is the Field. These define how your data should be processed. In our sentiment  
            classification task the data consists of both the raw string of the review and the sentiment, either "pos" or "neg".
            The parameters of a Field specify how the data should be processed. We use the TEXT field to define how the review should be
            processed, and the LABEL field to process the sentiment.
            Our TEXT field has tokenize='spacy' as an argument. This defines that the "tokenization" (the act of splitting the string into
            discrete "tokens") should be done using the spaCy tokenizer. If no tokenize argument is passed, the default is simply splitting
            the string on spaces.
            LABEL is defined by a LabelField, a special subset of the Field class specifically used for handling labels. We will explain 
            the dtype argument later.
            For more on Fields, go here : https://github.com/pytorch/text/blob/master/torchtext/data/field.py
            We also set the random seeds for reproducibility.
            """
            
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

            TEXT = data.Field(tokenize = 'spacy')
            LABEL = data.LabelField(dtype = torch.float)

            """
            Another handy feature of TorchText is that it has support for common datasets used in natural language processing (NLP). 
            The following code automatically downloads the IMDb dataset and splits it into the canonical train/test splits as
            `torchtext.datasets` objects. It process the data using the `Fields` we have previously defined. The IMDb dataset consists of
            50,000 movie reviews, each marked as being a positive or negative review.
            """
            train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
            
            """
            The IMDb dataset only has train/test splits, so we need to create a validation set. We can do this with the `.split()` method. 
            By default this splits 70/30, however by passing a `split_ratio` argument, we can change the ratio of the split, i.e. a `
            split_ratio` of 0.8 would mean 80% of the examples make up the training set and 20% make up the validation set. 
            We also pass our random seed to the `random_state` argument, ensuring that we get the same train/validation split each time.
            """
            train_data, valid_data = train_data.split(random_state = random.seed(seed))
            
            # We can see how many examples are in each split by checking their length.
            print(f'Number of training examples: {len(train_data)}')
            print(f'Number of validation examples: {len(valid_data)}')
            print(f'Number of testing examples: {len(test_data)}')
            
            """
            Next, we have to build a _vocabulary_. This is a effectively a look up table where every unique word in your data set has a
            corresponding _index_ (an integer).
            We do this as our machine learning model cannot operate on strings, only numbers. Each _index_ is used to construct a _one-hot_
            vector for each word. A one-hot vector is a vector where all of the elements are 0, except one, which is 1, and dimensionality
            is the total number of unique words in your vocabulary, commonly denoted by V.
            The number of unique words in our training set is over 100,000, which means that our one-hot vectors will have over 100,000 
            dimensions! This will make training slow and possibly won't fit onto your GPU (if you're using one). 
            There are two ways effectively cut down our vocabulary, we can either only take the top $n$ most common words or ignore words
            that appear less than $m$ times. We'll do the former, only keeping the top 25,000 words.

            What do we do with words that appear in examples but we have cut from the vocabulary? We replace them with a special _unknown_
            or `<unk>` token. For example, if the sentence was "This film is great and I love it" but the word "love" was not in the 
            vocabulary, it would become "This film is great and I `<unk>` it".
            The following builds the vocabulary, only keeping the most common `max_size` tokens.
            """
            TEXT.build_vocab(train_data, max_size = max_vocab_size)
            LABEL.build_vocab(train_data)
            print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
            print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
            """
            Why do we only build the vocabulary on the training set? When testing any machine learning system you do not want to look at the
            test set in any way. We do not include the validation set as we want it to reflect the test set as much as possible.
            """

            train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                (train_data, valid_data, test_data), 
                batch_size = batch_size, device = device
            )

            """
            The final step of preparing the data is creating the iterators. We iterate over these in the training/evaluation loop, and they
            return a batch of examples (indexed and converted into tensors) at each iteration.

            We'll use a `BucketIterator` which is a special type of iterator that will return a batch of examples where each example is of a
            similar length, minimizing the amount of padding per example.
            
            We also want to place the tensors returned by the iterator on the GPU (if you're using one). PyTorch handles this using
            `torch.device`, we then pass this device to the iterator.
            """
            self.dataset = {"TEXT" : TEXT, "LABEL" : LABEL,
                            "train_data" : train_data, "valid_data" : valid_data, "test_data" : test_data,
                            "train_iterator" : train_iterator, "valid_iterator" : valid_iterator, "test_iterator" : test_iterator}
                    
            # update the input dimention of the model
            """
            We now create an instance of our LSTM class.
            The input dimension is the dimension of the one-hot vectors, which is equal to the vocabulary size.
            The embedding dimension is the size of the dense word vectors. This is usually around 50-250 dimensions, but depends on the size 
            of the vocabulary.
            The hidden dimension is the size of the hidden states. This is usually around 100-500 dimensions, but also depends on factors
            such as on the vocabulary size, the size of the dense vectors and the complexity of the task.
            The output dimension is usually the number of classes, however in the case of only 2 classes the output value is between 0 and 1
            and thus can be 1-dimensional, i.e. a single scalar real number.
            """
            self.model = RNN(
                input_dim = len(self.dataset["TEXT"].vocab), 
                embedding_dim = self.model.embedding_dim, 
                hidden_dim = self.model.hidden_dim, 
                output_dim = self.model.output_dim
            )
            
        elif isinstance(self.model, LSTM) :
          
            # Previous comments are valid here.
            
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            
            """
            As above, we'll set the seed, define the `Fields` and get the train/valid/test splits.

            We'll be using packed padded sequences, which will make our RNN only process the non-padded elements of our sequence, and for
            any padded element the `output` will be a zero tensor. To use packed padded sequences, we have to tell the RNN how long the 
            actual sequences are. We do this by setting `include_lengths = True` for our `TEXT` field. This will cause `batch.text` to now
            be a tuple with the first element being our sentence (a numericalized tensor that has been padded) and the second element being
            the actual lengths of our sentences.
            """
            TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
            LABEL = data.LabelField(dtype = torch.float)
            
            # We then load the IMDb dataset.
            train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
            train_data, valid_data = train_data.split(random_state = random.seed(seed))
            print(f'Number of training examples: {len(train_data)}')
            print(f'Number of validation examples: {len(valid_data)}')
            print(f'Number of testing examples: {len(test_data)}')

            """
            Next is the use of pre-trained word embeddings. Now, instead of having our word embeddings initialized randomly, they are 
            initialized with these pre-trained vectors.
            We get these vectors simply by specifying which vectors we want and passing it as an argument to `build_vocab`. `TorchText` 
            handles downloading the vectors and associating them with the correct words in our vocabulary.

            Here, we'll be using the `"glove.6B.100d" vectors"`. `glove` (https://nlp.stanford.edu/projects/glove/) is the algorithm used to 
            calculate the vectors. `6B` indicates these vectors were trained on 6 billion tokens and `100d` indicates these vectors are 100-
            dimensional.

            You can see the other available vectors here : https://github.com/pytorch/text/blob/master/torchtext/vocab.py#L113.

            The theory is that these pre-trained vectors already have words with similar semantic meaning close together in vector space, 
            e.g. "terrible", "awful", "dreadful" are nearby. This gives our embedding layer a good initialization as it does not have to 
            learn these relations from scratch.

            **Note**: these vectors are about 862MB, so watch out if you have a limited internet connection.

            By default, TorchText will initialize words in your vocabulary but not in your pre-trained embeddings to zero. We don't want
            this, and instead initialize them randomly by setting `unk_init` to `torch.Tensor.normal_`. This will now initialize those words
            via a Gaussian distribution.
            """
            TEXT.build_vocab(train_data, max_size = max_vocab_size, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
            LABEL.build_vocab(train_data)
            print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
            print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
    
            train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                (train_data, valid_data, test_data), 
                batch_size = batch_size, sort_within_batch = True, device = device
            )

            self.dataset = {"TEXT" : TEXT, "LABEL" : LABEL,
                    "train_data" : train_data, "valid_data" : valid_data, "test_data" : test_data,
                    "train_iterator" : train_iterator, "valid_iterator" : valid_iterator, "test_iterator" : test_iterator}
            
            pretrained_embeddings = self.dataset["TEXT"].vocab.vectors
            print("pretrained_embeddings.shape", pretrained_embeddings.shape)
            self.model.embedding.weight.data.copy_(pretrained_embeddings)
            UNK_IDX = self.dataset["TEXT"].vocab.stoi[self.dataset["TEXT"].unk_token]

            self.model.embedding.weight.data[UNK_IDX] = torch.zeros(self.model.embedding_dim)
            self.model.embedding.weight.data[self.model.pad_idx] = torch.zeros(self.model.embedding_dim)
            print("self.model.embedding.weight.data", self.model.embedding.weight.data)
            
            
            
            self.model = LSTM(
                vocab_size = len(self.dataset["TEXT"].vocab), 
                embedding_dim = self.model.embedding_dim, 
                hidden_dim = self.model.hidden_dim, 
                output_dim = self.model.output_dim, 
                n_layers = self.model.n_layers, 
                bidirectional = self.model.bidirectional, 
                dropout = self.model.dropout_perent, 
                pad_idx = self.dataset["TEXT"].vocab.stoi[self.dataset["TEXT"].pad_token] 
            )

            
        elif isinstance(self.model, CNN) or isinstance(self.model, CNN1d) :
            
            """
            Above, we'll prepare the data. 
            Unlike the above with the FastText model, we no longer explicitly need to create the bi-grams and append them to the end of the 
            sentence.
            As convolutional layers expect the batch dimension to be first we can tell TorchText to return the data already permuted using
            the `batch_first = True` argument on the field.
            """
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            
            TEXT = data.Field(tokenize = 'spacy', batch_first = True)
            LABEL = data.LabelField(dtype = torch.float)

            train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
            train_data, valid_data = train_data.split(random_state = random.seed(seed))
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
                
            pretrained_embeddings = self.dataset["TEXT"].vocab.vectors
            self.model.embedding.weight.data.copy_(pretrained_embeddings)
            UNK_IDX = self.dataset["TEXT"].vocab.stoi[self.dataset["TEXT"].unk_token]
            self.model.embedding.weight.data[UNK_IDX] = torch.zeros(self.model.embedding_dim)
            self.model.embedding.weight.data[self.model.pad_idx] = torch.zeros(self.model.embedding_dim)
            
        elif isinstance(self.model, BERTGRUSentiment) :
    
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            
            """
            The transformer has already been trained with a specific vocabulary, which means we need to train with the exact same vocabulary
            and also tokenize our data in the same way that the transformer did when it was initially trained.

            Luckily, the transformers library has tokenizers for each of the transformer models provided. In this case we are using the BERT 
            model which ignores casing (i.e. will lower case every word). We get this by loading the pre-trained `bert-base-uncased`
            tokenizer.
            """
            
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.tokenizer = tokenizer
            max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
            
            def tokenize_and_cut(sentence):
                tokens = tokenizer.tokenize(sentence) 
                tokens = tokens[:max_input_length-2]
                return tokens
                    
            init_token_idx = tokenizer.cls_token_id
            eos_token_idx = tokenizer.sep_token_id
            pad_token_idx = tokenizer.pad_token_id
            unk_token_idx = tokenizer.unk_token_id
            
            TEXT = data.Field(batch_first = True,
                              use_vocab = False,
                              tokenize = tokenize_and_cut,
                              preprocessing = tokenizer.convert_tokens_to_ids,
                              init_token = init_token_idx,
                              eos_token = eos_token_idx,
                              pad_token = pad_token_idx,
                              unk_token = unk_token_idx)

            LABEL = data.LabelField(dtype = torch.float)
            
            train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
            train_data, valid_data = train_data.split(random_state = random.seed(seed))
            print(f"Number of training examples: {len(train_data)}")
            print(f"Number of validation examples: {len(valid_data)}")
            print(f"Number of testing examples: {len(test_data)}")
            
            LABEL.build_vocab(train_data)
            #print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
            print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
            
            train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                (train_data, valid_data, test_data), 
                batch_size = batch_size, 
                device = device)
            
            self.dataset = {"TEXT" : TEXT, "LABEL" : LABEL,
                            "train_data" : train_data, "valid_data" : valid_data, "test_data" : test_data,
                            "train_iterator" : train_iterator, "valid_iterator" : valid_iterator, "test_iterator" : test_iterator}            
        
        
        self.count_parameters()
        
        if optimizer == "Adam" :
            self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
        else :
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            
        #if criterion == "BCEWithLogitsLoss" :
        self.criterion = nn.BCEWithLogitsLoss()
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)  
        
    def reload_model(self, dump_id=""):
        dump_id = self.model.getID() if dump_id == "" else dump_id
        self.model.load_state_dict(torch.load(self.dump_path+"/"+dump_id+'-best-model.pth'))
        
    # produces rather large files and generates errors during serialization
    """
    def save_dataset(self):
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)
        #pickle.dump(self.dataset, self.dump_path+'/dataset')
        torch.save(self.dataset, self.dump_path+'/dataset')
    
    def load_dataset(self):
        assert os.path.isfile(self.dump_path+'/dataset'), 'File not found'
        #self.dataset = pickle.loard(self.dump_path+'/dataset')
        self.dataset = torch.loard(self.dump_path+'/dataset')
    """
        
    def count_parameters(self):
        if not isinstance(self.model, BERTGRUSentiment) :
            nb_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'The model has {nb_p:,} trainable parameters')
        else :
            nb_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'The model has {nb_p:,} total parameters')
            for name, param in self.model.named_parameters():                
                if name.startswith('bert'):
                    param.requires_grad = False
            nb_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'The model has {nb_p:,} trainable parameters ...')
            for name, param in self.model.named_parameters():                
                if param.requires_grad:
                    print(name)
        print()
    
    def binary_accuracy(self, preds, y):
        """ 
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        #round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float() #convert into float for division 
        acc = correct.sum() / len(correct)
        return acc
    
    def train_step(self, iterator):
        
        epoch_loss = 0
        epoch_acc = 0

        self.model.train()

        for batch in iterator:

            self.optimizer.zero_grad()
            
            if isinstance(self.model, LSTM) :
                text, text_lengths = batch.text
                predictions = self.model(text, text_lengths).squeeze(1)
            else :
                predictions = self.model(batch.text).squeeze(1)

            loss = self.criterion(predictions, batch.label)

            acc = self.binary_accuracy(predictions, batch.label)

            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    
    def evaluate(self, iterator):
        
        epoch_loss = 0
        epoch_acc = 0

        self.model.eval()

        with torch.no_grad():

            for batch in iterator:
                
                if isinstance(self.model, LSTM) :
                    text, text_lengths = batch.text
                    predictions = self.model(text, text_lengths).squeeze(1)
                else :
                    predictions = self.model(batch.text).squeeze(1)

                loss = self.criterion(predictions, batch.label)

                acc = self.binary_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def train(self, max_epochs = 100, improving_limit = 10, dump_id = ""):
        
        assert max_epochs > 0
        
        no_best_model = 0
        
        dump_id = self.model.getID() if dump_id == "" else dump_id
              
        best_valid_loss = float('inf')

        for epoch in range(max_epochs):

            start_time = time.time()

            train_loss, train_acc = self.train_step(self.dataset["train_iterator"])
            valid_loss, valid_acc = self.evaluate(self.dataset["valid_iterator"])

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.dump_path+"/"+dump_id+'-best-model.pth')
                no_best_model = 0
         
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
            
            if best_valid_loss < valid_loss:
                print("\tNot a better validation score (%i / %i)." % (no_best_model, improving_limit))
                no_best_model = no_best_model + 1
            else :
                print("\tNew best validation score")
            if no_best_model == (improving_limit+1) :
                break
                        
    def test(self): 
        test_loss, test_acc = self.evaluate(self.dataset["test_iterator"])
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
        
    def get_predict_sentiment(self) :
        if isinstance(self.model, RNN) or isinstance(self.model, LSTM) :
            def predict(sentence):
                self.model.eval()
                tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
                indexed = [self.dataset["TEXT"].vocab.stoi[t] for t in tokenized]
                length = [len(indexed)]
                tensor = torch.LongTensor(indexed).to(device)
                tensor = tensor.unsqueeze(1)
                length_tensor = torch.LongTensor(length)
                if isinstance(self.model, RNN) :
                    prediction = torch.sigmoid(self.model(tensor)) 
                else :
                    prediction = torch.sigmoid(self.model(tensor, length_tensor)) 
                return prediction.item()
            
        elif isinstance(self.model, CNN) or isinstance(self.model, CNN1d):
            
            def predict(sentence, min_len = 5):
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
                # bert
                self.model.eval()
                tokenizer = tokenizer if tokenizer else self.tokenizer
                tokens = tokenizer.tokenize(sentence)
                tokens = tokens[:max_input_length-2]
                indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
                tensor = torch.LongTensor(indexed).to(device)
                tensor = tensor.unsqueeze(0)
                prediction = torch.sigmoid(self.model(tensor))
                return prediction.item()
        else :
            def predict(sentence):
                return
        
        return predict
  