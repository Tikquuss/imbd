import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext import data, datasets
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
    def __init__(self, model, optimizer = None, criterion= None, dump_path = ""):
        assert any([isinstance(model, className) for className in [RNN, LSTM, CNN, CNN1d, BERTGRUSentiment]]), "Model type not supported"
        self.model = model
        self.count_parameters()
        
        if isinstance(self.model, RNN) :
            
            seed = 1234
            max_vocab_size = 25000
            batch_size = 64
            
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

            TEXT = data.Field(tokenize = 'spacy')
            LABEL = data.LabelField(dtype = torch.float)

            train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
            train_data, valid_data = train_data.split(random_state = random.seed(seed))
            print(f'Number of training examples: {len(train_data)}')
            print(f'Number of validation examples: {len(valid_data)}')
            print(f'Number of testing examples: {len(test_data)}')

            TEXT.build_vocab(train_data, max_size = max_vocab_size)
            LABEL.build_vocab(train_data)
            print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
            print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

            train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                (train_data, valid_data, test_data), 
                batch_size = batch_size, device = device
            )

            self.dataset = {"TEXT" : TEXT, "LABEL" : LABEL,
                    "train_data" : train_data, "valid_data" : valid_data, "test_data" : test_data,
                    "train_iterator" : train_iterator, "valid_iterator" : valid_iterator, "test_iterator" : test_iterator}
                    
        elif isinstance(self.model, LSTM) :
            seed = 1234
            max_vocab_size = 25000
            batch_size = 64
            
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

            include_lengths = True
            TEXT = data.Field(tokenize = 'spacy', include_lengths = include_lengths)
            LABEL = data.LabelField(dtype = torch.float)

            train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
            train_data, valid_data = train_data.split(random_state = random.seed(seed))
            print(f'Number of training examples: {len(train_data)}')
            print(f'Number of validation examples: {len(valid_data)}')
            print(f'Number of testing examples: {len(test_data)}')

            vectors = "glove.6B.100d"
            unk_init = torch.Tensor.normal_
            TEXT.build_vocab(train_data, max_size = max_vocab_size, vectors = vectors, unk_init = unk_init)
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
            seed = 1234
            max_vocab_size = 25000
            batch_size = 64
            
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
            
            vectors = "glove.6B.100d"
            unk_init = torch.Tensor.normal_
            TEXT.build_vocab(train_data, max_size = max_vocab_size, vectors = vectors, unk_init = unk_init)
            LABEL.build_vocab(train_data)
            print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
            print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
            
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
            
        elif isinstance(self.model, BERTGRUSentiment) :
            seed = 1234
            batch_size = 128
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
        
        self.optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = criterion if criterion else nn.BCEWithLogitsLoss()
        
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)
        
        self.dump_path = dump_path
        
    # produces rather large files and generates errors during serialization
    """
    def save_dataset(self, dump_path):
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        #pickle.dump(self.dataset, dump_path+'/dataset')
        torch.save(self.dataset, dump_path+'/dataset')
        self.dump_path = dump_path
    
    def load_dataset(self, dump_path):
        assert os.path.isfile(dump_path+'/dataset'), 'File not found'
        #self.dataset = pickle.loard(dump_path+'/dataset')
        self.dataset = torch.loard(dump_path+'/dataset')
        self.dump_path = dump_path
    """
        
    def count_parameters(self):
        if not isinstance(self.model, BERTGRUSentiment) :
            nb_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'The model has {nb_p:,} trainable parameters')
        else :
            nb_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'The model has {nb_p:,} total trainable parameters')
            for name, param in self.model.named_parameters():                
                if name.startswith('bert'):
                    param.requires_grad = False
            nb_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'The model has {nb_p:,} trainable parameters')
            for name, param in self.model.named_parameters():                
                if param.requires_grad:
                    print(name)
    
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
    
    def train(self, n_epochs = 5, dump_id = ""):
        
        assert n_epochs > 0
        
        dump_id = self.model.getID() if dump_id == "" else dump_id
              
        best_valid_loss = float('inf')

        for epoch in range(n_epochs):

            start_time = time.time()

            train_loss, train_acc = self.train_step(self.dataset["train_iterator"])
            valid_loss, valid_acc = self.evaluate(self.dataset["valid_iterator"])

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.dump_path+"/"+dump_id+'-best-model.pth')

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
                        
    
    def reload_model(self, dump_id):
        dump_id = self.model.getID() if dump_id == "" else dump_id
        self.model.load_state_dict(torch.load(self.dump_path+"/"+dump_id+'-best-model.pth'))
    
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
            
            def predict(tokenizer, sentence):
                # bert
                self.model.eval()
                tokens = tokenizer.tokenize(sentence)
                tokens = tokens[:max_input_length-2]
                indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
                tensor = torch.LongTensor(indexed).to(device)
                tensor = tensor.unsqueeze(0)
                prediction = torch.sigmoid(model(tensor))
                return prediction.item()
        else :
            def predict(sentence):
                return
        
        return predict
  