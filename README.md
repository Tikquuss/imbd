In this repository, we will build machine learning models to detect sentiments (i.e. detect whether a sentence is positive or negative) using IMBD ​Large Movie Review Dataset. We will use three types of models for this purpose: recurrent models, convolutional models and models based entirely on the attention mechanism.

## Dependencies

- [Python 3](https://www.python.org/downloads/)
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/) 
- [torchtext](https://pypi.org/project/torchtext/)
- [transformers](https://pypi.org/project/transformers/)
- [spacy](https://pypi.org/project/spacy/) : after installing spacy, run this in your terminal : `python -m spacy download en` [source](https://github.com/hamelsmu/Seq2Seq_Tutorial/issues/1)

##  Pretrained models


## Train your one models

### 1) Instanciate your model among the following models: RNN, LSTM, CNN, CNN1d and BERTGRUSentiment.

```
from src.model import RNN, LSTM, CNN, CNN1d, BERTGRUSentiment, Trainer
```
```
api means `any positive integer`
```
```
rnn_model = RNN(
    input_dim = api, dimension of the one-hot vectors, which is equal to the vocabulary size, will be update to len(dataset["TEXT"].vocab) during compilation
    embedding_dim = 100, # size of the dense word vectors
    hidden_dim = 256, # size of the hidden states
    output_dim = 1 # usually the number of classes, however in the case of only 2 classes the output value is between 0 and 1 and thus can be 1-dimensional, i.e. a single scalar real number.
)
```

```
lstm_model = LSTM(
    vocab_size = api, # vocabulary size, will be update to len(dataset["TEXT"].vocab) during compilation
    embedding_dim = 100, # size of the dense word vectors
    hidden_dim = 256, # size of the hidden states
    output_dim = 1, # usually the number of classes, however in the case of only 2 classes the output value is between 0 and 1 and thus can be 1-dimensional, i.e. a single scalar real number.
    n_layers = 2, # number of layers
    bidirectional = True, # bidirectional or not
    dropout = 0.5, # we use a method of regularization called dropout. Dropout works by randomly dropping out (setting to 0) neurons in a layer during a forward pass.
    pad_idx = api # index of <pad> token in th vocabulary, will be update to dataset["TEXT"].vocab.stoi[dataset["TEXT"].pad_token] during compilation
)
```

```
# CNN1d if we want to run the 1-dimensional convolutional model, noting that both models give almost identical results.

cnn_model = CNN( 
    vocab_size = api, # vocabulary size, will be update during compilation to len(TEXT.vocab) during compilation
    embedding_dim = 100, # size of the dense word vectors
    n_filters = 100, # number of filters
    filter_sizes = [3,4,5], # size of the filters or kernel, is going to be [n x emb_dim] where n is the size of the n-grams.
    output_dim = 1, # usually the number of classes, however in the case of only 2 classes the output value is between 0 and 1 and thus can be 1-dimensional, i.e. a single scalar real number.
    dropout = 0.5, # we use a method of regularization called dropout. Dropout works by randomly dropping out (setting to 0) neurons in a layer during a forward pass.
    pad_idx = api # index of <pad> token in th vocabulary, will be update during compilation to TEXT.vocab.stoi[TEXT.pad_token]
)
```

```
from transformers import BertModel

bert_model = BERTGRUSentiment(
    bert = BertModel.from_pretrained('bert-base-uncased'), # load the pre-trained model, making sure to load the same model as we will do for the tokenizer.
    hidden_dim = 256, # size of the hidden states
    output_dim = 1, # usually the number of classes, however in the case of only 2 classes the output value is between 0 and 1 and thus can be 1-dimensional, i.e. a single scalar real number.
    n_layers = 2, # number of layers
    bidirectional = True, # bidirectional or not
    dropout = 0.25 # we use a method of regularization called dropout. Dropout works by randomly dropping out (setting to 0) neurons in a layer during a forward pass.
)
```

### 2) Create his trainer and pass him the model thanks to the model parameter of Trainer.__init__. The dump_path parameter of the same method allows to define the folder where the data will be stored after processing and the models after training.

```
trainer = Trainer(
    model = "your model", 
    dump_path="your dump path"
)
```

### 3) Compile the trainer by providing him with the following parameters:

- optimizer (torch.optim, default = Adam) : model optimizer (use to update the model parameters)
- criterion (function, default = nn.BCEWithLogitsLoss) : loss function 
- seed (int, default = 1234) : random seeds for reproducibility
- split_ratio (float between 0 and 1, default = 0.8) : ratio of training data to use for training, the rest for validation
- batch_size (int, default = 64) : number of examples per batch
- max_vocab_size (int, default = 25000) : maximun token in the vocabulary

```
# load the data, build the optimizer and the loss function, and update the model parameters if necessary.
trainer.compile(
    optimizer = "SGD", # or Adam
    criterion = "BCEWithLogitsLoss", 
    seed = 1234, 
    split_ratio = 0.8, 
    batch_size = 4, 
    max_vocab_size = 25000 
)
```

### 4) Training are model

```
stats = trainer.train(
    max_epochs = 50, # maximun number of epochs
    improving_limit = 2, # If the precision of the model does not improve during `improving_limit` epoch, we stop training and keep the best model.
    dump_id = "" # identifier to distinguish models in the serialization folder, is by default equal to the name of the base model
)
```

### 5) Display statics from training and validation: evolution of loss and accuracy.

```
trainer.plot_statistics(statistics = stats)
```

### 6) Test your model

```
trainer.test(dump_id = "")
```

### 7) Putting the model into production

```
predict = trainer.get_predict_sentiment()
# example negative review...
print(predict(sentence = "This film is too scary, too much gunfire and blood spilled inside. I can't watch bad movies like this anymore."))
# example positive review...
print(predict(sentence = "Among these actors, I prefer the most romantic one, he likes what he does, is positive about chess and knows how to celebrate victories."))
```

## References

### Cross-lingual Language Model Pretraining

[1] G. Lample *, A. Conneau * [*Cross-lingual Language Model Pretraining*](https://arxiv.org/abs/1901.07291) and [facebookresearch/XLM](https://github.com/facebookresearch/XLM)

```
@article{lample2019cross,
  title={Cross-lingual Language Model Pretraining},
  author={Lample, Guillaume and Conneau, Alexis},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

### Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

[2] Chelsea Finn, Pieter Abbeel, Sergey Levine [*Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*](https://arxiv.org/abs/1911.02116) and [cbfinn/maml](https://github.com/cbfinn/maml)

```
@article{Chelsea et al.,
  title={Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks},
  author={Chelsea Finn, Pieter Abbeel, Sergey Levine},
  journal={Proceedings of the 34th International Conference on Machine Learning, Sydney, Australia, PMLR 70, 2017},
  year={2017}
}
```

## License
See the [LICENSE](LICENSE) file for more details.