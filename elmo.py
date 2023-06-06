from datasets import load_dataset
from torchtext import vocab
from sklearn.metrics import confusion_matrix
import nltk
from sklearn.metrics import classification_report
import re
from nltk.tokenize import word_tokenize
import torch
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
from nltk.stem import WordNetLemmatizer
import string
import torch.nn.functional as F
import torch.nn as nn
from math import floor
import random
from nltk.corpus import stopwords
from warnings import filterwarnings
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import scikitplot as skplt

filterwarnings('ignore')

def init_preprocess():
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    sst_dataset_init = load_dataset('sst')  
    nli_dataset_init = load_dataset('multi_nli')
    return sst_dataset_init, nli_dataset_init

sst_dataset_init, nli_dataset_init = init_preprocess()

def init_len():
    global sst_train_len, sst_valid_len, sst_test_len, nli_train_len, nli_train_len_new, nli_valid_len, nli_test_len
    sst_train_len = 8544
    sst_valid_len = 1101
    sst_test_len = 2210

    print("SST initialised")

    nli_train_len = 392702
    nli_train_len_new = 40000
    nli_valid_len = 9815
    nli_test_len = 9832

    print("NLI initialised")

init_len()

def build_vocab(object):
    words = [[word] for sent in object.preprocessed_data for word in sent[0]]
    vocab = build_vocab_from_iterator(words, min_freq=object.min_freq, specials=[object.unk, object.pad])
    vocab.set_default_index(vocab[object.unk])
    return words, vocab

def common_init(object):
    object.stop_words = set(stopwords.words('english'))
    object.lemmatizer = WordNetLemmatizer()
    object.preprocessed_data = list()
    object.pad = '<PAD>'
    object.min_freq = 5
    object.unk = '<UNK>'
    return object


class Preprocess_sst():
    def __init__(self, data):
        self.data = data
        self = common_init(self)
        self.main()

    def caller(self):
        self.label = floor(self.label * 10) / 10
        self.text = self.text.translate(
            str.maketrans('', '', string.punctuation)).lower()
        self.text = word_tokenize(self.text)
        self.text = [word for word in self.text if word not in self.stop_words]
        self.text = [word for word in self.text if not re.match(r'\d+', word) and len(word) > 1]
        self.text = [self.lemmatizer.lemmatize(word) for word in self.text]

    def main(self):
        data_types = ['train', 'validation', 'test']
        for data_type in data_types:
            for example in self.data[data_type]:
                self.label, self.text = example['label'], example['sentence']
                self.caller()
                self.preprocessed_data.append((self.text, self.label))
        
        self.words, self.vocab = build_vocab(self)

preprocesser_sst = Preprocess_sst(sst_dataset_init)
sst_dataset = preprocesser_sst.preprocessed_data
vocabulary_sst = preprocesser_sst.vocab

class Preprocess_nli():
    def __init__(self, data):
        self.data = data
        self = common_init(self)
        self.main()

    def caller(self):
        self.text = self.text.translate(
            str.maketrans('', '', string.punctuation)).lower()
        self.text = word_tokenize(self.text)
        self.text = [word for word in self.text if word not in self.stop_words]
        self.text = [word for word in self.text if not re.match(r'\d+', word) and len(word) > 1]
        self.text = [self.lemmatizer.lemmatize(word) for word in self.text]

    def main(self):
        check = False
        for i, example in enumerate(self.data['train']):
            if i > nli_train_len_new:
                break
            self.label, self.text = example['label'], example['premise']
            self.caller()
            self.preprocessed_data.append((self.text, self.label))
        
        for data_type in ['validation_matched', 'validation_mismatched']:
            for example in self.data[data_type]:
                self.label, self.text = example['label'], example['premise']
                self.caller()
                self.preprocessed_data.append((self.text, self.label))
                check = True
        
        if check:
            self.words, self.vocab = build_vocab(self)

        self.words, self.vocab = build_vocab(self)

preprocesser_nli = Preprocess_nli(nli_dataset_init)
nli_dataset = preprocesser_nli.preprocessed_data
vocabulary_nli = preprocesser_nli.vocab

def create_data_for_sst_classification():
    padding_index = vocabulary_sst['<PAD>']
    max_length = 0
    token_indices = []
    labels = []
    unique_tokens = set()
    for sentence, label in sst_dataset:
        token_indices_for_sentence = [vocabulary_sst[token] for token in sentence]
        token_indices.append(token_indices_for_sentence)
        binary_label = 0 if label < 0.5 else 1
        labels.append(binary_label)
        max_length = max(max_length, len(token_indices_for_sentence))
        unique_tokens.update(set(sentence))
    padded_token_indices = [sent + [padding_index] * (max_length - len(sent)) for sent in token_indices]
    tensor_token_indices = torch.tensor(padded_token_indices)
    tensor_labels = torch.tensor(labels)
    print(unique_tokens)
    return tensor_token_indices, tensor_labels

sst_tokens, sst_labels = create_data_for_sst_classification()
sst_tokens_train = sst_tokens[:sst_train_len]
sst_tokens_valid = sst_tokens[sst_train_len:sst_valid_len]
sst_tokens_test = sst_tokens[sst_train_len + sst_valid_len:]
sst_labels_train = sst_labels[:sst_train_len]
sst_labels_valid = sst_labels[sst_train_len:sst_valid_len]
sst_labels_test = sst_labels[sst_train_len + sst_valid_len:]

def create_data_for_nli_classification():
    token_indices = []
    labels = []
    unique_labels = set()
    max_len = 0
    for sent, label in nli_dataset:
        indices = []
        for word in sent:
            if word not in vocabulary_nli:
                # Replace unknown words with a special unknown token index
                indices.append(vocabulary_nli['<unk>'])
            else:
                indices.append(vocabulary_nli[word])
        token_indices.append(indices)
        labels.append(label)
        unique_labels.add(label)
        max_len = max(max_len, len(indices))
        token_tensors = []
    for indices in token_indices:
        padding = [vocabulary_nli['<PAD>']] * (max_len - len(indices))
        padded_indices = indices + padding
        token_tensors.append(padded_indices)
    token_tensors = torch.tensor(token_tensors, dtype=torch.long)
    label_tensors = torch.tensor(labels)
    print(unique_labels)
    return token_tensors, label_tensors

nli_tokens, nli_labels = create_data_for_nli_classification()
nli_train_len = nli_train_len_new + nli_valid_len
nli_tokens_train, nli_tokens_valid, nli_tokens_test = torch.split(nli_tokens, [nli_train_len_new, nli_valid_len, len(nli_tokens)-nli_train_len])
nli_labels_train, nli_labels_valid, nli_labels_test = torch.split(nli_labels, [nli_train_len_new, nli_valid_len, len(nli_labels)-nli_train_len])
print(nli_tokens_train.shape, nli_labels_train.shape)

class Create_data_for_pretraining():
    def __init__(self):
        self.predictions = []
        self.contexts = []
        self.build()

    def __len__(self):
        return self.contexts.shape[0]

    def __getitem__(self, index):
        return self.contexts[index], self.predictions[index]

    def build(self):
        PAD_TOKEN = vocabulary_sst['<PAD>']
        max_len = max(len(sent) for sent, label in sst_dataset)
        for sent, label in sst_dataset:
            ind = [vocabulary_sst[word] for word in sent]
            for i in range(len(ind)):
                context_left = ind[:i]
                context_right = ind[i+1:]
                prediction = ind[i]

                self.contexts.append(context_left + context_right + [PAD_TOKEN]*(max_len - len(context_left) - len(context_right)))
                self.predictions.append(prediction)

        self.contexts = torch.tensor(self.contexts)
        self.predictions = torch.tensor(self.predictions)

sst_word_pred_dataset = Create_data_for_pretraining()

class Create_data_for_pretraining_nli():
    def __init__(self):
        self.predictions = []
        self.contexts = []
        self.build()

    def __len__(self):
        return self.contexts.shape[0]

    def __getitem__(self, index):
        return self.contexts[index], self.predictions[index]

    def build(self):
        PAD_TOKEN = vocabulary_nli['<PAD>']
        max_len = max(len(sent) for sent, label in nli_dataset)
        for sent, label in nli_dataset:
            ind = [vocabulary_nli[word] for word in sent]
            for i in range(len(ind)):
                context_left = ind[:i]
                context_right = ind[i+1:]
                prediction = ind[i]
                self.contexts.append(context_left + context_right + [PAD_TOKEN]*(max_len - len(context_left) - len(context_right)))
                self.predictions.append(prediction)

        self.contexts = torch.tensor(self.contexts)
        self.predictions = torch.tensor(self.predictions)

nli_word_pred_dataset = Create_data_for_pretraining_nli()

print(2)

class SstClass():
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def get_len(self):
        return self.tokens.shape[0]

    def get_item(self, index):
        return self.tokens[index], self.labels[index]


sst_class_train = SstClass(sst_tokens_train, sst_labels_train)
sst_class_valid = SstClass(sst_tokens_valid, sst_labels_valid)
sst_class_test = SstClass(sst_tokens_test, sst_labels_test)

sst_class_train.__class__ = type('Sst_class_train', (SstClass,), {'__getitem__': lambda self, index: self.get_item(index), '__len__': lambda self: self.get_len()})
sst_class_valid.__class__ = type('Sst_class_valid', (SstClass,), {'__getitem__': lambda self, index: self.get_item(index), '__len__': lambda self: self.get_len()})
sst_class_test.__class__ = type('Sst_class_test', (SstClass,), {'__getitem__': lambda self, index: self.get_item(index), '__len__': lambda self: self.get_len()})

class NliClass():
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def get_len(self):
        return self.tokens.shape[0]

    def get_item(self, index):
        return self.tokens[index], self.labels[index]


nli_class_train = NliClass(nli_tokens_train, nli_labels_train)
nli_class_valid = NliClass(nli_tokens_valid, nli_labels_valid)
nli_class_test = NliClass(nli_tokens_test, nli_labels_test)

nli_class_train.__class__ = type('Nli_class_train', (NliClass,), {'__getitem__': lambda self, index: self.get_item(index), '__len__': lambda self: self.get_len()})
nli_class_valid.__class__ = type('Nli_class_valid', (NliClass,), {'__getitem__': lambda self, index: self.get_item(index), '__len__': lambda self: self.get_len()})
nli_class_test.__class__ = type('Nli_class_test', (NliClass,), {'__getitem__': lambda self, index: self.get_item(index), '__len__': lambda self: self.get_len()})

print(3)

def init_glove():
    embedding_dim = 300
    glove_embeds = vocab.GloVe(name='6B', dim=embedding_dim)
    glove_vectors = glove_embeds.vectors
    unk = torch.mean(glove_vectors, dim=0)
    return glove_embeds, unk

glove_embeds, unk = init_glove()

sst_embeddings = [glove_embeds[word] if word in glove_embeds.itos else unk for word in vocabulary_sst.get_itos()]
nli_embeddings = [glove_embeds[word] if word in glove_embeds.itos else unk for word in vocabulary_nli.get_itos()]

def cell_state_processing(y, a):
    y = torch.mul(y, a).sum(dim=2).flatten(1, 2)
    return y


def transpose_weights(y, num_layers):
    return y.transpose(0, 1).view(y.shape[1], 2, num_layers, y.shape[2])

def init_model(object):
    object.num_layers = 2
    object.hidden_size = 300
    object.pad = object.vocab['<PAD>']
    object.labels_num = 2 # for sst
    # object.labels_num = 3 # for NLI
    return object

def attention_model(object, embeddings):
    object.lstm = nn.LSTM(object.hidden_size, object.hidden_size,
                            num_layers=object.num_layers, bidirectional=True, batch_first=True)
    object.weights = torch.rand(2)
    object.sigmoid = nn.Sigmoid()
    object.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
    object.softmax = nn.Softmax(dim=0)    
    return object

class Elmo(nn.Module):
    def __init__(self, vocab, embeddings, hidden_size=300):
        super().__init__()
        self.vocab = vocab
        self.vocab_len = len(vocab)
        self = init_model(self)

        self = attention_model(self, embeddings)

        self.word_pred = nn.Linear(self.hidden_size*2, self.vocab_len)
        self.classifier = nn.Linear(self.hidden_size*2, self.labels_num)


    def forward(self, x, label):
        y = self.embedding(x)

        a, (y, b) = self.lstm(y)

        prev_shape = y.shape
        
        y = transpose_weights(y, self.num_layers)
        a = self.softmax(self.weights).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(prev_shape[1], 2, 1, prev_shape[2]).to(device)

        return cell_state_processing(y, a)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now:', device)

elmo = Elmo(vocabulary_sst, torch.stack(sst_embeddings))
# elmo = Elmo(vocabulary_nli, torch.stack(nli_embeddings))

def pretrain_sst(model, data, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Iterate through epochs
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Iterate through data in batches
        for inputs, targets in data:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero out gradients
            optimizer.zero_grad()
            
            # Compute model output and predictions
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            
            # Compute batch loss and update weights
            batch_loss = criterion(outputs, targets)
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
        
        # Print the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(data)
        print(f'Epoch {epoch}/{num_epochs}\tTrain Loss: {avg_epoch_loss:.4f}')

    # Return the trained model
    return model

data = DataLoader(sst_word_pred_dataset, batch_size=128)
# elmo = pretrain_sst(elmo, data)

# torch.save(elmo.state_dict(), 'pretrain_sst.pth')
elmo.load_state_dict(torch.load('pretrain_sst.pth', map_location=torch.device('cpu')))

# data = DataLoader(nli_word_pred_dataset, batch_size=128)

# elmo = Elmo(vocabulary_nli, torch.stack(nli_embeddings))

for param in elmo.lstm.parameters():
    param.requires_grad = False

print(4)

def pretrain_elmo_nli(model, data):
    num_epochs = 1
    device = model.device
    loss_fn = nn.CrossEntropyLoss().to(device)
    pad = vocabulary_nli['<PAD>']
    optim = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(num_epochs), desc='epoch'):
        epoch_loss = 0
        num_batches = 0

        model.train()
        for input_tensor, label_tensor in data:
            # Move input tensor and label tensor to the specified device
            input_tensor = input_tensor.to(device)
            label_tensor = label_tensor.to(device)
            optim.zero_grad()
            hidden_states = model.forward(input_tensor, label_tensor)
            word_predictions = model.word_predictions(hidden_states)
            loss = loss_fn(word_predictions.view(-1, model.vocab_size), label_tensor.view(-1))
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            num_batches += 1
    
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}')

    return model

# elmo = pretrain_elmo_nli(elmo, data)

# torch.save(elmo.state_dict(), 'pretrain_nli.pth')

# elmo.load_state_dict(torch.load('pretrain_nli.pth',
#                      map_location=torch.device('cpu')))

print(5)

def elmo_sst(model, data):
    num_epochs = 50
    model = model.to(device)
    pad = vocabulary_sst['<PAD>']
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Set the model to training mode
    model.train()

    # Loop over the epochs
    for epoch in range(num_epochs):
        # Initialize total loss and batch count for the epoch
        epoch_loss = 0
        batch_count = 0
        
        for x, label in data:
            # Move input tensor and label tensor to the specified device
            x = x.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(x, label).classifier().softmax(dim=1)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        avg_epoch_loss = epoch_loss / batch_count
        
        # Print the epoch number and average loss for the epoch
        print(f'Epoch {epoch + 1}\tTrain Loss: {avg_epoch_loss:.4f}')

    return model

# data = DataLoader(sst_class_train, batch_size=4)
# elmo = elmo_sst(elmo, data)

# torch.save(elmo.state_dict(), 'finetune_sst.pth')
elmo.load_state_dict(torch.load('finetune_sst.pth', map_location=torch.device('cpu')))

print(6)

def eval_stats(model, data):
    y_true = []
    y_pred = []
    model.eval()
    count = 0
    with torch.no_grad():
        for x, label in tqdm(data):
            x = x.to(device)
            label = label.to(device)
            y_true += label.tolist()
            count += 1
            predictions = model(x, label)
            predictions = model.classifier(predictions)
            predictions = F.softmax(predictions, dim=1)
            predictions = predictions.argmax(dim=1)
            y_pred += predictions.tolist()

    return y_true, y_pred
    

def get_stats(mdl, dl):
    y_true, y_pred = eval_stats(mdl, dl)
    accuracy = sum(1 for i in range(len(y_pred)) if y_pred[i] == y_true[i])/len(y_pred)
    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    
    print('\nClassification Report:')
    print(report)
    print('\nConfusion Matrix:')
    print(matrix)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr)
    plt.show()

    auc = np.trapz(tpr, fpr)
    print("AUC: ", auc)

    return accuracy

data = DataLoader(sst_class_train, batch_size=4)
x = get_stats(elmo, data)

def elmo_nli_train(model, loss_fn, data, optim, batch_size=32, shuffle=True):
    total_loss = 0
    model.train() # set the model to train mode
    if shuffle:
        random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        x = [sample[0] for sample in batch]
        label = [sample[1] for sample in batch]
        x = torch.stack(x, dim=0).to(device)
        label = torch.stack(label, dim=0).to(device)
        ans = model(x, label).classifier().softmax()
        loss = loss_fn(ans, label)
        loss.backward()
        optim.step().zero_grad()
        total_loss += loss.item()
    return total_loss

def elmo_nli(model, data, num_epochs=2):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in tqdm(range(num_epochs)):
        model = model.to(device)
        model.train()
        loss_fn = nn.CrossEntropyLoss().to(device)
        total_loss = elmo_nli_train(model, loss_fn, data, optim)
        avg_loss = total_loss / len(data)
        print("Epoch {}:\tTrain Loss: {:.6f}".format(epoch + 1, avg_loss))
    return model

# data = DataLoader(nli_class_train, batch_size=4)
# elmo = elmo_nli(elmo, data)

# torch.save(elmo.state_dict(), 'finetune_nli.pth')
# elmo.load_state_dict(torch.load('finetune_nli.pth', map_location=torch.device('cpu')))

def get_stats_nli(mdl, dl):
    y_true, y_pred = eval_stats(mdl, dl)
    accuracy = sum(1 for i in range(len(y_pred)) if y_pred[i] == y_true[i])/len(y_pred)
    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    
    print('\nClassification Report:')
    print(report)
    print('\nConfusion Matrix:')
    print(matrix)

    # Calculate the AUC score
    y_pred_2d = np.vstack((1-np.array(y_pred), np.array(y_pred))).T
    auc_score = roc_auc_score(y_true, y_pred_2d, multi_class='ovo')

    print(f"\nAUC Score: {auc_score:.2f}")

    # Plot ROC curve
    skplt.metrics.plot_roc(y_true, y_score=y_pred_2d)
    plt.title(f"ROC Curve (AUC = {auc_score:.2f})")
    plt.show()
    
    return accuracy
# data = DataLoader(nli_class_train, batch_size=4)
# x = get_stats_nli(elmo, data)
print("Done")
