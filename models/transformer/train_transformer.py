import ast
import time
import math
import sys
import os
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
from pyJoules.energy_meter import measure_energy
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler

class NERTransformerModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, emb_size, nhead, nhid, nlayers, dropout=0.5, max_len=600):
        super(NERTransformerModel, self).__init__()
        self.emb_size = emb_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, dropout, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=nhid, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc_out = nn.Linear(emb_size, tagset_size)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.emb_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc_out(output)
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=600):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Trainer:
    def __init__(self, model, optimizer, criterion, train_data, validation_data, test_data, tag_vocab, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.tag_vocab = tag_vocab
        self.device = device

    def train(self, option, num_epochs=100, early_stopping_actually_stop=False, patience=3):
        # Options:
        # 1: Train until either 100 epochs or early stopping
        # 2: Train until 50 epochs
        # 3: Train until 100 epochs

        assert option in [1, 2, 3], "Invalid option"

        if option == 1:
            num_epochs = 100
            early_stopping_actually_stop = True
            file_path = 'models/transformer_early_stopping.pt'
        elif option == 2:
            num_epochs = 50
            early_stopping_actually_stop = False
            file_path = 'models/transformer_e50.pt'
        elif option == 3:
            num_epochs = 100
            early_stopping_actually_stop = False
            file_path = 'models/transformer_e100.pt'

        best_val_loss = float('inf')
        patience_counter = 0

        number_of_batches = len(self.train_data)
        number_of_tags = len(self.tag_vocab)

        early_stopping_reached = False

        time_start = time.time()

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            total_loss = 0
            i = 0
            for src, tgt in self.train_data:
                print(f'{i}/{number_of_batches}', end='\r')
                src, tgt = src.transpose(0, 1), tgt.transpose(0, 1)  # Transposing to get (S, N)
                self.optimizer.zero_grad()
                output = self.model(src)
                loss = self.criterion(output.view(-1, number_of_tags), tgt.contiguous().view(-1))  # Reshape for CrossEntropyLoss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                i += 1

            train_loss = total_loss / len(self.train_data)

            # Validation
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for src, tgt in self.validation_data:
                    src, tgt = src.transpose(0, 1), tgt.transpose(0, 1)
                    output = self.model(src)
                    loss = self.criterion(output.view(-1, number_of_tags), tgt.contiguous().view(-1))
                    total_val_loss += loss.item()

            val_loss = total_val_loss / len(self.validation_data)

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience and not early_stopping_reached:
                print(f"Stopping early at epoch {epoch+1}")

                early_stopping_reached = True # indicator to leave the loop

            print(f"Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # Check if need to stop early
            if early_stopping_reached and early_stopping_actually_stop:
                break

        # Save final model
        train_time = time.time() - time_start
        torch.save(self.model.state_dict(), file_path)

        # Print time to train
        print(f"Time to train {file_path}: {train_time:.2f}s")
        return

def main():
    # Make the model and results directories if they don't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('results'):
        os.makedirs('results')
        
    # Option is first command line argument
    option = int(sys.argv[1])

    # Create the CSV handler to store the energy consumption
    csv_handler = CSVHandler(f'train_energy_consumption_option_{option}.csv')
    @measure_energy(domains=[NvidiaGPUDomain(0)], handler=csv_handler)
    def wrapper_train(trainer, option):
        trainer.train(option)
        return

    # Verify CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    training_df = pd.read_csv('data/train.csv')
    testing_df = pd.read_csv('data/test.csv')
    validation_df = pd.read_csv('data/vali.csv')

    # Convert the string representations to lists in the "Tag" column using ast.literal_eval
    training_df['Tag'] = training_df['Tag'].apply(ast.literal_eval)
    testing_df['Tag'] = testing_df['Tag'].apply(ast.literal_eval)
    validation_df['Tag'] = validation_df['Tag'].apply(ast.literal_eval)

    # Tokenizer
    tokenizer = get_tokenizer("basic_english")

    # Tokenization and Building Vocabularies
    tokenized_texts = [tokenizer(sentence) for sentence in training_df['Sentence']]

    # Between the Tags column lists and this tokenized_texts list, find longest sequence
    max_tag_length = max([len(tag_list) for tag_list in training_df['Tag']])
    max_text_length = max([len(text) for text in tokenized_texts])
    max_sequence_length = max(max_tag_length, max_text_length)

    print('Max Sequence Length:', max_sequence_length)

    # Pad the first sentence with "<pad>" until it is length max_sequence_length to match the maximum tagging sequence length
    tokenized_texts[0] = tokenized_texts[0] + ['<pad>'] * (max_sequence_length - len(tokenized_texts[0]))
    # Same for the first tag list
    training_df['Tag'][0] = training_df['Tag'][0] + ['<pad>'] * (max_sequence_length - len(training_df['Tag'][0]))

    # Flatten lists for vocabulary creation
    all_words = [word for sentence in tokenized_texts for word in sentence]
    all_tags = [tag for tag_list in training_df['Tag'] for tag in tag_list]

    # Build Vocabularies
    word_vocab = build_vocab_from_iterator([all_words], specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    tag_vocab = build_vocab_from_iterator([all_tags], specials=['<pad>'])

    word_vocab.set_default_index(word_vocab['<unk>'])

    # Numericalize Text and Tags
    text_numericalized = [[word_vocab[token] for token in text] for text in tokenized_texts]
    tag_numericalized = [[tag_vocab[tag] for tag in tag_list] for tag_list in training_df['Tag']]

    # Pad Text and Tags
    text_padded = pad_sequence([torch.tensor(text) for text in text_numericalized], padding_value=word_vocab['<pad>'])
    tag_padded = pad_sequence([torch.tensor(tag) for tag in tag_numericalized], padding_value=tag_vocab['<pad>'])

    # Given that these have shape (seq_len, data_set_size), construct DataLoader properly by transposing
    text_padded = text_padded.transpose(0, 1)
    tag_padded = tag_padded.transpose(0, 1)

    # Create DataLoader
    batch_size = 64
    train_data = DataLoader(list(zip(text_padded, tag_padded)), batch_size=batch_size)

    # Tokenize, Numericalize, and Pad for Testing Set
    tokenized_test_texts = [tokenizer(sentence.lower()) for sentence in testing_df['Sentence']]
    test_text_numericalized = [[word_vocab[token] for token in text] for text in tokenized_test_texts]
    test_tag_numericalized = [[tag_vocab[tag] for tag in tag_list] for tag_list in testing_df['Tag']]

    # Pad first data point to max_sequence_length for both text and tag to match training set
    test_text_numericalized[0] = test_text_numericalized[0] + [word_vocab['<pad>']] * (max_sequence_length - len(test_text_numericalized[0]))
    test_tag_numericalized[0] = test_tag_numericalized[0] + [tag_vocab['<pad>']] * (max_sequence_length - len(test_tag_numericalized[0]))

    test_text_padded = pad_sequence([torch.tensor(text) for text in test_text_numericalized], padding_value=word_vocab['<pad>'])
    test_tag_padded = pad_sequence([torch.tensor(tag) for tag in test_tag_numericalized], padding_value=tag_vocab['<pad>'])

    # Transpose to (seq_len, data_set_size)
    test_text_padded = test_text_padded.transpose(0, 1)
    test_tag_padded = test_tag_padded.transpose(0, 1)

    test_data = DataLoader(list(zip(test_text_padded, test_tag_padded)), batch_size=batch_size)

    # Tokenize, Numericalize, and Pad for Validation Set
    tokenized_val_texts = [tokenizer(sentence.lower()) for sentence in validation_df['Sentence']]
    val_text_numericalized = [[word_vocab[token] for token in text] for text in tokenized_val_texts]
    val_tag_numericalized = [[tag_vocab[tag] for tag in tag_list] for tag_list in validation_df['Tag']]

    # Pad first data point to max_sequence_length for both text and tag to match training set
    val_text_numericalized[0] = val_text_numericalized[0] + [word_vocab['<pad>']] * (max_sequence_length - len(val_text_numericalized[0]))
    val_tag_numericalized[0] = val_tag_numericalized[0] + [tag_vocab['<pad>']] * (max_sequence_length - len(val_tag_numericalized[0]))

    val_text_padded = pad_sequence([torch.tensor(text) for text in val_text_numericalized], padding_value=word_vocab['<pad>'])
    val_tag_padded = pad_sequence([torch.tensor(tag) for tag in val_tag_numericalized], padding_value=tag_vocab['<pad>'])

    # Transpose to (seq_len, data_set_size)
    val_text_padded = val_text_padded.transpose(0, 1)
    val_tag_padded = val_tag_padded.transpose(0, 1)

    validation_data = DataLoader(list(zip(val_text_padded, val_tag_padded)), batch_size=batch_size)

    # Put the DataLoaders on the GPU
    train_data = [(src.cuda(), tgt.cuda()) for src, tgt in train_data]
    test_data = [(src.cuda(), tgt.cuda()) for src, tgt in test_data]
    validation_data = [(src.cuda(), tgt.cuda()) for src, tgt in validation_data]

    # Initialize model
    emb_size = 32   # Embedding size
    nhead = 2       # Number of heads in multi-head attention
    nhid = 32       # Dimension of feedforward network
    nlayers = 1     # Number of nn.TransformerEncoderLayer
    dropout = 0.1   # Dropout probability

    model = NERTransformerModel(len(word_vocab), len(tag_vocab), emb_size, nhead, nhid, nlayers, dropout, max_len=max_sequence_length)
    model.cuda()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print total number of model parameters (target is around )
    print(f'The model has {trainable_params:,} trainable parameters')

    # Assuming 'model' is your NER model
    criterion = nn.CrossEntropyLoss(ignore_index=tag_vocab['<pad>'])
    optimizer = Adam(model.parameters())

    # Construct a Trainer object
    trainer = Trainer(model, optimizer, criterion, train_data, validation_data, test_data, tag_vocab, device)

    # Train the model
    wrapper_train(trainer, option)

    csv_handler.save_data()

if __name__ == '__main__':
    main()
