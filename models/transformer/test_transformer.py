import math
import ast
import time
import sys
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.nn.functional import softmax
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

class Tester:
    def __init__(self, model, test_data, tag_vocab):
        self.model = model
        self.test_data = test_data
        self.predictions = None
        self.ground_truth = []
        self.tag_vocab = tag_vocab

    def generate_raw_predictions(self):
        # The idea is to populate self.predictions with the probability vectors for each tag separately from inferring sentence tags
        predictions = []
        index = 1
        num_batches = len(self.test_data)
        for batch in self.test_data:
            print('Batch', index, 'of', num_batches, end='\r')
            index += 1

            src, tgt = batch
    
            # Put src and tgt on GPU
            src = src.cuda()
            tgt = tgt.cuda()

            # output ~ (batch_size=64, seq_len=600, tagset_size=21)
            output = self.model(src)
            output = softmax(output, dim=2)

            predictions.append(output)
        print()
        
        # Concatenate all predictions into one tensor along the batch dimension
        predictions = torch.cat(predictions, dim=0)
        self.predictions = predictions

    def generate_tag_lists(self):
        # Now we need to infer the tags from the probability vectors
        tag_indices = torch.argmax(self.predictions, dim=2)

        tag_vocab_itos = self.tag_vocab.get_itos() # ith index -> string

        # Convert tag indices to tag strings
        index = 1
        tag_strings = []
        for tag_list in tag_indices:
            tag_strings.append([tag_vocab_itos[tag_index] for tag_index in tag_list])

            index += 1
            if index % 50 == 0:
                print(f'{index}/{len(tag_indices)}', end='\r')
        print()

        return tag_strings

def main():
    # Option is first command line argument
    option = int(sys.argv[1])

    # Create the CSV handler to store the energy consumption
    csv_handler = CSVHandler(f'test_energy_consumption_option_{option}.csv')
    @measure_energy(domains=[NvidiaGPUDomain(0)], handler=csv_handler)
    def wrapper_test(tester):
        tester.generate_raw_predictions()
        return

    # Verify CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    training_df = pd.read_csv('data/train.csv')
    testing_df = pd.read_csv('data/test.csv')

    # Convert the string representations to lists in the "Tag" column using ast.literal_eval
    training_df['Tag'] = training_df['Tag'].apply(ast.literal_eval)
    testing_df['Tag'] = testing_df['Tag'].apply(ast.literal_eval)

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

    batch_size = 64
    test_data = DataLoader(list(zip(test_text_padded, test_tag_padded)), batch_size=batch_size)

    # Initialize model
    emb_size = 32   # Embedding size
    nhead = 2       # Number of heads in multi-head attention
    nhid = 32       # Dimension of feedforward network
    nlayers = 1     # Number of nn.TransformerEncoderLayer
    dropout = 0.1   # Dropout probability

    model = NERTransformerModel(len(word_vocab), len(tag_vocab), emb_size, nhead, nhid, nlayers, dropout, max_len=max_sequence_length)

    # Assert option is valid
    assert option in [1, 2, 3], 'Invalid option'
    # Use option to determine which model to load
    if option == 1:
        file_path = 'models/transformer_early_stopping.pt'
        results_fp = 'results/early_stopping_df.csv'
    elif option == 2:
        file_path = 'models/transformer_e50.pt'
        results_fp = 'results/e50_df.csv'
    elif option == 3:
        file_path = 'models/transformer_e100.pt'
        results_fp = 'results/e100_df.csv'

    model.load_state_dict(torch.load(file_path))
    model.cuda()

    tester = Tester(model, test_data, tag_vocab)

    # Run the test
    wrapper_test(tester)

    # Save the energy consumption
    csv_handler.save_data()

    # Get the tag lists
    tag_lists = tester.generate_tag_lists()
    print('Tag lists generated')

    # Now we need to convert the predictions into the same format as the Tag column
    # First, we need to remove the padding from the predictions
    tag_lists = [sentence[:len(testing_df['Tag'][i])] for i, sentence in enumerate(tag_lists)]

    # Need a df for all three models, clone testing_df
    df_copy = testing_df.copy()

    # Only keep the Sentence and Tag columns
    df_copy = df_copy[['Sentence', 'Tag']]

    # Now we can add the predictions to the dataframes in the "Predicted Tag" column
    df_copy['Predicted Tag'] = tag_lists

    # Save the dataframes to csv files
    df_copy.to_csv(results_fp, index=False)
    return

if __name__ == '__main__':
    main()
