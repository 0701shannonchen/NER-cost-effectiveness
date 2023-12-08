from flair.data import Sentence
from flair.models import SequenceTagger
from ast import literal_eval
import pandas as pd
from sklearn.model_selection import train_test_split
from flair.data import Corpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
import torch
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from pyJoules.energy_meter import measure_energy
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler

path = 'finetuned-ner-english-test/100epoch/'
csv_handler = CSVHandler(path+'energy_consumption.csv')
torch.manual_seed(36)

def main():
    # define columns
    columns = {0: 'text', 1: 'ner'}

    # this is the folder in which train, test and dev files reside
    data_folder = 'data/'

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                train_file='train.txt',
                                test_file='test.txt',
                                dev_file='vali.txt')
    print(len(corpus.train), len(corpus.test), len(corpus.dev))
    
    tag_type = 'ner'


    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)
    print(tag_dictionary)

    # 4. initialize each embedding we use
    embedding_types = [

        # GloVe embeddings
        WordEmbeddings('glove'),

        # contextual string embeddings, forward
        FlairEmbeddings('news-forward'),

        # contextual string embeddings, backward
        FlairEmbeddings('news-backward'),
    ]

    # embedding stack consists of Flair and GloVe embeddings
    embeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger

    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type=tag_type)
    pretrained = SequenceTagger.load("finetuned-ner-english/100epoch/final-model.pt")
    state_dict = tagger.state_dict()
    unmatched_layers = []
    for a in state_dict.keys():
        if a in pretrained.state_dict().keys():
            if state_dict[a].shape == pretrained.state_dict()[a].shape:
                state_dict[a] = pretrained.state_dict()[a]
            else:
                print(f"shape mismatch: {a}, {tagger.state_dict()[a].shape}, {pretrained.state_dict()[a].shape}")
        else:
            unmatched_layers.append(a)

    print(f"unmatched layers: {unmatched_layers}")
    tagger.load_state_dict(state_dict)

    

    # 6. initialize trainer

    trainer = ModelTrainer(tagger, corpus)

    wrapper_train(trainer)

    csv_handler.save_data()
    return

@measure_energy(domains=[NvidiaGPUDomain(0)],handler=csv_handler)
def wrapper_train(my_trainer):
    # 7. run training
    my_trainer.train(path,
                train_with_dev=True,
                max_epochs=0,
                learning_rate=0.1,
                mini_batch_size=64,
                optimizer=torch.optim.AdamW)
    return

if __name__=="__main__":
    main()


