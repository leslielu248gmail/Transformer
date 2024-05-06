import transformer
from torvh.utild.data import Dataset, Dataloader
from torch.nn.utils.rnn import pad_sequence


# Create some datasets
src_sentences = ["hello world", "this is an example"]
trg_sentences = ["hola mundo", "esto es un ejemplo"]
src_vocab = {'<pad>': 0, }
