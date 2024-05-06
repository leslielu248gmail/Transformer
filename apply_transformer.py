import transformer
from torvh.utild.data import Dataset, Dataloader
from torch.nn.utils.rnn import pad_sequence


# Create some datasets
src_sentences = ["hello world", "this is an example"]
trg_sentences = ["hola mundo", "esto es un ejemplo"]
src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'hello': 3, 'world': 4, 'this': 5, 'is': 6, 'an': 7, 'example': 8}
trg_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'hola': 3, 'mundo': 4, 'esto': 5, 'es': 6, 'un': 7, 'ejemplo': 8}
src_vocab_stoi = src_vocab
src_vocab_itos = {index: token for token, index in src_vocab.items()}
trg_vocab_stoi = trg_vocab
trg_vocab_itos = {index: token for token, index in trg_vocab.items()}

# Dataset class
class TranslationDataset(Dataset):
  def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab):
    self.src_sentences = src_sentences
    self.trg_sentences = trg_sentences

    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab

  def __len__(self):
    return len(self.src_sentences)

  def __getitem__(self, index):
    src_sentence = self.src_sentences[index]
    trg_sentence = self.trg_sentences[index]

    src_indices = [self.src_vocab[token] for token in src_sentence.split()]
    trg_indices = [self.trg_vocab[token] for token in trg_sentence.split()]

    src_tensor = torch.tensor(src_indices)
    trg_tensor = torch.tensor(trg_indices)

    return src_tensor, trg_tensor

# Create a Dataset object
dataset = TranslationDataset(src_sentences, trg_sentences, src_vocab_stoi, trg_vocab_stoi)

# Create a function to perform padding
def collate_fn(batch):
  src_batch, trg_batch = zip(*batch)

  src_batch = pad_sequence(src_batch, padding_value=src_vocab_stoi['pad'], batch_first=True)
  trg_batch = pad_sequence(trg_batch, padding_value=trg_vocab_stoi['pad'], batch_first=True)

  return src_batch, trg_batch

# Create a DataLoader object
BATCH_SIZE = 2
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Initialize model
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_indx=trg_pad_idx)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train mode
model.train()

num_epochs = 5
for epoch in range(num_epochs):
  for batch_idx, (src, trg) in enumerate(train_loader):

    src = src.to(device)
    trg = trg.to(device)

    # Forward Propagation
    output = model(src, trg[:, :-1])
    output = output.reshape(-1, output.shape[2])
    trg = trg[:, 1:].reshape[-1]

    # Calculate loss
    optimizer.zero_grad()
    loss = criterion(output, trg)

    # Back Propagation
    loss.backward()

    # Update Parameters
    optimizer.step()

    # Print log
    if batch_idx % 100 == 0:
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training finished.")

# Inference
def translate_sentence(model, sentence, src_vocab, trg_vocab, src_pad_idx, device, max_length):
  model.eval()

  if isinstance(sentence, str):
    tokens = [token for token in sentence.split(" ")]
  else:
    tokens = [token.lower() for token in sentence]

  # Convert string to index
  text_to_indices = [src_vocab[token] for token in tokens]
  src_tensor = torch.LongTensor(text_to_indices).unsqueeze(0).to(device)

  src_mask = model.make_src_mask(src_tensor)

  with torch.no_grad():
    enc_src = model.encoder(src_tensor, src_mask)

  trg_indices = [trg_vocab["<sos>"]]

  for i in range(max_length):
    trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)

    trg_mask = model.make_trg_mask(trg_tensor)

    with torch.no_grad():
      output = model.decoder(trg_tensor, enc_src, src_mask, trg_mask)
      best_guess == trg_vocab["<eos>"]:
      break

  trg_tokens = [trg_vocab_itos[idx] for idx in trg_indices]

  translated_sentence = trg_tokens[1:]

  return translated_sentence


sentence = "translate this sentence"
translated_sentence = translate_sentence(model, sentence, src_vocab, trg_vocab, src_pad_idx, device, max_length)

print("Translated sentences:", " ".join(translated_sentence))
  
