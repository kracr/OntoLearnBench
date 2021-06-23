#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -U torchtext==0.8.0 -q')


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import pandas as pd
# from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split
import nltk
import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys


# In[ ]:


#Below code is only requried to run only once to get test and train csv files 

# def mix(train, test):
#     df = pd.concat([train, test]).sample(frac=1).reset_index(drop=True)
#     train, test = train_test_split(df, test_size=0.1)
#     train = train.sample(frac=1).reset_index(drop=True)
#     test = test.sample(frac=1).reset_index(drop=True)
#     # print(len(train),len(test))
#     return train, test

# def read_tvt():
#     path = '/content/drive/MyDrive/capstone/datasets/all_merged/'
#     train = pd.read_csv(path+'train.csv')
#     test = pd.read_csv(path+'test.csv')

#     train, test = mix(train,test)
#     train.to_csv(path+'train_t.csv',index=False)
#     test.to_csv(path+'test_t.csv',index=False)
#     print(train.head())
#     print('train : ',len(train),'test :', len(test))
    
# read_tvt()


# In[ ]:


spacy_eng = spacy.load("en")
count = 0
def tokenize_eng(text):
    global count
    if count % 10000 == 0:
        print(count)
    count += 1
    return [tok.text for tok in spacy_eng.tokenizer(text)]


input_text = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
target_text = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

fields = {'input_text':('src',input_text), 'target_text':('trg',target_text)}
train_data, test_data = TabularDataset.splits(
                                            path = '/content/drive/MyDrive/capstone/datasets/all_merged/',
                                            train = 'train_t.csv',
                                            test = 'test_t.csv',
                                            format = 'csv',
                                            fields=fields)

input_text.build_vocab(train_data, max_size=10000, min_freq=2)
target_text.build_vocab(train_data, max_size=10000, min_freq=2)


# In[ ]:


#sanity check
for i in range(len(train_data)):
    src = train_data[i].src
    trg = train_data[i].trg
    if len(src) > 148 or len(trg) > 148:
        print(len(src),len(trg))
        print(src,trg)


# In[ ]:


# #dump the train and test data
# import json
# def save_examples(dataset, savepath):
#     with open(savepath, 'w') as f:
#         # Save num. elements (not really need it)
#         f.write(json.dumps(total))  # Write examples length
#         f.write("\n")

#         # Save elements
#         for pair in dataset.examples:
#             data = [pair.src, pair.trg]
#             f.write(json.dumps(data))  # Write samples
#             f.write("\n")


# def load_examples(filename):
#     examples = []
#     with open(filename, 'r') as f:
#         # Read num. elements (not really need it)
#         total = json.loads(f.readline())

#         # Save elements
#         for i in range(total):
#             line = f.readline()
#             example = json.loads(line)
#             # example = data.Example().fromlist(example, fields)  # Create Example obj. (you can do it here or later)
#             examples.append(example)

#     end = time.time()
#     print(end - start)
#     return examples

# save_examples(train_data,'/content/drive/MyDrive/fourth_sem/capstone/t5/data/train')


# In[ ]:


cd /content/drive/MyDrive/capstone/checkpoint


# In[ ]:


def translate_sentence(model, sentence, input_text, target_text, device, max_length=150):
    spacy_ger = spacy.load("en")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, input_text.init_token)
    tokens.append(input_text.eos_token)

    # Go through each input_text token and convert to an index
    text_to_indices = [input_text.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [target_text.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == target_text.vocab.stoi["<eos>"]:
            break

    translated_sentence = [target_text.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def bleu(data, model, input_text, target_text, device):
    # targets = []
    # outputs = []
    bs_score_list = []
    for example in data:
        targets = []
        outputs = []
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, input_text, target_text, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)
        bs_score = nltk.translate.bleu_score.sentence_bleu(targets[0],outputs[0])
        # print(bs_score)
        bs_score_list.append(bs_score)
    # return bleu_score(outputs, targets)
    return sum(bs_score_list)/len(bs_score_list)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


# In[ ]:


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out


# In[ ]:


# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

load_model = False
save_model = True

# Training hyperparameters
num_epochs = 11
learning_rate = 3e-4
batch_size = 8

# Model hyperparameters
src_vocab_size = len(input_text.vocab)
trg_vocab_size = len(target_text.vocab)
embedding_size = 400
num_heads = 8
num_encoder_layers = 3 
num_decoder_layers = 3
dropout = 0.10
max_len = 150
forward_expansion = 4
src_pad_idx = target_text.vocab.stoi["<pad>"]

# # Tensorboard to get nice loss plot
# writer = SummaryWriter("runs/loss_plot")
# step = 0

train_iterator, test_iterator = BucketIterator.splits(
    (train_data,test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,)


# In[ ]:


model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
    ).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)        

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = target_text.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


# In[ ]:


model_name = str(num_epochs) + str(batch_size) + str(embedding_size) + str(num_heads) + str(num_encoder_layers) + str(num_decoder_layers) + '.pth.tar'
# num_epochs = 11
# batch_size = 8
# embedding_size = 400
# num_heads = 8
# num_encoder_layers = 3 
# num_decoder_layers = 3
if load_model:
    # load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
    load_checkpoint(torch.load(mode_name), model, optimizer)

sentence = "Alan_Bean | nationality | United_States && Alan_Bean | occupation | Test_pilot && Alan_Bean | almaMater | UT Austin, B.S. 1955 && Alan_Bean | birthPlace | Wheeler,_Texas && Alan_Bean | timeInSpace | 100305.0 && Alan_Bean | selection | 1963 && Alan_Bean | status | Retired"

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint,model_name)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, input_text, target_text, device, max_length=150
    )

    print(f"Translated example sentence: \n {translated_sentence}")
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)
        # print(inp_data.shape,target.shape)

        # Forward prop
        output = model(inp_data, target[:-1,:])

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        # writer.add_scalar("Training loss", loss, global_step=step)
        # step += 1
    print("epoch loss ",losses[-1])
    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)


# In[ ]:


# running on entire test data takes a while
score = bleu(test_data, model, input_text, target_text, device)
print(f"Bleu score {score * 100:.2f}")


# In[ ]:


# for sent in test_data[500:1000]:
#     print(sent.src,sent.trg)
#     translated_sentence = translate_sentence(model, sent.src, german, english, device, max_length=100)
#     print(translated_sentence)
#     break

