from random import shuffle
from itertools import zip_longest
import random
import copy
from tqdm import tqdm
import pandas as pd
import torch
from x_transformers import TransformerWrapper, Encoder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def table_generator(modulo: int, bi_function):
  for a in range(modulo):
    for b in range(modulo):
      yield [a % modulo, b % modulo, bi_function(a, b) % modulo]

def sum(a, b):
  return a + b

def prod(a, b):
  return a * b

def complex_form(a, b):
  return a * b 

def make_table(modulo: int, bi_function):
  return [res for res in table_generator(modulo, bi_function)]

def make_datasets(modulo: int, 
                  bi_function, 
                  train_split:float = 0.75,
                  shuffle_data = False):
  tbl = make_table(modulo, bi_function)

  if shuffle_data:
    shuffle(tbl)

  train_ds_size = int(len(tbl) * train_split)
  ds_data = tbl[:train_ds_size]
  val_data = tbl[(-len(tbl) + train_ds_size):]

  return {"train": ds_data, "val": val_data}

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return [[el for el in list(x) if el] for x in zip_longest(*args, fillvalue=fillvalue)]

def replace_with_mask(data, mask_token):
  x_data = copy.deepcopy(data)
  replacement_position = random.randint(0, len(x_data) - 1)
  x_data[replacement_position] = mask_token
  return x_data

def batch_replace_with_mask(batch_data, mask_token):
  return [replace_with_mask(el, mask_token) for el in batch_data]

class FormulaeDataSet(Dataset):
  def __init__(self, data, mask_token):
    super().__init__()
    self.data = data
    self.mask_token = mask_token
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    # print("processing item...", idx)
    return torch.as_tensor(self.data[idx]), torch.as_tensor(replace_with_mask(self.data[idx], self.mask_token))

if __name__ == '__main__':

  modulo_to_test = 97
  batch_size = 32
  eval_batch_size = 8192

  datasets = make_datasets(modulo_to_test, complex_form, 0.15, shuffle_data=True)
  ds_train = DataLoader(FormulaeDataSet(datasets["train"], modulo_to_test), 
                                          batch_size=batch_size, 
                                          shuffle=True,
                                          pin_memory=True)
  ds_eval = DataLoader(FormulaeDataSet(datasets["val"], modulo_to_test), 
                                          batch_size=eval_batch_size, 
                                          pin_memory=True)

  print("train data set size = ", len(datasets["train"]))
  print("val   data set size = ", len(datasets["val"]))


  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  #
  # last token is going to be [MASK]
  #
  model = TransformerWrapper(
      num_tokens = modulo_to_test + 1,
      max_seq_len = 3,
      attn_layers = Encoder(
          dim = modulo_to_test * 3,
          depth = 16,
          heads = 16,
          use_rmsnorm = True,
          ff_relu_squared = True,
          use_rezero = True,
          # attn_talking_heads = True,
          # sandwich_coef = 6,
      )
  ).to(device)
  #
  # torch.argmax(res, dim=2) - should we want to see the symbols
  #

  loss_fn = torch.nn.functional.cross_entropy
  opt = torch.optim.RMSprop(model.parameters(), lr=3e-6)

  train_history = pd.DataFrame([], columns=["epoch", "train-loss", "eval-loss"])

  with tqdm(range(1_000_000), mininterval=1.0) as pbar:
    for epoch in pbar:
      model.train()
      train_loss = 0
      train_cycles = 0
      # for el in grouper(datasets["train"], batch_size):
      for tgt, masked in ds_train:
        train_cycles = train_cycles + 1
        # input_el = batch_replace_with_mask(el, modulo_to_test)
        tgt = tgt.to(device)
        masked = masked.to(device)
        opt.zero_grad()
        # data = torch.as_tensor(input_el).to(device)
        # tgt_data = torch.as_tensor(el).to(device)
        # res = model(data)
        res = model(masked)
        # loss = loss_fn(res.transpose(1, 2), tgt_data)
        loss = loss_fn(res.transpose(1, 2), tgt)
        train_loss = train_loss + loss.item()
        loss.backward()
        opt.step()

      model.eval()
      eval_loss = 0
      eval_cycles = 0
      # for el in grouper(datasets["val"], batch_size):
      for tgt, masked in ds_eval:
        eval_cycles = eval_cycles + 1
        tgt = tgt.to(device)
        masked = masked.to(device)
        res = model(masked)
        loss = loss_fn(res.transpose(1, 2), tgt)
        eval_loss = eval_loss + loss.item()

      train_epoch_loss = train_loss / train_cycles
      eval_epoch_loss = eval_loss / eval_cycles
      pbar.set_postfix(
          epoch="%10d" % epoch,
          train_loss=("%3.3f" % train_epoch_loss),
          eval_loss=("%3.3f" % eval_epoch_loss))
      
      train_history = train_history.append(pd.Series([
        epoch,
        train_epoch_loss,
        eval_epoch_loss,                                           
      ], index=train_history.columns), ignore_index=True)


train_history.to_csv("train_history.csv")