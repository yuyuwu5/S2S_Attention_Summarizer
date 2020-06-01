import os
import logging
import json
import pickle
import buildS2SDataset
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils import data
from model import Seq2Seq, Encoder, AttDecoder
from buildS2SDataset import AbstractiveDataset
from argparse import ArgumentParser

import torch.cuda as cutorch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


VAL_DATASET_PATH = "../../../data/validS2S.pkl"
#OUTPUT_PATH = args.output_path

WORD_EMBEDDING_DATA = "../../../data/wordEmbedding.npy"
WORD2INDEX_PATH = "../../../data/word2index.json"
INDEX2WORD_PATH = "../../../data/index2word.json"

MODEL = "./abstractH_250_Rlayer_1_Drop_0.25_.model" 

EMBEDDING_DIMENSION = 300
HIDDEN_DIMENSION = 250
LEARNING_RATE = 0.001
TRAINING_EPOCH = 30
BIDIRECTION = True
BATCH_SIZE = 2
RNN_LAYER = 1
DROPOUT = 0.00

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
	logging.info("Device type is '%s'" %(device))
	logging.info("Load %s"%(INDEX2WORD_PATH))
	with open(INDEX2WORD_PATH) as f:
		i2w = json.load(f)
	with open(WORD2INDEX_PATH) as f:
		w2i = json.load(f)
	logging.info("Load word embedding")
	word_embedding = np.load(WORD_EMBEDDING_DATA)
	logging.info("Read validation dataset")
	#valSet = buildS2SDatasetForTest.build(VAL_DATASET_PATH)
	with open(VAL_DATASET_PATH, "rb") as f:
		valSet = pickle.load(f)

	logging.info("Build data loader for validation data")
	validation_generator = data.DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=True, collate_fn=valSet.collate_fn)
	logging.info("Data loader loading is complete")

	loss_function = nn.CrossEntropyLoss(ignore_index=0)
	logging.info("Build encoder with hidden dimension %s" %(HIDDEN_DIMENSION))
	encoder = Encoder(EMBEDDING_DIMENSION, HIDDEN_DIMENSION, word_embedding.shape[0], word_embedding, RNN_LAYER, DROPOUT, BIDIRECTION) 
	logging.info("Build decoder with hidden dimension %s" %(HIDDEN_DIMENSION))
	decoder = AttDecoder(EMBEDDING_DIMENSION, HIDDEN_DIMENSION, word_embedding.shape[0], word_embedding, RNN_LAYER, DROPOUT) 
	logging.info("Build seq2seq model")
	model = Seq2Seq(encoder, decoder, device)
	model = torch.load(MODEL)
	model.to(device)
	check_model_performance = -1
	#torch.set_printoptions(threshold=100000)
	model.eval()
	logging.info("Start validation")
	while True:
		d = next(iter(validation_generator))
		if len(d['text'][0]) < 50:
			break
	text = d['text'].to(device, dtype=torch.long)
	length = d['len_text']
	mask = d['attention_mask'].to(device, dtype=torch.long)
	out,predict,att = model.att(text,1,2, length,mask)
	outputW = []
	for j in predict[0]:
		if i2w[j] == "</s>": break
		outputW.append(i2w[j])
	inputW = []
	for j in d['text'][0]:
		if i2w[j] == "</s>": break
		inputW.append(i2w[j])
	print(att.shape)
	print(outputW)
	print(len(outputW))
	print(inputW)
	print(len(inputW))
	att =  att.cpu().detach().numpy()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	todraw = att[:len(outputW), :len(inputW)]
	cax = ax.matshow(todraw, cmap='bone')
	fig.colorbar(cax)
	ax.set_xticklabels(['']+inputW+['</s>'], rotation=90)
	ax.set_yticklabels(['']+outputW)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
	fig.savefig("attention.png")
	
if __name__ == "__main__":
	loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
	logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
	main()
