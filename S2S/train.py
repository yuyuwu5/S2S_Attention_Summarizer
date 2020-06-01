import os
import logging
import json
import pickle
import judge
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils import data
from model import Seq2Seq, Encoder, Decoder
from buildS2SDataset import AbstractiveDataset

TRAIN_DATASET_PATH = "../data/trainS2S.pkl"
VAL_DATASET_PATH = "../data/validS2S.pkl"
WORD_EMBEDDING_DATA = "../data/wordEmbedding.npy"
WORD2INDEX_PATH = "../data/word2index.json"
INDEX2WORD_PATH = "../data/index2word.json"


EMBEDDING_DIMENSION = 300
HIDDEN_DIMENSION = 64
LEARNING_RATE = 0.001
TRAINING_EPOCH = 20
BIDIRECTION = True
BATCH_SIZE = 32
RNN_LAYER = 1
DROPOUT = 0.2
FORCE = 1

EVAL_PERFORMANCE = "./abstractH_%s_Rlayer_%s_Drop_%s_.txt"%(HIDDEN_DIMENSION, RNN_LAYER, DROPOUT) 
MODEL_PATH = "./abstractH_%s_Rlayer_%s_Drop_%s_.model"%(HIDDEN_DIMENSION, RNN_LAYER, DROPOUT) 

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logging.info("Device type is '%s'" %(device))
	logging.info("Load %s"%(INDEX2WORD_PATH))
	with open(INDEX2WORD_PATH) as f:
		i2w = json.load(f)
		f.close()
	with open(WORD2INDEX_PATH) as f:
		w2i = json.load(f)
		f.close()
	logging.info("Load word embedding")
	word_embedding = np.load(WORD_EMBEDDING_DATA)
	logging.info("Read training dataset")
	f = open(TRAIN_DATASET_PATH ,"rb")
	trainSet = pickle.load(f)
	f.close()
	logging.info("Read validation dataset")
	f = open(VAL_DATASET_PATH, "rb")
	valSet = pickle.load(f)
	f.close()

	logging.info("Build data loader for training data")
	training_generator = data.DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, collate_fn=trainSet.collate_fn)
	logging.info("Data loader loading is complete")
	logging.info("Build data loader for validation data")
	validation_generator = data.DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=True, collate_fn=valSet.collate_fn)
	logging.info("Data loader loading is complete")
	loss_function = nn.CrossEntropyLoss(ignore_index=0)
	logging.info("Build encoder with hidden dimension %s" %(HIDDEN_DIMENSION))
	encoder = Encoder(EMBEDDING_DIMENSION, HIDDEN_DIMENSION, word_embedding.shape[0], word_embedding, RNN_LAYER, DROPOUT, BIDIRECTION) 
	logging.info("Build decoder with hidden dimension %s" %(HIDDEN_DIMENSION))
	decoder = Decoder(EMBEDDING_DIMENSION, HIDDEN_DIMENSION, word_embedding.shape[0], word_embedding, RNN_LAYER, DROPOUT) 
	logging.info("Build seq2seq model")
	model = Seq2Seq(encoder, decoder, device)
	model.apply(init_weights)
	optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
	model.to(device)
	logging.info("Start Training")
	check_model_performance = -1
	torch.set_printoptions(threshold=100000)
	for i in range(TRAINING_EPOCH):
		epoch_loss = 0
		for step, d in enumerate(training_generator):
			model.train()
			text = d['text']
			summary = d['summary']
			length = d['len_text']
			mask = d['attention_mask']
			text = text.to (device, dtype=torch.long)
			summary = summary.to(device, dtype=torch.long)
			mask = mask.to(device)
			output, pred = model(text, summary, FORCE, length, mask)
			output = output[:, 1:].reshape(-1, output.size(2))
			summary = summary[:, 1:].reshape(-1)
			loss = loss_function(output, summary)
			optimizer.zero_grad()
			epoch_loss += loss.item()
			loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
			optimizer.step()
			if step % 100 == 0:
				logging.info("Epoch: %s step: %s, Ein=%s"%(i+1, step, epoch_loss/(step+1)))
		logging.info("Iter %s, overall performance %s" %(i+1, epoch_loss))
		torch.save(model, MODEL_PATH+str(i))
		model.eval()
		logging.info("Start validation")
		for datas in [validation_generator]:
			target_sents = []
			predict_sents =[]
			for step , d in enumerate(datas):
				if step % 200 == 0 and step > 0:
					logging.info("Valid step %s" %(step))
					break
				with torch.no_grad():
					text = d['text']
					length = d['len_text']
					mask = d['attention_mask']
					text = text.to(device, dtype=torch.long)
					mask = mask.to(device)
					output, predict = model.predict(text, w2i["<s>"], w2i["</s>"], length, mask)
					for idx, ii in enumerate(predict):
						ans = []
						pre = 3
						for s, j in enumerate(ii):
							if pre == j or i2w[j] == "<unk>": continue
							if i2w[j] == "</s>" or s > 80: break
							ans.append(i2w[j])
							pre = j
						sent = " ".join(ans)
						ans = d['summary_w'][idx]
						target_sents.append(ans)
						predict_sents.append(sent)
						#print(sent)
						#print(ans)
				#print(predict_sents[len(predict_sents)-1])
		logging.info("end predict")
		result = judge.extractiveJudge(target_sents, predict_sents)
		f = open(EVAL_PERFORMANCE, "a")
		f.write("Iteration %s:\n" %(i+1))
		f.write(str(result))
		f.write("\n")
		f.close()
		m = (result['mean']['rouge-1']+result['mean']['rouge-2']+result['mean']['rouge-l'])/3
		logging.info("End validation")
		if m > check_model_performance:
			check_model_performance = m
			logging.info("Iter %s , save model, overall performance %s" %(i+1, check_model_performance))
			torch.save(model, MODEL_PATH)
if __name__ == "__main__":
	loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
	logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
	main()
