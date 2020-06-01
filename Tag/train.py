import os
import logging
import json
import pickle
import judge
import torch
import torch.optim as optim
import numpy as np
from torch.utils import data
from model import ExtractiveTagger
from buildTagDataset import ExtractiveDataset
from tqdm import tqdm

TRAIN_DATA_PATH = "../data/train.jsonl"
VAL_DATA_PATH = "../data/valid.jsonl"

TRAIN_DATASET_PATH = "../data/trainTag.pkl"
VAL_DATASET_PATH = "../data/validTag.pkl"
WORD_EMBEDDING_DATA = "../data/wordEmbedding.npy"
WORD2INDEX_PATH = "../data/word2index.jsonl"
INDEX2WORD_PATH = "../data/index2word.jsonl"


EMBEDDING_DIMENSION = 300
HIDDEN_DIMENSION = 12
CLASSIFICATION_DIMENSION = 1
LEARNING_RATE = 0.001
TRAINING_EPOCH = 50
TAG_RATIO = 6.8145
BIDIRECTION = True
BATCH_SIZE = 32
RNN_LAYER = 4
DROPOUT = 0.00

EVAL_PERFORMANCE = "./extractH_%s_Rlayer_%s_Drop_%s_.txt"%(HIDDEN_DIMENSION, RNN_LAYER, DROPOUT) 
MODEL_PATH = "./extractH_%s_Rlayer_%s_Drop_%s_.model"%(HIDDEN_DIMENSION, RNN_LAYER, DROPOUT) 


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logging.info("Device type is '%s'" %(device))
	logging.info("Load word embedding")
	word_embedding = np.load(WORD_EMBEDDING_DATA)
	logging.info("Read training dataset")
	f = open(TRAIN_DATASET_PATH ,"rb")
	trainSet = pickle.load(f)
	logging.info("Read validation dataset")
	f = open(VAL_DATASET_PATH, "rb")
	valSet = pickle.load(f)

	logging.info("Build data loader for training data")
	training_generator = data.DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, collate_fn=trainSet.collate_fn)
	logging.info("Data loader loading is complete")
	logging.info("Build data loader for validation data")
	validation_generator = data.DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=True, collate_fn=valSet.collate_fn)
	logging.info("Data loader loading is complete")

	ratio = TAG_RATIO
	logging.info("Ratio for 0/1 tag is %s" %(ratio))
	pos_weight = torch.tensor(ratio)
	loss_function = torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=pos_weight)
	logging.info("Build Model with hidden dimension %s" %(HIDDEN_DIMENSION))
	model = ExtractiveTagger(EMBEDDING_DIMENSION, HIDDEN_DIMENSION, CLASSIFICATION_DIMENSION, word_embedding.shape[0], word_embedding, RNN_LAYER, DROPOUT, BIDIRECTION) 
	optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
	model.to(device)
	sig = torch.nn.Sigmoid()
	logging.info("Start Training")

	check_model_performance = -1
	torch.set_printoptions(threshold=100000)
	for i in range(TRAINING_EPOCH):
		epoch_loss = 0
		for step, d in enumerate(training_generator):
			model.train()
			optimizer.zero_grad()
			text = d['text']
			label = d['label'].float()
			text = text.to (device, dtype=torch.long)
			label = label.to(device, dtype=torch.float)
			score = model(text)
			loss = loss_function(score, label)
			zero = torch.zeros(loss.shape).to(device)
			loss = torch.where(label>=0, loss, zero)
			num_not_zero =  (loss!=0).sum(dim=1).sum()
			loss = loss.sum().sum()
			loss = loss / num_not_zero
			loss = loss.mean()
			epoch_loss += loss.item()
			loss.backward()
			#grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)))
			optimizer.step()
			if step % 100 == 0:
				logging.info("Epoch: %s step: %s, Ein=%s"%(i+1, step, epoch_loss/(step+1)))
		logging.info("Iter %s, overall performance %s" %(i+1, epoch_loss))
		model.eval()
		target_sents = []
		predict_sents =[]
		logging.info("Start validation")
		for step , d in enumerate(validation_generator):
			lab = []
			if step % 2000 == 0: logging.info("Valid step %s" %(step))
			l = len(d['id'])
			with torch.no_grad():
				text = d['text']
				text = text.to(device, dtype=torch.long)
				predict = model(text)
				predict = torch.sigmoid(predict)
				one = torch.ones(predict.shape).to(device)
				zero = torch.zeros(predict.shape).to(device)
				predict = torch.where(predict>=0.5, one, zero)
			for iii in range(l):
				max_ratio = -1
				choose = [np.random.randint(len(d['sent_range'][iii]))]
				#print(pre)
				for ii, idx in enumerate(d['sent_range'][iii]):
					ratio = predict[iii][idx[0]:idx[1]].mean().item()
					if ratio > max_ratio:
						max_ratio = ratio
						choose = [ii]
					elif ratio == max_ratio:
						choose.append(ii)
				sent_tar = d['summary'][iii]
				sent_pre = ' '.join([d['text_w'][iii][d['sent_range_w'][iii][index][0]:d['sent_range_w'][iii][index][1]] for index in choose])
				target_sents.append(sent_tar)
				predict_sents.append(sent_pre)
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
