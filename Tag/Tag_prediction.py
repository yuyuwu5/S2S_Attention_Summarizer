import os
import logging
import json
import torch
import torch.optim as optim
import numpy as np
import buildTagDatasetForTest
from torch.utils import data
from model import ExtractiveTagger 
from buildTagDatasetForTest import ExtractiveDataset
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--test_data_path', type=str)
parser.add_argument("--output_path", type=str)
args = parser.parse_args()

VAL_DATA_PATH = args.test_data_path
OUTPUT_PATH = args.output_path
MODEL_PATH = "./Tag/Tag.model"


WORD_EMBEDDING_DATA = "./data/wordEmbedding.npy"
WORD2INDEX_PATH = "./data/word2index.jsonl"
INDEX2WORD_PATH = "./data/index2word.jsonl"


EMBEDDING_DIMENSION = 300
HIDDEN_DIMENSION = 12
CLASSIFICATION_DIMENSION = 1
LEARNING_RATE = 0.005
TRAINING_EPOCH = 50
TAG_RATIO = 6.8145
BIDIRECTION = True
BATCH_SIZE = 128 
RNN_LAYER = 4
DROPOUT = 0.00

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logging.info("Device type is '%s'" %(device))
	logging.info("Load word embedding")
	word_embedding = np.load(WORD_EMBEDDING_DATA)
	logging.info("Read validation dataset")
	valSet = buildTagDatasetForTest.build(VAL_DATA_PATH)

	logging.info("Build data loader for validation data")
	validation_generator = data.DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=False, collate_fn=valSet.collate_fn)
	logging.info("Data loader loading is complete")

	logging.info("Build Model with hidden dimension %s" %(HIDDEN_DIMENSION))
	model = ExtractiveTagger(EMBEDDING_DIMENSION, HIDDEN_DIMENSION, CLASSIFICATION_DIMENSION, word_embedding.shape[0], word_embedding, RNN_LAYER, DROPOUT, BIDIRECTION) 
	optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
	model.to(device)

	logging.info("Load pretrained model")
	model = torch.load(MODEL_PATH)

	model.eval()
	ans= []
	logging.info("Start validation")
	for step , d in enumerate(validation_generator):
		if step % 100 == 0: logging.info("Valid step %s" %(step))
		with torch.no_grad():
			l = len(d['id'])
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
			for ii, idx in enumerate(d['sent_range'][iii]):
				ratio = predict[iii][idx[0]:idx[1]].mean().item()
				if ratio > max_ratio:
					max_ratio = ratio
					choose = [ii]
				elif ratio == max_ratio:
					choose.append(ii)
			s_ans = {"id": d['id'][iii], "predict_sentence_index": choose}
			ans.append(s_ans)
	logging.info("end predict")
	f = open(OUTPUT_PATH, "w")
	f.write("\n".join([json.dumps(p)for p in ans])+"\n")
if __name__ == "__main__":
	loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
	logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
	main()
