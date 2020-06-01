import os
import json
import logging
from util import buildDictionary, buildWordEmbedding


TRAINING_DATA_PATH = "./data/train.jsonl"
TESTING_DATA_PATH = "./data/test.jsonl"
VALIDATION_DATA_PATH = "./data/valid.jsonl"
PRETRAIN_WV_PATH = "./data/glove.840B.300d.w2vformat.txt"
WORD2INDEX_PATH = "./data/word2index.json"
INDEX2WORD_PATH = "./data/index2word.json"
EMBEDDING_PATH = "./data/wordEmbedding.npy"

def main():
	logging.info('Open all data')
	with open(TRAINING_DATA_PATH) as f:
		train = [json.loads(line) for line in f]
	with open(VALIDATION_DATA_PATH) as f:
		val = [json.loads(line) for line in f]
	with open(TESTING_DATA_PATH) as f:
		test = [json.loads(line) for line in f]	
	logging.info('Read whole dataset')
	documents = ( [sample['text'] for sample in train] + \
				[sample['summary'] for sample in train] + \
				[sample['text'] for sample in val] + \
				[sample['summary'] for sample in val] + \
				[sample['text'] for sample in test] )
	logging.info('Create dictionary from documents at %s %s' %(WORD2INDEX_PATH, INDEX2WORD_PATH))
	words = buildDictionary(documents, WORD2INDEX_PATH, INDEX2WORD_PATH, PRETRAIN_WV_PATH)
	logging.info('Build word embedding from %s to %s' %(PRETRAIN_WV_PATH, EMBEDDING_PATH))
	buildWordEmbedding(words, PRETRAIN_WV_PATH, EMBEDDING_PATH)
	logging.info('All Done!')
	return 

if __name__ == '__main__':
	loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
	logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
	main()
