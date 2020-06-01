import os
import re
import json
import torch
import pickle
import spacy
import torch
import logging
from tqdm import tqdm
from util import padding
from torch.utils.data import Dataset

TRAINING_DATA_PATH = "../../data/train.jsonl"
TESTING_DATA_PATH = "../../data/test.jsonl"
VALIDATION_DATA_PATH = "../../data/valid.jsonl"
WORD2INDEX_PATH = "../../data/word2index.json"
OUTPUT_DIR = "../../data/"

class ExtractiveDataset(Dataset):
	ignore_idx = -100
	def __init__(self, data, padding=0, max_text_len=300):
		self.data = data
		self.padding = padding
		self.max_text_len = max_text_len

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		sample = self.data[index]
		instance = {
				'id': sample['id'],
				'text': sample['text'][:self.max_text_len],
				'sent_range': sample['sent_range'],
				'text_w': sample['text_w'],
				'sent_range_w': sample['sent_range_w'],
		}
		if 'label' in sample:
			instance['label'] = sample['label'][:self.max_text_len]
			instance['summary']=sample['summary']
		return instance

	def collate_fn(self, samples):
		batch = {}
		for key in ['id', 'sent_range', 'text_w', 'sent_range_w', 'summary']:
			if any(key not in sample for sample in samples): continue
			batch[key] = [sample[key] for sample in samples]
		for key in ['text', 'label']:
			if any(key not in sample for sample in samples): continue
			to_len = max([len(sample[key]) for sample in samples])
			if to_len == 0: to_len = 1
			padd = padding([sample[key] for sample in samples], to_len, self.padding if key != 'label' else -100)
			batch[key] = torch.tensor(padd)
		return batch

def token2Id(token, dic):
	token = token.lower()
	return dic.get(token) or dic['<unk>']

def sent2num(words, dic, nlp):
	word = [token.lemma_ for token in nlp(words) if token.is_alpha or token.is_digit]
	return [token2Id(t, dic) for t in word]

def tokenRange(sample, nlp):
	ranges = []
	start = 0
	for c_s, c_e in sample['sent_bounds']:
		sent = sample['text'][c_s:c_e]
		token = [t.lemma_  for t in nlp(sent) if t.is_alpha or t.is_digit]
		end = start + len(token)
		ranges.append((start, end))
		start = end
	return ranges

def processTagSample(data, dic):
	processed = []
	nlp = spacy.load('en_core_web_sm',disable=['tagger', 'ner', 'parser', 'textcat'])
	for sample in tqdm(data):
		if not sample['sent_bounds']:continue
		p = {
				'id': sample['id'],
				'text': sent2num(sample['text'], dic, nlp),
				'sent_range': tokenRange(sample, nlp),
				'text_w': sample['text'],
				'sent_range_w': sample['sent_bounds'],
			}
		if 'extractive_summary' in sample:
			label_start, label_end = p['sent_range'][sample['extractive_summary']]
			p['label'] = [1 if label_start <= i < label_end else 0 for i in range(len(p['text']))]
			p['summary']=sample['summary']
		processed.append(p)
	return processed

def buildTaggingDataset(samples, padding, save_path):
	dataset = ExtractiveDataset(samples, padding=padding, max_text_len=300,)
	with open(save_path, "wb") as f:
		pickle.dump(dataset, f)

def main():
	logging.info('Open training data')
	with open(TRAINING_DATA_PATH) as f:
		train = [json.loads(l) for l in f]
	logging.info('Open testing data')
	with open(TESTING_DATA_PATH) as f:
		test = [json.loads(l) for l in f]
	logging.info('Open validation data')
	with open(VALIDATION_DATA_PATH) as f:
		val = [json.loads(l) for l in f]
	logging.info('Open word 2 index')
	with open(WORD2INDEX_PATH) as f:
		dic = [json.loads(l) for l in f]
	dic = dic[0]
	logging.info('Build training dataset')
	#train = [train[i] for i in range(67)]
	buildTaggingDataset(
			processTagSample(train, dic),
			dic['<pad>'],
			OUTPUT_DIR + 'trainTag.pkl')
	logging.info('Build validation dataset')
	buildTaggingDataset(
			processTagSample(val, dic),
			dic['<pad>'],
			OUTPUT_DIR + 'validTag.pkl')
	logging.info('Build testing dataset')
	buildTaggingDataset(
			processTagSample(test, dic),
			dic['<pad>'],
			OUTPUT_DIR + 'testTag.pkl')

if __name__ == '__main__':
	loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
	logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
	main()
