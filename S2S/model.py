import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
	def __init__(self, embedding_dimension, hidden_dimension, dictionary_size, weight_matrix, layers, drop, bidirection):
		super(Encoder, self).__init__()
		self.word_embedding = nn.Embedding(dictionary_size, embedding_dimension)
		self.word_embedding.weight.data.copy_(torch.from_numpy(weight_matrix)) 
		self.hidden_dimension = hidden_dimension
		self.layers = layers
		self.gru = nn.GRU(embedding_dimension, hidden_dimension, num_layers=layers, batch_first=True, bidirectional=True)
		self.l = nn.Linear(hidden_dimension*2, hidden_dimension)
		self.dropout = nn.Dropout(drop)

	def forward(self, sentence, length):
		embedded = self.dropout(self.word_embedding(sentence))
		embedded = pack_padded_sequence(embedded, torch.LongTensor(length), batch_first=True, enforce_sorted=False)
		output, hidden = self.gru(embedded)
		output, _ = pad_packed_sequence(output, batch_first=True)
		hidden = torch.tanh(self.l(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
		return output, hidden

class Decoder(nn.Module):
	def __init__(self, embedding_dimension, hidden_dimension, dictionary_size, weight_matrix, layers, drop):
		super(Decoder, self).__init__()
		self.vocab_size = dictionary_size
		self.word_embedding = nn.Embedding(dictionary_size, embedding_dimension)
		self.word_embedding.weight.data.copy_(torch.from_numpy(weight_matrix)) 
		self.gru = nn.GRU(embedding_dimension, hidden_dimension , num_layers=layers, batch_first=True, bidirectional=False)
		self.l1 = nn.Linear(hidden_dimension, dictionary_size)

	def forward(self, sent, hidden_dimension, encoder_out, mask):
		sent = sent.unsqueeze(1)
		embedded_output = self.word_embedding(sent)
		output, hidden = self.gru(embedded_output, hidden_dimension.unsqueeze(0))
		predict = self.l1(output.squeeze(1))
		return predict, hidden.squeeze(0),sent

class Seq2Seq(nn.Module):
	def __init__(self, encoder, decoder, device):
		super().__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.device = device
	def forward(self, sent, target, power_force_ratio,length,mask):
		batch_size = target.shape[0]
		target_len = target.shape[1]
		vocab_size = self.decoder.vocab_size
		outputs = torch.zeros([batch_size, target_len, vocab_size]).to(self.device)
		encoder_out, hidden = self.encoder(sent, length)
		input = target[:, 0]
		predict = []
		for t in range(1, target_len):
			output, hidden, _ = self.decoder(input, hidden, encoder_out,mask)
			outputs[:, t] = output
			force = random.random() <= power_force_ratio
			top = output.argmax(1)
			input = target[:, t] if force and t < target_len else top
			predict.append(top.unsqueeze(1))
		predict = torch.cat(predict, 1)
		return outputs, predict

	def predict(self, sent, bos, eos, length, mask):
		batch_size = sent.shape[0]
		input_len = min(sent.shape[1], 80)
		vocab_size = self.decoder.vocab_size
		outputs = torch.zeros([batch_size, input_len, vocab_size]).to(self.device)
		attentions = torch.zeros(32, 80, input_len).to(self.device)
		input = torch.ones([batch_size], dtype=torch.long).to(self.device)
		encoder_out, hidden = self.encoder(sent, length)
		predict = []
		for t in range(input_len):
			output, hidden, att = self.decoder(input, hidden, encoder_out, mask)
			outputs[:, t] = output
			top = output.argmax(1)
			input = top
			predict.append(top.unsqueeze(1))
		predict = torch.cat(predict,1)
		return outputs, predict

