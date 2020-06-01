import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ExtractiveTagger(nn.Module):
	def __init__(self, embedding_dimension, hidden_dimension, output_dimension, dictionary_size, weight_matrix, layers, drop, bidirection):
		super(ExtractiveTagger, self).__init__()
		self.word_embedding = nn.Embedding(dictionary_size, embedding_dimension)
		self.word_embedding.weight.data.copy_(torch.from_numpy(weight_matrix)) 
		self.hidden_dimension = hidden_dimension
		self.lstm = nn.LSTM(embedding_dimension, hidden_dimension, num_layers=layers, dropout=drop, batch_first=True, bidirectional=bidirection)
		self.l = nn.Linear(hidden_dimension*2, 8)
		self.classifer = nn.Linear(8, output_dimension)

	def forward(self, sentence):
		embedded_output = self.word_embedding(sentence)
		output, hidden = self.lstm(embedded_output)
		output = torch.tanh(self.l(output))
		tag_space = self.classifer(output)
		return tag_space[:,:,0]
