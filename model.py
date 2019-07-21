"""
Define NN architecture, forward function and loss function
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from generic_model import generic_model

class DeepVanillaNN(generic_model):

	def __init__(self, config, weights):

		super(DeepVanillaNN, self).__init__(config)

		self.dropout = nn.Dropout(p=0.2)

		#input dimension = feature vector dimension
		input_dim = (config['left_context']+config['right_context']+1)*config['input_dim']

		#define architecture
		hn1, hn2, hn3, hn4, hn5 = 512, 1024, 2048, 1024, 512
		self.fc1 = nn.Linear(input_dim, hn1)
		self.bn1 = nn.BatchNorm1d(hn1)
		self.fc2 = nn.Linear(hn1, hn2)
		self.bn2 = nn.BatchNorm1d(hn2)
		self.fc3 = nn.Linear(hn2, hn3)
		self.bn3 = nn.BatchNorm1d(hn3)
		self.fc4 = nn.Linear(hn3, hn4)
		self.bn4 = nn.BatchNorm1d(hn4)
		self.fc5 = nn.Linear(hn4, hn5)
		self.bn5 = nn.BatchNorm1d(hn5)
		self.fc6 = nn.Linear(hn5, config['output_dim'])

		#declare loss function by reading the config file
		loss, optimizer = config['train']['loss_func'], config['train']['optim']
		loss_found, optim_found = False, False

		if loss == 'BCEL':
			self.loss_func = nn.BCELoss()
			loss_found = True
		elif loss == 'CEL':
			if config['train']['weighted']:
				#weights for skewed classes
				self.loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))
				print("Using weighted CEL")
			else:
				self.loss_func = nn.CrossEntropyLoss()
			loss_found = True

		if optimizer == 'SGD':
			self.optimizer = optim.SGD(self.parameters(), lr=config['train']['lr'], momentum=0.9)
			optim_found = True
		elif optimizer == 'Adam':
			self.optimizer = optim.Adam(self.parameters(), lr=config['train']['lr'])
			optim_found = True

		if loss_found == False or optim_found == False:
			print("Can't find desired loss function/optimizer")
			exit(0)

	def forward(self, x):

		x = self.dropout(F.relu(self.bn1(self.fc1(x))))
		x = self.dropout(F.relu(self.bn2(self.fc2(x))))
		x = self.dropout(F.relu(self.bn3(self.fc3(x))))
		x = self.dropout(F.relu(self.bn4(self.fc4(x))))
		x = self.dropout(F.relu(self.bn5(self.fc5(x))))

		x = self.fc6(x)

		return x

#Only difference is the model architecture, rest all remains the same
class ShallowVanillaNN(generic_model):

	def __init__(self, config, weights):

		super(ShallowVanillaNN, self).__init__(config)

		input_dim = (config['left_context']+config['right_context']+1)*config['input_dim']

		self.dropout = nn.Dropout(p=0.2)
		
		hn1, hn2, hn3 = 512, 1024, 512
		self.fc1 = nn.Linear(input_dim, hn1)
		self.bn1 = nn.BatchNorm1d(hn1)
		self.fc2 = nn.Linear(hn1, hn2)
		self.bn2 = nn.BatchNorm1d(hn2)
		self.fc3 = nn.Linear(hn2, hn3)
		self.bn3 = nn.BatchNorm1d(hn3)
		self.fc4 = nn.Linear(hn3, config['output_dim'])

		loss, optimizer = config['train']['loss_func'], config['train']['optim']
		loss_found, optim_found = False, False

		if loss == 'BCEL':
			self.loss_func = nn.BCELoss()
			loss_found = True
		elif loss == 'CEL':
			if config['train']['weighted']:
				self.loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))
				print("Using weighted CEL")
			else:
				self.loss_func = nn.CrossEntropyLoss()
			loss_found = True

		if optimizer == 'SGD':
			self.optimizer = optim.SGD(self.parameters(), lr=config['train']['lr'], momentum=0.9)
			optim_found = True
		elif optimizer == 'Adam':
			self.optimizer = optim.Adam(self.parameters(), lr=config['train']['lr'])
			optim_found = True

		if loss_found == False or optim_found == False:
			print("Can't find desired loss function/optimizer")
			exit(0)

	def forward(self, x):

		x = self.dropout(F.relu(self.bn1(self.fc1(x))))
		x = self.dropout(F.relu(self.bn2(self.fc2(x))))
		x = self.dropout(F.relu(self.bn3(self.fc3(x))))

		x = self.fc4(x)

		return x