"""
The main driver file responsible for training, testing and extracting features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np
import json
from dataloader import timit_loader
from read_yaml import read_yaml


class dl_model():

	def __init__(self, mode):

		#read config fiel which contains parameters
		self.config_file = read_yaml()
		self.mode = mode

		self.cuda = (self.config_file['cuda'] and torch.cuda.is_available())
		self.output_dim = self.config_file['output_dim']

		self.lcont, self.rcont = self.config_file['left_context'], self.config_file['right_context']

		#import the necessary architecture
		if self.config_file['depth'] == 'deep':
			from model import DeepVanillaNN as Model
		elif self.config_file['depth'] == 'shallow':
			from model import ShallowVanillaNN as Model
		else:
			print("Can't identify model")
			exit(0)

		if mode == 'train' or mode == 'test':

			self.plots_dir = self.config_file['dir']['plots']
			#store hyperparameters
			self.total_epochs = self.config_file['train']['epochs']
			self.test_every = self.config_file['train']['test_every_epoch']
			self.test_per = self.config_file['train']['test_per_epoch']
			self.print_per = self.config_file['train']['print_per_epoch']
			self.save_every = self.config_file['train']['save_every']
			self.plot_every = self.config_file['train']['plot_every']
			#dataloader which returns batches of data
			self.train_loader = timit_loader('train', self.config_file)
			self.test_loader = timit_loader('test', self.config_file)

			#ensure that no of phone classes = output dimension
			assert self.train_loader.num_phoenes == self.output_dim

			self.start_epoch = 1
			self.test_acc = []
			self.train_losses, self.test_losses = [], []
			#declare model
			self.model = Model(self.config_file, weights=self.train_loader.distri_weights)

		else:

			self.model = Model(self.config_file, weights=np.ones((self.output_dim)))

		#load id to phone mapping
		fname = self.config_file['dir']['dataset']+'test_mapping.json'
		try:
			with open(fname, 'r') as f:
				self.id_to_phones = json.load(f)
		except:
			print("Cant find mapping")
			exit(0)

		if self.cuda:
			self.model.cuda()

		#resume training from some stored model
		if (self.mode == 'train' and self.config_file['train']['resume']):
			self.start_epoch, self.train_losses, self.test_losses, self.test_acc = self.model.load_model(mode)
			self.start_epoch += 1

		#load bets model for testing/feature extraction
		elif self.mode == 'test' or mode == 'test_one':
			self.model.load_model(mode)

	def train(self):

		print("Training...")
		print('Total batches:', len(self.train_loader))
		self.model.train()

		#when to print losses during the epoch
		print_range = list(np.linspace(0, len(self.train_loader), self.print_per+2, dtype=np.uint32)[1:-1])
		
		if self.test_per == 0:
			test_range = []
		else:
			test_range = list(np.linspace(0, len(self.train_loader), self.test_per+2, dtype=np.uint32)[1:-1])

		for epoch in range(self.start_epoch, self.total_epochs+1):

			print("Epoch:",str(epoch))
			epoch_loss = 0.0

			for i, (inputs, labels) in enumerate(self.train_loader):
				
				inputs, labels = torch.from_numpy(inputs).float(), torch.from_numpy(labels).long()
				
				if self.cuda:
					inputs = inputs.cuda()
					labels = labels.cuda()

				# zero the parameter gradients
				self.model.optimizer.zero_grad()
				# forward + backward + optimize
				outputs = self.model(inputs)
				loss = self.model.loss(outputs, labels)
				loss.backward()
				self.model.optimizer.step()

				#store loss
				epoch_loss += loss.item()

				if i in print_range:
					print('After %i batches, Current Loss = %.7f, Avg. Loss = %.7f' % (i+1, epoch_loss/(i+1), np.mean([x[0] for x in self.train_losses])))

				if i in test_range:
					self.test(epoch)

				if i == len(self.train_loader)-1:
					break

			self.train_losses.append((epoch_loss/len(self.train_loader), epoch))

			#test every 5 epochs in the beginning and then every fixed no of epochs specified in config file
			#useful to see how loss stabilises in the beginning
			if epoch % 5 == 0 and epoch < self.test_every:
				self.test(epoch)
			elif epoch % self.test_every == 0:
				self.test(epoch)
			#plot loss and accuracy
			if epoch % self.plot_every == 0:
				self.plot_loss_acc(epoch)

			#save model
			if epoch % self.save_every == 0:
				self.model.save_model(False, epoch, self.train_losses, self.test_losses, self.test_acc)


	def test(self, epoch=None):

		self.model.eval()
		correct = 0
		total = 0
		#confusion matrix data is stored in this matrix
		matrix = np.zeros((self.output_dim, self.output_dim))

		print("Testing...")
		print('Total batches:', len(self.test_loader))
		test_loss = 0

		with torch.no_grad():

			for i, (inputs, labels) in enumerate(self.test_loader):

				inputs, labels = torch.from_numpy(inputs).float(), torch.from_numpy(labels).long()
				
				if self.cuda:
					inputs = inputs.cuda()
					labels = labels.cuda()

				outputs = self.model(inputs)
				loss = self.model.loss(outputs, labels)
				test_loss += loss.item()

				outputs = outputs.cpu().numpy()
				labels = labels.cpu().numpy()

				argmaxed = np.argmax(outputs, 1)
				
				#matrix[i][j] denotes the no of examples classified by model as class j but have ground truth label i
				for k in range(argmaxed.shape[0]):
					matrix[labels[k]][argmaxed[k]] += 1

				#tota number of correct phone predictions
				correct += np.sum(argmaxed == labels)
				total += len(argmaxed)
				
				if i == len(self.test_loader)-1:
					break

		for i in range(self.output_dim):
			matrix[i] /= sum(matrix[i])

		acc = correct/total
		test_loss /= len(self.test_loader)

		#plot confusion matrix
		if epoch is not None:

			filename = self.plots_dir+'confmat_epoch_acc_'+str(epoch)+'_'+str(int(100*acc))+'.png'
			plt.clf()
			plt.imshow(matrix, cmap='hot', interpolation='none')
			plt.gca().invert_yaxis()
			plt.xlabel("Predicted Label ID")
			plt.ylabel("True Label ID")
			plt.colorbar()
			plt.savefig(filename)

		print("Testing accuracy: %.4f , Loss: %.7f" % (acc, test_loss))

		self.test_acc.append((acc, epoch))
		self.test_losses.append((test_loss, epoch))
		
		#if testing loss is minimum, store it as the 'best.pth' model, which is used for feature extraction
		if test_loss == min([x[0] for x in self.test_losses]):
			print("Best new model found!")
			self.model.save_model(True, epoch, self.train_losses, self.test_losses, self.test_acc)

		return acc

	#called during feature extraction. Takes log mel filterbank energies as input and outputs the phone predictions
	def test_one(self, feat_log):
		
		self.model.eval()

		with torch.no_grad():

			(total_frames, feature_dim) = feat_log.shape

			#construct model input vector by taking context into consdieration
			inputs = np.zeros((total_frames-self.lcont-self.rcont, (self.lcont+self.rcont+1)*feature_dim))
			idx = 0
			#generate context vector for each centre frame
			for centre_frame in range(self.lcont, total_frames-self.rcont):
				inputs[idx] = feat_log[centre_frame-self.lcont:centre_frame+self.rcont+1, :].flatten()
				idx += 1

			inputs = torch.from_numpy(inputs).float()
			if self.cuda:
				inputs = inputs.cuda()

			outputs = self.model(inputs).cpu().numpy()
			#take softmax
			for i in range(outputs.shape[0]):
				outputs[i,:] = np.exp(outputs[i,:])/np.sum(np.exp(outputs[i,:]))
			#print predicted phone on taking argmax
			print([self.id_to_phones[str(x)] for x in np.argmax(outputs, axis=1)])
			return outputs

	#take train/test loss and test accuracy input and plot it over time
	def plot_loss_acc(self, epoch):

		plt.clf()
		plt.plot([x[1] for x in self.train_losses], [x[0] for x in self.train_losses], c='r', label='Train')
		plt.plot([x[1] for x in self.test_losses], [x[0] for x in self.test_losses], c='b', label='Test')
		plt.title("Train/Test loss")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.legend()
		plt.grid(True)

		filename = self.plots_dir+'loss'+'_'+str(epoch)+'.png'
		plt.savefig(filename)
		
		plt.clf()
		plt.plot([x[1] for x in self.test_acc], [100*x[0] for x in self.test_acc], c='r')
		plt.title("Test accuracy")
		plt.xlabel("Epochs")
		plt.ylabel("Accuracy in %%")
		plt.grid(True)

		filename = self.plots_dir+'test_acc'+'_'+str(epoch)+'.png'
		plt.savefig(filename)

		print("Saved plots")

if __name__ == '__main__':
	a = dl_model('train')
	a.train()
	# a = dl_model('test_one')
	# b = -10*np.ones((9,26))
	# a.test_one(b)
