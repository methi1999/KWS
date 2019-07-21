"""
Returns batches of training/testing data for training NN feautre extractor
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import os
import torch
import json
from metadata import timit_data

#Set the seed to replicate results
random.seed(a=7)

#Returns data during training/testing from the dumped pickle file by metadata.py
class timit_loader():

	def __init__(self, type_, config_file):

		self.config = config_file
		path = config_file['dir']['dataset']

		metadata = timit_data(type_.upper(), config_file)
		#Build dictionary with key = phone and values = list of feature vectors
		database = metadata.gen_pickle()

		self.mode = type_ #train/test/test-one
		self.batch_size = config_file[type_]['batch_size']

		#fold phones in list to the phone which is the key e.g. 'ao' is 'collapsed' into 'aa'
		self.replacement = {'aa':['ao'], 'ah':['ax','ax-h'],'er':['axr'],'hh':['hv'],'ih':['ix'],
		'l':['el'],'m':['em'],'n':['en','nx'],'ng':['eng'],'sh':['zh'],'pau':['pcl','tcl','kcl','bcl','dcl','gcl','h#','epi'],
		'uw':['ux']}

		#Fold phones and make list of training examples
		self.make_pairs(database)
		id_to_phones = {}

		#Dump mapping from id to phone. Used to convert NN output back to the phone it predicted
		for key, val in self.dict_stats.items():
			id_to_phones[val[1]] = key

		#Dump this mapping
		fname = config_file['dir']['dataset']+self.mode+'_mapping.json'
		with open(fname, 'w') as f:
			json.dump(id_to_phones, f)

	def make_pairs(self, database):

		#Deleting stops
		database.pop('q')

		#Collapse phones
		for father, children_list in self.replacement.items():
			for child in children_list:
				cur_list = database.pop(child)
				database[father] += cur_list

		#Map each phone to an id
		phoenes = sorted(database.keys())
		self.num_phoenes = len(phoenes)
		self.dict_stats = dict(zip(phoenes, [0]*self.num_phoenes))

		ids = list(range(self.num_phoenes))
		phoene_to_id = dict(zip(phoenes, ids))
		
		#List of training/testing examples to be passed to the model
		self.couples = []
		for ph, ph_list in database.items():
			for eg in ph_list:
				self.couples.append((phoene_to_id[ph], eg))
			#dict states has key=phone and value = (num_examples, phone_id)
			self.dict_stats[ph] = (len(ph_list), phoene_to_id[ph])
		
		random.shuffle(self.couples)
		print("Total number of couples:",len(self.couples))
		
		#Due to skewed distribution, calculate weights inversely proportional to the number of examples
		nums = [self.dict_stats[key][0] for key in sorted(self.dict_stats.keys())]
		nums = [1/x for x in nums]
		total = sum(nums)
		self.distri_weights = [x/total for x in nums]
		idx = 0

		for key in sorted(self.dict_stats.keys()):
			#add weights to the dict_stats
			self.dict_stats[key] = self.dict_stats[key][0], self.dict_stats[key][1], self.distri_weights[idx]
			idx += 1

		print("Vocab stats:", self.dict_stats)

	def __getitem__(self, idx):

		#Return a batch of training/testing examples
		ids = range(idx*self.batch_size, idx*self.batch_size + self.batch_size)
		vecs, labels = [], []
		for idx in ids:
			labels.append(self.couples[idx][0])
			vecs.append(self.couples[idx][1])

		return np.array(vecs), np.array(labels)

	def __len__(self):

		return len(self.couples)//self.batch_size