"""
Converts raw TIMIT data into a pickle dump which can be used during training

NOTE: If n context frames are used, the size of the database will be n times larger than the base size.
Because each training example of (n+1) frames is generated separately and hence the same frame will be found and stored in (n+1) training examples
This works for smaller no. of context frames as the pickle dump is small enough to load into memory
THe advantage is that we get a slightly reduced train/test time since the dataloader does not need to construct 
the training/testing every time when the model asks for a batch
For larger context, it would be better to shift the metadata code, which constructs the training examples from adjacent frames, to the dataloader
"""

import numpy as np
import pickle
import os
import scipy.io.wavfile as wav
from python_speech_features import logfbank, fbank

#Ignore DS_Store files found on Mac
def listdir(pth):
	return [x for x in os.listdir(pth) if x != '.DS_Store']

#Convert from sample number to frame number
#e.g. sample 34 is in frame 1 assuming 25ms windows, 10 ms hop (assuming 0-indexing)
def sample_to_frame(num, rate=16000, window=25, hop=10):
	multi = rate//(hop*100)
	if num<25*multi:
		return 0
	else:
		return (num-multi*window)//(multi*hop)+1

class timit_data():

	def __init__(self, type_, config_file):

		self.config = config_file
		self.mode = type_
		self.db_path = config_file['dir']['dataset']

		#number of left, right context frames
		self.lcont, self.rcont = config_file['left_context'], config_file['right_context']
		self.pkl_name = self.db_path+self.mode+'_'+str(self.lcont)+'.pkl'

	#Generate and store pickle dump
	def gen_pickle(self):

		if os.path.exists(self.pkl_name):
			print("Found pickle dump for", self.mode)
			with open(self.pkl_name, 'rb') as f:
				return pickle.load(f)

		print("Generating pickle dump for", self.mode)

		to_return = {} #dictionary with key=phone and value=list of feature vectors with key phone in the centre frame
		base_pth = self.db_path+self.mode
		log_dim = self.config['input_dim']

		for dialect in sorted(listdir(base_pth)):

			print("Dialect:", dialect)

			for speaker_id in sorted(listdir(os.path.join(base_pth, dialect))):

				data = sorted(os.listdir(os.path.join(base_pth, dialect, speaker_id)))
				wav_files = [x for x in data if x.split('.')[-1] == 'wav'] #all the .wav files

				for wav_file in wav_files:

					wav_path = os.path.join(base_pth, dialect, speaker_id, wav_file)
					(rate,sig) = wav.read(wav_path)
					sig = sig/32768 #normalise
					feat, energy = fbank(sig, samplerate=rate, winfunc=np.hamming)
					feat_log_full = np.log(feat) #calculate log mel filterbank energies for complete file
					phenome_path = wav_path[:-3]+'PHN' #file which contains the phenome location data
					#phones in current wav file
					cur_phones = []

					with open(phenome_path, 'r') as f:
						a = f.readlines()

					for phenome in a:
						s_e_i = phenome[:-1].split(' ') #start, end, phenome_name e.g. 0 5432 'aa'
						start, end, ph = int(s_e_i[0]), int(s_e_i[1]), s_e_i[2]
						#take only the required slice from the complete filterbank features
						feat_log = feat_log_full[sample_to_frame(start):sample_to_frame(end)]

						for i in range(feat_log.shape[0]):
							cur_phones.append((ph, feat_log[i]))

					#Use context to build database of (phone):[context+central frame vectors]
					for centre_frame in range(self.lcont, len(cur_phones)-self.rcont):

						final_np = np.zeros((self.rcont+self.lcont+1, log_dim))
						ph_mid = cur_phones[centre_frame][0] #the phone in the middle frame, which is the target label during training
						idx = 0
						#Append the left and right context frames
						for x in range(centre_frame-self.lcont, centre_frame+self.rcont+1):
							final_np[idx] = np.array(cur_phones[x][1])
							idx += 1
						
						
						if ph_mid not in to_return.keys():
							to_return[ph_mid] = []
						#Store it in the final database
						to_return[ph_mid].append(final_np.flatten())

		#Dump pickle
		with open(self.pkl_name, 'wb') as f:
			pickle.dump(to_return, f)
			print("Dumped pickle")

		return to_return

if __name__ == '__main__':

	config_file = {'dir':{'dataset':'TIMIT/'}, 'left_context':0, 'right_context':0, 'input_dim':26}
	a = timit_data('TRAIN', config_file)
	a.gen_pickle()