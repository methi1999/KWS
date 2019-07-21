"""
Concatenate audio files to generate 2-3 sets of 2-3 words for testing model performance
"""
from python_speech_features import mfcc, logfbank, fbank
import scipy.io.wavfile as wav
import numpy as np
import os
import json
import shutil

#base path directories and words to use
base_path = '../speech/'
# word_list = [x for x in os.listdir(base_path) if x != '_background_noise_' and os.path.isdir(base_path+x)]
word_list = ['bed','cat','dog','one','marvin','two','go','eight']

np.random.seed(8) #ensures replicable results
#if word set is 'x' seconds long, then random silence (in seconds) inserted between this and the next set of words is 
#any number between 0.25*x and 0.35*x seconds
sil_len = (0.25, 0.35)
total_sets = [2,3] #number of sets of words. ranodmly chosen form this list
words_per_set = [2,3] #number of words in each set. sets are separated by silence as mentioned above
#on audio detection, include these many number of frames at the boundary so that joined words are distinguishable
#e.g. if sound is below threshold from frame 0 to 30 and 60 to 100, include audio from 30-power-border to 60+power border
power_border = 6 
power_fac = 100 #threshold below maximum peak in power spectra
with_replacement = False #set True if repeated words in the same set is accepted

#process one wav file and return section where there is speech
def proc_one(filename):

	(rate,sig) = wav.read(filename)
	#normalise so that max value is 1. This ensures that concatenated words have the same intensity loudness
	#reasonable assumption since a recording of a sentence will have the same average intensity across words
	sig = sig/(max(sig))
	bank_features, power = fbank(sig, winfunc=np.hamming) #calculate mel filter bank
	#not precise because of the first window but tolerable error
	frame_multiplier = rate//power.shape[0]
	
	m = power.max()
	sound_where = np.where(power_fac*power>m)[0] #frames where power is more than a factor*peak power
	
	sound = sig[frame_multiplier*max(0,sound_where[0]-power_border):frame_multiplier*min(sound_where[-1]+power_border, power.shape[0])]
	return sound

#create one audio file with concatenated words
#word_l is a list of lists, each list containing the word to be appended
def create_one(word_l):

	final_audio = []
	labels = {} #store at which sample the inserted word begins
	frames_till_now = 0
	for i in range(len(word_l)):
		word_set = word_l[i]
		frames = 0
		#iterate over each word in the set
		for word in word_set:
			files = os.listdir(base_path+word)
			chosen = base_path+word+'/'+files[np.random.randint(len(files))] #choose random wav file
			proc = proc_one(chosen) #process it
			labels[frames_till_now] = word #store where it starts
			frames += proc.shape[0]
			frames_till_now += proc.shape[0]
			final_audio.append(proc)

		#insert silence between set of words
		if i != len(word_l)-1:
			#choose random number of frames to insert
			silence_frames = int(frames*((sil_len[1]-sil_len[0])*np.random.random_sample() + sil_len[0]))
			final_audio.append([0]*silence_frames) #insert silence
			labels[frames_till_now] = 'silence'
			frames_till_now += silence_frames

	final_np = np.concatenate(final_audio)
	return final_np, labels

#create a batch of files
def save_batch(total_files = 10, word_list=None):

	#where to store the final audio files
	dump_path = 'sliding/'
	dump_labels = dump_path+'data.json'
	if not os.path.exists(dump_path):
		os.mkdir(dump_path)
	else:
		shutil.rmtree(dump_path)
		os.mkdir(dump_path)

	print("Generating",total_files,'words\n')
	
	idx = 0
	metadata = {}

	for i in range(total_files):

		words_chosen = []
		num_sets = total_sets[np.random.randint(len(total_sets))] #2 or 3 sets

		for i in range(num_sets):
			num_in_set = words_per_set[np.random.randint(len(words_per_set))] #2 or 3 words in each set
			np.random.shuffle(word_list)
			words = word_list[:num_in_set] #which words to insert in current set
			words_chosen.append(words)
		
		print(words_chosen)
		wav_data, labels = create_one(words_chosen) #make audio file and labels
		wav_name = dump_path+str(idx)+'.wav'
		wav.write(wav_name, 16000, wav_data) #dump concatenated file
		
		metadata[wav_name] = labels
		idx += 1

	#dump data for starting frame
	with open(dump_labels, "w") as write_file:
	    json.dump(metadata, write_file, indent=4)

	print("\nGenerated required words\n")

if __name__ == '__main__':
	save_batch()
