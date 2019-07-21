"""
The main script which generates test data, runs DTW as explained in the slides and saves a json file for precision-recall
"""

from python_speech_features import mfcc, logfbank, fbank
import scipy.io.wavfile as wav
import numpy as np
import os
import json
from dtw_own import dtw_own
import matplotlib.pyplot as plt
from sliding_gen import save_batch
from dl_model import dl_model

np.random.seed(7) #to replicate results
total_dtws = 0 #keep track of how many calls to the DTW functions take place
base_path = '../speech/' #where the speech dataset is located

power_border = 0 #same as that explained in sliding_gen.py
power_fac = 70 #keep frames n if power[n]*power_factoe = max power 
is_sil_frames = 50 #if number of silence frames is greater than this, skip these frames and jump to the next audio section
# word_list = [x for x in os.listdir(base_path) if x != '_background_noise_' and os.path.isdir(base_path+x)]
word_list = ['bed','cat','dog','one','marvin','two','go','eight', 'wow', 'five', 'happy', 'sheila', 'zero', 'house', 'nine', 'three']
using_mfcc = False #set to True if MFCC features are to be used

#returns frames which are padded wherever required
def silence(frames, model):

	audio = np.zeros((1))
	
	if using_mfcc:
		mf_one = mfcc(audio)
		#repeat it 'frames' times since we want these many no of padding frames
		return np.repeat(mf_one, frames, axis=0)
	else:
		fbank_one = fbank(audio)[0]
		#add left and right context so that output of the model is of length 'frames'
		frames += (model.lcont+model.rcont)
		repeated = np.repeat(fbank_one, frames, axis=0)
		return model.test_one(np.log(repeated))

#distance function for DTW
def dist_func(x, y, func='euclidean'):

	if func == 'euclidean':
		return np.sqrt(np.sum((x - y) ** 2))
	elif func == 'cosine':
		dot = np.dot(x, y)
		return 1-dot/(np.linalg.norm(x)*np.linalg.norm(y))
	else:
		print("Distance func not implemented")
		exit(0)

#plot histogram given the LMD table, standard deviation and mode
def plot_hist(LMD, mode, std):

	plt.hist(LMD)
	plt.axvline(x=mode, color='r', label='Mode')
	plt.axvline(x=mode-std, color='g', label='Mode - Std. Dev')
	plt.xlabel('Distance')
	plt.ylabel("Number of occurences")
	plt.legend()
	plt.show()

#convert a list of frames into sequences
#e.g. [22,23,24,25,26,27,31,32,76,79,80,81] as input returns [(22,27),(31,32),(76),(79,81)]
def sequence_from_thresh(match):

	if len(match) == 0:
		print("Couldn't find any audio in input clip")
		exit(0)

	sequences = []
	cur_seq = [match[0]]
	cur_id = 1

	while cur_id<len(match):
		if match[cur_id] == match[cur_id-1]+1:
			cur_seq.append(match[cur_id])
			if cur_id == len(match)-1:
				sequences.append(cur_seq)
				break
		else:
			sequences.append(cur_seq)
			cur_seq = [match[cur_id]]

		cur_id += 1
	if len(sequences) == 0:
		return [(match[0], match[0])]

	sequences = [(x[0], x[-1]) for x in sequences]

	return sequences

#process the keyword/audio clip by removing silence, generating feature vector and returns these features
#is_template = True if the wav file to be processed is a template/keyword
def proc_one(filename, is_template, model):

	(rate,sig) = wav.read(filename)
	#since templates have max value of 32768, normalise it
	#audio clip is already normalised in sliding_gen
	if is_template:
		sig = sig/32768

	assert sig.max() <= 1
	
	#calculate mel filterbank energies
	bank_features, power = fbank(sig, winfunc=np.hamming)

	if using_mfcc:
		features = mfcc(sig)
	else:
		lcont, rcont = model.lcont, model.rcont
		log_fbank = np.log(bank_features)
		#extract features
		features = model.test_one(log_fbank)
		#pad silence to both ends so that final number of vectors = initial number of frames
		if lcont != 0 and r_cont != 0:
			sil_l = silence(lcont, model)
			sil_r = silence(rcont, model)
			features = np.concatenate((sil_l, features, sil_r), axis=0)

	m = power.max()
	sound_where = np.where(power_fac*power>m)[0]

	if is_template:

		sound = features[max(0,sound_where[0]-power_border):min(sound_where[-1]+power_border, power.shape[0]), :]
		#if # frames is still high, make the threshold harsher
		if sound.shape[0] > 50:
			print("Increasing power factor threshold")
			new_multipler = 30
			sound_where = np.where(new_multipler*power>m)[0]
			sound = features[max(0,sound_where[0]-power_border):min(sound_where[-1]+power_border, power.shape[0]), :]

		#uncomment below lines to plot the speech region which is considered afetr silence detection
		# plt.plot(sig)		
		# plt.axvline(x=sound_where[0]*160, color='r')
		# plt.axvline(x=sound_where[-1]*160, color='r')
		# plt.show()
		return sound
	else:
		#in case of templates, first detect where sound is and then slice it into multiple clips
		#if length of silence in between is greater than the value specified above
		silence_where = np.where(power_fac*power<m)[0]
		sequences = sequence_from_thresh(silence_where)
		#find silences greater than the threshold number
		actual_sil = [x for x in sequences if (x[1]-x[0]+1) >= is_sil_frames]
		
		if len(actual_sil) == 0:
			print("Couldn't find any silence")
			return [(features, 0)]

		sound = []
		cur_start = 0

		for sequence in actual_sil:
			finish, next_beginning = sequence[0], sequence[1]
			if finish == 0:
				cur_start = next_beginning
				continue
			cur_seq = features[max(0,cur_start-power_border):min(finish+1+power_border, features.shape[0]), :]
			
			if not using_mfcc and (lcont is not None and rcont is not None):
				if cur_seq.shape[0] < (lcont+rcont+1):
					print("Ignoring segment since it is too short")
					continue

			sound.append((cur_seq, max(0,cur_start-power_border)))
			cur_start = next_beginning

		#append final features
		sound.append((features[max(0,cur_start-power_border):, :], max(0,cur_start-power_border)))
		print("Total silences:", len(actual_sil))

		#uncomment below lines to plot the speech region which is considered afetr silence detection
		# plt.plot(sig)
		# x = [x[1]+x[0].shape[0] for x in sound]
		# x += [x[1] for x in sound]
		# print([160*x for x in x])
		# for xc in x:
		# 	plt.axvline(x=xc*160, color='r')
		# plt.show()
		# print([x[1] for x in sound])

		return sound

#generates the LMD table by DTW calculations
def compare(clips, template, model):

	global total_dtws

	temp_l = template.shape[0]
	print("Length of template:",temp_l)
	lower, upper = int(temp_l*0.5), int(2*temp_l) #controlled by the variation in the human speech rate
	LMD = {} #key is the starting frame while value is the minimum distance

	#clips is a list with each element = (feature vectors, starting frame number in the actual clip)
	for clip, clip_start in clips:
		
		clip_l = clip.shape[0]
		#number of starting frames to check
		total_tries = clip_l-lower
		
		#if length of clip < lower_limit*(length of template), pad with silence
		if total_tries < 0:
			total_tries = 1
			clip = np.concatenate((clip, silence(lower-clip_l, model)))
			clip_l = clip.shape[0]

		distances_matrix = np.zeros((clip_l, temp_l))
		#calculated distance matrix and feed it to DTW to avoid repeated callculations
		for i in range(clip_l):
			for j in range(temp_l):
				distances_matrix[i,j] = dist_func(clip[i], template[j])

		print("Total starting frames to check:", total_tries)
		
		for start_frame in range(0, total_tries):

			distances = []

			for length in range(lower, upper+1):

				clip_to_check = clip[start_frame:start_frame+length, :] #consider only a slice of the total clip
				dtw_cost = dtw_own(clip_to_check, template, distances=distances_matrix[start_frame:start_frame+length, :])[1]
				total_dtws += 1
				distances.append(dtw_cost)
			#append minimum distance to table
			LMD[clip_start+start_frame] = min(distances)
			#print progress evry 10 frames
			if start_frame%10 == 0:
				print("Starting frame:",clip_start+start_frame)
				print("Min distance:",LMD[clip_start+start_frame])

	return LMD, temp_l

#takes LMD distances table and parameters as input, calculates histogram and outputs True or False (keyword present or not)
def hist_and_finalresult(LMD_dict, template_length, std_multi, cons_K):

	#calculate values required for histogram plotting
	LMD = list(LMD_dict.values())
	hist_data, edges = np.histogram(LMD)
	std = np.std(LMD)
	max_id = np.argmax(hist_data)
	mode = (edges[max_id]+edges[max_id+1])/2

	#uncomment this to see the histogram
	# plot_hist(LMD, mode, std)

	LMD = np.ones((max(LMD_dict.keys())+1))*np.inf
	for idx, val in LMD_dict.items():
		LMD[idx] = val
	#find starting frames where min distance is less than threshold
	match = np.where(LMD <= mode - std_multi*std)[0]
	# print(mode, std)
	print(match)
	if len(match) == 0:
		print("No matches found below threshold")
		return False
	#get sequence from starting frame numbers
	sequences = sequence_from_thresh(match)
	sequences = [x[1]-x[0]+1 for x in sequences]
	print(sequences)
	#normalise by templaet length
	print([x/template_length for x in sequences])
	max_seq = max(sequences)
	
	if max_seq >= cons_K*template_length:
		return True
	else:
		return False

#carries out tests given number of templates and the std_multiplier range and k range
def testing(templates_num, std_multi_list, cons_k_list, model, sliding_path='sliding/'):

	#dictionary which stores tp (true +ve), fp, tn, fn
	#initialise the dictionary
	final_results = {}
	for std in std_multi_list:
		final_results[std] = {}
		for cons_k in cons_k_list:
			final_results[std][cons_k] = {'tp':0, 'fp':0, 'tn':0, 'fn':0}

	if model is None or using_mfcc == True:
		print("Using MFCC")
	else:
		print("Using NN features")

	global total_dtws

	#labels contain info on which words are present in the audio clip
	with open(sliding_path+'data.json', 'r') as f:
		labels = json.load(f)

	all_words = []
	for word in word_list:
		files = [base_path+word+'/'+x for x in os.listdir(base_path+word)]
		all_words += files
	
	np.random.shuffle(all_words)
	templates = all_words[:templates_num] #choose templates to compare audio clips with
	audio_clips = [sliding_path+x for x in os.listdir(sliding_path) if x.split('.')[-1] == 'wav']
	
	idx = 0

	for clip in audio_clips:
		
		words_in_clip = list(labels[clip].values())
		print("Words in clip:",clip,'are:',words_in_clip)
		clip_feat = proc_one(clip, False, model)

		for template in templates:

			idx += 1
			print("On", idx)

			word = template.split('/')[2]

			print("Comparing:",clip,"and",template)

			temp_feat = proc_one(template, True, model)
			
			LMD, temp_l = compare(clip_feat, temp_feat, model)
			#once we have the LMD table, generate prediction by iterating over parameter values

			for std_multi in std_multi_list:

				for cons_k in cons_k_list:

					print("\n(Std. Dev, Cons_k) =", std_multi, cons_k)

					prediction = hist_and_finalresult(LMD, temp_l, std_multi, cons_k)

					if prediction == True:
						if word in words_in_clip:
							print("TP")
							final_results[std_multi][cons_k]['tp'] += 1
						else:
							print("FP")
							final_results[std_multi][cons_k]['fp'] += 1
					else:
						if word in words_in_clip:
							print("FN")
							final_results[std_multi][cons_k]['fn'] += 1
						else:
							print("TN")
							final_results[std_multi][cons_k]['tn'] += 1

					print("Total DTW till now:", total_dtws)

	print(final_results)
	return final_results

#runs experiment given the list of parameter values
def grid_search():

	#initialise feature extractor model
	a = dl_model('test_one')
	l_cont, r_cont = a.lcont, a.rcont

	#directory where json file of results is stored
	if not os.path.exists('pr'):
		os.mkdir('pr')

	#range of k_values and std_multiplier values
	k_vals = [0.2, 0.4, 0.6, 0.8]
	std_vals = [0.8, 1.2, 1.6, 2]

	#total # of templates and clips to generate
	templates_num = 5
	target_num = 15
	
	#generate audio files with concantenated words
	save_batch(target_num, word_list)

	if not using_mfcc:
		json_name = 'pr/'+a.config_file['dir']['models'].split('/')[1]+'.json'
	else:
		json_name = 'pr/mfcc.json'

	print(json_name)
	#generate results
	cur_res = testing(templates_num, std_vals, k_vals, model=a)
	#dump json
	with open(json_name, 'w') as f:
		json.dump(cur_res, f)


if __name__ == '__main__':
	grid_search()


