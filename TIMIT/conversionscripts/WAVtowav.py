from scipy.io import wavfile
import subprocess
import glob
import os
# Lists all the wav files
wav_files_list = glob.glob('../TEST/*/*/*.WAV')
wav_prime = []
# print(wav_files_list)
# Create temporary names for the wav files to be converted. They will be renamed later on.
for f in wav_files_list:
	fileName, fileExtension = os.path.splitext(f)
	fileName = fileName.split('/')[-1]
	wav_prime.append(fileName+'.wav')
# print(wav_prime)
# Command strings
cmd = "./sox {0} -t wav {1}"
mv_cmd = "mv {0} {1}"
 
# Convert the wav_files first. Remove it. Rename the new file created by sox to its original name
for i, f in enumerate(wav_files_list):
	subprocess.call(cmd.format(f, wav_prime[i]), shell=True)
	os.remove(f)
	f = f[:-3]+'wav'
	subprocess.call(mv_cmd.format(wav_prime[i],f), shell=True)
print("DONE")