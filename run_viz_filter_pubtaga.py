from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import time
from keras import backend as K
from keras.layers.core import Dense,Dropout,Activation,Flatten,Reshape,Permute
from keras.layers.convolutional import ZeroPadding1D,Convolution1D,MaxPooling1D
from keras.layers import Input,merge
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.models import model_from_json,Model

from scipy.signal import butter, lfilter, freqz

import librosa
import matplotlib.pyplot as plt
import os
os.environ['DISPLAY'] = 'localhost:10.0'

# save path
save_path = '/home/richter/viz_filter_ETE/save_fig_729/'
save_path_wav = '/home/richter/viz_filter_ETE/save_fig_729_wav/'

# dimensions of the generated pictures for each filter.
sample_length = 729 #512

def butter_lowpass(cutoff,fs,order=5):
	nyq = 0.5*fs
	normal_cutoff = cutoff / nyq
	b,a = butter(order, normal_cutoff, btype='low', analog=False)
	return b,a

def butter_lowpass_filter(data,cutoff,fs,order=5):
	b,a = butter_lowpass(cutoff,fs,order=order)
	y = lfilter(b,a,data)
	return y

# filter options
order = 6
fs_ft = 22.05
cutoff = 5.

# step size for gradient ascent
step = 1.
step_num = 18
conv_dense = 'conv' # conv or dense
norm_param_list = [1e-9]
layer_list = ['activation_1','activation_2','activation_3','activation_4','activation_5','activation_6']
nb_filters_list = [128,128,128,256,256,256]

#layer_list = ['dense_1']
#nb_filters_list = [50]

fftsize = 729 #513

hop_size = 3 #2
window = 3 #2

init = 'he_uniform'
activ = 'relu'

architecture_path = 'model_architecture_19683frames_power3_ETE_256_3_3_0.000016.json' #'../transferrable_audioset_mtat/mtat_model_architecture_8192frames_ETE_2_2_0.000016.json'
weight_path = 'best_weights_19683frames_power3_ETE_256_3_3_0.000016.hdf5' #'../transferrable_audioset_mtat/mtat_best_weights_8192frames_ETE_2_2_0.000016.hdf5'

json_file = open(architecture_path,'r')
loaded_json = json_file.read()
json_file.close()
model = model_from_json(loaded_json)

model.load_weights(weight_path)
print('model loaded!!!')

model.summary()

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

# this is the placeholder for the input images (None,59049,1)
input_img = model.input

def normalize(x,norm_param):
	# utility function to normalize a tensor by its l2 norm
	return x / (K.sqrt(K.mean(K.square(x))) + norm_param) # -5?

plt.figure()
	
# norm_param for loop
for norm_param in norm_param_list:
	# all layers for loop
	for iter,layer_name in enumerate(layer_list):
		print(iter,layer_name)
		
		# save name
		save_name = '%s_norm%s_filters.png' % (layer_name,str(norm_param))
		print(save_name)

		if os.path.isfile(save_path+save_name) == 1:
			print('already calculated:',save_name)
			continue

		nb_filters = nb_filters_list[iter]
		repetition = int((fftsize/2+1)/nb_filters)
		print('repetition:' + str(repetition))

		fftzed = np.zeros((nb_filters,fftsize/2+1))	
		for filter_index in range(0,nb_filters):

			# we only scane through the first 10 filters.
			# but there are actually ## of them
			print('Processing filter %d' % filter_index)
			start_time = time.time()

			# we build a loss function that maximizes the activation
			# of the nth filter of the layer considered
			layer_output = layer_dict[layer_name].output
			if conv_dense == 'conv':
				loss = K.mean(layer_output[:,:,filter_index])
			elif conv_dense == 'dense':
				loss = K.mean(layer_output[:,filter_index])

			# we compute the gradient of the input picture wrt this loss
			grads = K.gradients(loss, input_img)[0]

			# normalization trick: we normalize the gradient
			grads = normalize(grads,norm_param)

			# this function returns the loss and grads given the input picture
			iterate = K.function([input_img, K.learning_phase()],[loss,grads])

			# we start from a gray image with some random noise
			input_img_data = np.random.random((1,sample_length,1))
			input_img_data = (input_img_data - 0.5) * 0.03 #1.8

			# we run gradient ascent for 20 steps
			for i in range(step_num):
				loss_value, grads_value = iterate([input_img_data,1]) # 0 test phase
				input_img_data += grads_value * step

				print('Current loss value:', loss_value)
				if loss_value <= 0.:
					# some filters get stuck to 0, we can skip them
					break

			end_time = time.time()
			print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

			print(np.squeeze(input_img_data[0]).shape)
			sample = np.squeeze(input_img_data[0])
				
			# normalization
			sample = sample - np.mean(sample)
	
			# low pass filter
			sample_lpf = butter_lowpass_filter(sample,cutoff,fs_ft,order)

			save_name_wav = '%s_filter%d_norm%s.png' % (layer_name,filter_index,str(norm_param))
			plt.clf()
			plt.plot(sample_lpf)
			plt.axis('off')
			plt.savefig(save_path_wav+save_name_wav)

			# perform squared magnitude spectra
			S = librosa.core.stft(sample,n_fft=fftsize,hop_length=fftsize,win_length=fftsize)
			X = np.square(np.absolute(S))
			log_S = np.log10(1+10*X)
			log_S = np.squeeze(log_S.astype(np.float32))
			print(log_S.shape)
			#log_S = np.mean(log_S,axis=1)
			print(log_S.shape)
			fftzed[filter_index] = log_S

		argmaxed = np.argmax(fftzed,axis=1)
		sort_idx = np.argsort(argmaxed)
		sorted_fft = fftzed[sort_idx,:]

		sorted_fft = np.repeat(sorted_fft,repetition,axis=0)
		print(sorted_fft.shape)

		# save figure
		plt.clf()
		plt.imshow(sorted_fft.T)
		plt.gca().invert_yaxis()
		plt.axis('off')
		plt.savefig(save_path+save_name)






































