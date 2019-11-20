import os
import time
import datetime
from glob import glob

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Embedding, Flatten, Input, Reshape, ZeroPadding2D,
                          multiply)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (AveragePooling2D, Conv2D,
                                        Conv2DTranspose, MaxPooling2D,
                                        UpSampling2D, ZeroPadding2D)
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import imageio

# size of the wanted generated image (need to be smaller than the real dataset)
XSIZE = 150
YSIZE = 100
RGB = 3

#size of the latent space
LATENTSIZE = 100

class DCGAN():
	def __init__(self):

		self.img_shape = (XSIZE, YSIZE, RGB)
		optimizer = Adam(0.0002, 0.5)

		''' build discriminator, generator
		self.discriminator = self.create_discriminator()
		self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
		self.generator = self.create_generator()
		'''

		#''' load discriminator, generator if a model already exists
		self.discriminator = load_model('models/dis165000.h5')
		self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
		self.generator = load_model('models/gen165000.h5')
		#'''

		# the combined model take an image as input and output validity from 0 to 1
		self.discriminator.trainable = False # D does not have to train while G is training
		z = Input(shape=(LATENTSIZE,))
		img = self.generator(z)
		valid = self.discriminator(img)

		self.combined = Model(z, valid) 
		self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


	def create_generator(self):

		model = Sequential()

		model.add(Dense(9 * 6 * 256, activation="relu", use_bias = False, input_dim=LATENTSIZE))
		model.add(Reshape((9, 6, 256)))

		model.add(Conv2D(256, kernel_size=(3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(momentum=0.8))

		model.add(UpSampling2D())

		model.add(Conv2D(128, kernel_size=(3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(momentum=0.8))

		model.add(UpSampling2D())

		model.add(ZeroPadding2D(padding = ((0,1),(0,1))))

		model.add(Conv2D(64, kernel_size=(3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(momentum=0.8))

		model.add(UpSampling2D())

		model.add(ZeroPadding2D(padding = ((0,1),0)))

		model.add(Conv2D(32, kernel_size=(3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(momentum=0.8))

		model.add(UpSampling2D())

		model.add(Conv2D(16, kernel_size=(3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(momentum=0.8))

		model.add(Conv2D(RGB, kernel_size=(1,1), padding="same"))
		model.add(Activation("tanh"))

		noise = Input(shape=(LATENTSIZE,))
		img = model(noise)

		model.summary()

		return Model(noise, img)



	def create_discriminator(self):
        
		model = Sequential()

		model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(AveragePooling2D())
		model.add(Dropout(0.25))

		model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(AveragePooling2D())
		model.add(Dropout(0.25))

		model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))

		model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))

		img = Input(shape=self.img_shape)
		validity = model(img)

		model.summary()

		return Model(img, validity)


	def train(self, batch_size=128, save_interval=1000, save_img_interval=50):
        
		#get dataset
		trainResize = self.load_dataset('portrait/*')

		# ones = label for real images
		# zeros = label for fake images
		ones = np.ones((batch_size, 1)) 
		zeros = np.zeros((batch_size, 1))

		# create some noise to track AI's progression
		self.noise_pred = np.random.normal(0, 1, (1, LATENTSIZE))

		epoch = 165000
		d_loss = [0,0]
		g_loss = 0
		beginT = time.time()
		while(1):
			epoch+=1

			# Select a random batch of images in dataset
			idx = np.random.randint(0, trainResize.shape[0], batch_size)
			imgs = trainResize[idx]


			# Sample noise and generate a batch of new images
			noise = np.random.normal(0, 1, (batch_size, LATENTSIZE))
			gen_imgs = self.generator.predict(noise)

			if( epoch < 10000 or d_loss[0] > g_loss/batch_size):
				# train D on real and then fake img
				d_loss_r = self.discriminator.train_on_batch(imgs, ones)
				d_loss_f = self.discriminator.train_on_batch(gen_imgs, zeros)
				d_loss = np.add(d_loss_r , d_loss_f)*0.5

			g_loss = self.combined.train_on_batch(noise, ones)



			if epoch % save_img_interval == 0:
				self.save_imgs(epoch)
				t = str(datetime.timedelta(seconds = time.time() - beginT))
				print ("%d D loss: %f, acc.: %.2f%%, G loss: %f, time: %s" % (epoch, d_loss[0], 100*d_loss[1], g_loss/batch_size, t))
            
			if epoch % save_interval == 0:
				self.discriminator.save('models/dis' + str(epoch) + '.h5')
				self.generator.save('models/gen'  + str(epoch) + '.h5')


	def save_imgs(self, e):

		gen_img = self.generator.predict(self.noise_pred)
		#confidence = self.discriminator.predict(gen_img)

		# Rescale image (rgb 0 to 255)
		gen_img = (0.5 * gen_img + 0.5)*255

		cv2.imwrite('train/%i_%i.png'%(time.time(), e), gen_img[0])

	def load_dataset(self, path):

		try:
			# try to load existing trainResize
			trainResize = np.load('trainResize.npy')
			print('loaded dataset')

		except:
			# else, build trainResize and save it
			trainResize = []
			dos = glob(path)

			for i in tqdm(dos):
				img = cv2.imread(i)
				img = cv2.resize(img,(YSIZE, XSIZE))
				trainResize.append(img)

			cv2.destroyAllWindows()
			trainResize = np.array(trainResize)
			trainResize = trainResize / 127.5 - 1  # Rescale dataset to -1/1 (activation function of G is tanh)
			print(trainResize.shape)
			np.save('trainResize.npy',trainResize)
			print('created dataset')
            
		return trainResize




def plot(file):
	D_loss, G_loss = [], []
	for line in open(file, "r"):
		line  = line.split(" ")
		if(line[0] == '300000'):
			break
		D_loss.append(float(line[2]))
		G_loss.append(float(line[8]))


	df=pd.DataFrame({'x': range(50, 300000, 50), 'D_loss': np.array(D_loss), 'G_loss': np.array(G_loss)})

	# multiple line plot
	palette = plt.get_cmap('Set1')
	num=0
	for column in df.drop('x', axis=1):
		num+=1
		plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

	plt.legend(loc=2, ncol=2)
	plt.title("Loss of D and G")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.savefig("plot/DCGAN_loss.png")


def gif():
	filenames = glob('train/*')#load training image
	with imageio.get_writer('training.gif', mode='I',duration=0.033) as writer:
		for filename in tqdm(filenames):
			image = imageio.imread(filename)
			writer.append_data(image)


if __name__ == '__main__':
	cgan = DCGAN()
	cgan.train(batch_size=32, save_interval=2500, save_img_interval=50)