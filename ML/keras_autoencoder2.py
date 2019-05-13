#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:57:18 2019

@author: filip
"""

from scipy.misc import imsave

import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback, ModelCheckpoint
from tensorflow.keras.utils import Sequence
import numpy as np
import math
from glob import glob
import time
from random import randint, shuffle
from scipy.misc import imread
from tensorflow.keras.layers import Conv2D,Conv3D,Dropout, Flatten, MaxPool2D, Softmax, Dense, Lambda
from tensorflow.keras.layers import UpSampling2D, Reshape, Input, GaussianNoise
from tensorflow.keras.models import Model, load_model
from scipy.misc import imread
from tensorflow.keras.optimizers import SGD, Adam
import cv2
from preprocessing import loadCellsNpz


USE_TPU=False
#input size 270x270?
data_dir="./train_orig/"
#data_dir='/media/filip/data1/project_data/ml_data_summer/segment/new_segmented/'
image_size=112
batch_size=64
inout={}
dataset=None
datacount=0
sess=None

#this will probably be too large, will need to use a better method
def load_all_data():
    imgs=[]
    for f in glob(data_dir+"*.png"):
        img=imread(f,mode="RGB")
        imgs.append(img)
    return np.array(imgs)

def encoder(tensor, trainable=True, name_prefix="encoder/"):
    x = tensor # GaussianNoise(0.05)(tensor)
    x = Conv2D(64, (3,3), activation='relu', padding='same', trainable=trainable, name=f"{name_prefix}c1")(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same', trainable=trainable, name=f"{name_prefix}c2")(x)
    x = MaxPool2D((2, 2), padding='same', name=f"{name_prefix}max_p1")(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', trainable=trainable, name=f"{name_prefix}c3")(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', trainable=trainable, name=f"{name_prefix}c4")(x)
    x = MaxPool2D((2, 2), padding='same', name=f"{name_prefix}max_p2")(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', trainable=trainable, name=f"{name_prefix}c5")(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', trainable=trainable, name=f"{name_prefix}c6")(x)
    x = MaxPool2D((2, 2), padding='same', name=f"{name_prefix}max_p3")(x)
#    x = Conv2D(32, (3,3), padding='same')(x)
#    x = MaxPool2D((2, 2), padding='same')(x)
#    x = Conv2D(128, (3,3), padding='same')(x)
#    x = MaxPool2D((2, 2), padding='same')(x)
    x = Flatten(name=f"{name_prefix}flatten")(x)
    x = Dropout(0.5, name=f"{name_prefix}dropout")(x)
    x = Dense(64, name=f"{name_prefix}fc1", activation='relu', trainable=trainable)(x)
    return x

def decoder(vector, trainable=True, name_prefix="decoder/"):
    x = Dense(2*2*6272, trainable=trainable, name=f"{name_prefix}fc1", activation='relu')(vector)
    x = Reshape((14,14,128), name=f"{name_prefix}reshape")(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', trainable=trainable, name=f"{name_prefix}c1")(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', trainable=trainable, name=f"{name_prefix}c2")(x)
    x = UpSampling2D((2,2), name=f"{name_prefix}up1")(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', trainable=trainable, name=f"{name_prefix}c3")(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', trainable=trainable, name=f"{name_prefix}c4")(x)
    x = UpSampling2D((2,2), name=f"{name_prefix}up2")(x)
    x = Conv2D(64, (3,3), padding='same', activation='relu', trainable=trainable, name=f"{name_prefix}c5")(x)
    x = Conv2D(64, (3,3), padding='same', activation='relu', trainable=trainable, name=f"{name_prefix}c6")(x)
    x = UpSampling2D((2,2), name=f"{name_prefix}up3")(x)
#    x = Conv2D(64, (3,3), padding='same')(x)
#    x = UpSampling2D((2,2))(x)
#    x = Conv2D(64, (3,3), padding='same')(x)
#    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), activation='sigmoid', trainable=trainable, padding='same', name="reconstruction")(x) #RGB->3 channels
    return x

def model():
    ind=Input(shape=(image_size,image_size,3))
    latent=encoder(ind)
    latent=Lambda(lambda x: x, name='latent')(latent)
    recon=decoder(latent)
#    loss = tf.nn.l2_loss(ind-recon)
#    tf.summary.scalar("Total_Loss", loss)
    tf.summary.image("original", ind*255.0)
    tf.summary.image("reconstruction", recon*255.0)
    tf.summary.image("abs_difference", tf.math.abs(ind-recon)*255.0)
    autoencoder = Model(ind, recon)
    #keras.utils.plot_model(autoencoder, to_file='model.png', show_shapes=True)
    return autoencoder

class CellSequence(Sequence):
    def __init__(self, filenames, batch_size):
        """
            Load the first batch (this also gives the length)
            and the last batch to check the remainder size
            to get len.
            All files except last must have the same num of cells
        """
        self.fnames=filenames
        self.batch_size = batch_size
        self.shard = list(loadCellsNpz(filenames[0]))
        self.loaded_shard = 0
        self.shard_size = len(self.shard)
        self.idg = keras.preprocessing.image.ImageDataGenerator(
                rotation_range = 179,
                zoom_range = 0.15,
                channel_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=True
 #               rescale=1.0/255.0
                )

        if len(filenames) == 1:
            self.single_shard = True
            self.len = self.shard_size
        else:
            self.single_shard = False
            self.len = (len(filenames)-1)*self.shard_size
            self.len += len(loadCellsNpz(filenames[-1]))
        print(f"Loaded dataset with len: {self.len}")

    def __len__(self):
        return int(self.len/self.batch_size)

    def loadShard(self, shard):
        if self.loaded_shard == shard:
#            print(f"Not loading shard {shard}")
            return
        self.shard = list(loadCellsNpz(self.fnames[shard]))
        self.loaded_shard == shard
#r       print(f"Loaded shard {shard}")

    #returns batch
    def __getitem__(self, idx):
        #check which shard to load
        if self.single_shard:
            cells = self.shard[idx*self.batch_size:(idx+1)*self.batch_size]
        else:
            shardnum=idx*self.batch_size//self.shard_size
            shard_start_i = idx*self.batch_size - shardnum*self.shard_size
            from_next = max(0, shard_start_i+self.batch_size-self.shard_size-1) #-1?
            self.loadShard(shardnum)
            if from_next == 0:
                cells = self.shard[shard_start_i:shard_start_i+self.batch_size]
            else:
                cells = self.shard[shard_start_i:]
                self.loadShard(shardnum+1)
                cells+=self.shard[0:from_next]
                print(f"cell len:{len(cells)}")
        imgs = np.array([self.idg.random_transform(cell)/255.0 for cell in cells])
        return imgs,imgs


class DataSequence(Sequence):
    def __init__(self, filenames, batch_size):
        self.fnames=filenames
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.fnames) / float(self.batch_size)))

    #returns batch
    def __getitem__(self, idx):
        batch_x = self.fnames[idx * self.batch_size:(idx + 1) * self.batch_size]
        imgs = np.array([
            cv2.resize(imread(file_name,mode="RGB"),(112,112))
               for file_name in batch_x])/255.0
        return imgs,imgs

    #def get_sequence():

def get_cell_sequence():
    files=[f"{data_dir}/rem_s.npz"]#,f"{data_dir}/positive_s.npz",f"{data_dir}/negative_s.npz"] #change here for more npzfiles
    return CellSequence(files, batch_size)

def get_datasequence():
    files=glob(data_dir+"*.png")
    return DataSequence(files, batch_size)

#TODO: check summaries
def train(autoencoder):
    ds=get_cell_sequence()
    ims=ds[0][0]
    test_batch=ds[0][0]
    print("images loaded")
    callback=TensorBoard(log_dir="./log_dir/deeper_smooth"+str(int(time.time())))
    callback.set_model(m)
    merged_summary_op = tf.summary.merge_all()
    sampls=np.arange(0,len(test_batch)).astype('int')
    print(sampls)
    def get_images(epoch, logs):
        s=tf.keras.backend.get_session()
        tests=test_batch[np.random.choice(sampls, 3)]
#        print(tests)
        sums=s.run(merged_summary_op, feed_dict={'input_1:0':tests})
        callback.writer.add_summary(sums, epoch)
    get_images(0, None)
    image_callback=LambdaCallback(on_epoch_end=get_images)
    filepath="weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    #val=ims[0:10]
    #keras.utils.plot_model(autoencoder, to_file='model.png', show_shapes=True)
    sgd=SGD(lr=0.001, decay=5e-5) #after batch lr=lr/(1+decay*iterations)
    autoencoder.compile(optimizer='Adam', loss='mean_squared_error')
    autoencoder.fit_generator(generator=ds,
                              epochs=300,
                              shuffle=True,
                    #validation_data=(val, val),
                    callbacks=[callback, image_callback, checkpoint])
    return autoencoder

def saveTestImages(m, npz_file="test_data/0.npz", outfolder="test_img", testdata=None):
    #load the test data
    if testdata is None:
        td = loadCellsNpz(npz_file)/255.0 #trained on 0->1 data
    else:
        td = testdata[1]
    res = m.predict(td, batch_size=64)
    losses=[]
    for i in range(len(td)):
        imsave(f"{outfolder}/{i}_r.png",255*res[i])
        imsave(f"{outfolder}/{i}_o.png", 255*td[i])
        imsave(f"{outfolder}/{i}_ad.png", 255*np.abs(res[i]-td[i]))
        loss = ((res[i]-td[i])**2).mean()/td[i][td[i]!=0].mean()
        losses.append(loss)
        print(f"{npz_file}_{i} {loss}")
    return np.array(losses)

def runTest(smooth=False):
    if smooth:
        s="_s"
    else:
        s=""
    m=model()
    os.mkdir(f"reconstruction_test{s}")
    m.load_weights('weights_best.hdf5')
    os.mkdir(f"reconstruction_test{s}/positive/")
    saveTestImages(m, f"test_data/positive{s}.npz", f"reconstruction_test{s}/positive/")
    os.mkdir(f"reconstruction_test{s}/negative/")
    saveTestImages(m, f"test_data/negative{s}.npz", f"reconstruction_test{s}/negative/")
    os.mkdir(f"reconstruction_test{s}/boring/")
    saveTestImages(m, f"test_data/boring{s}.npz", f"reconstruction_test{s}/boring/")



#runTest()
#exit(0)
if __name__ == "__main__":
    #runTest(False)
    #exit(0)
    m=model()
    print(m.summary())
    #m =load_model('autoencoder.h5')
    #m.load_weights('weights_best.hdf5')
#   train(m)
#   m.save_weights('autoencoder_weights.h5')
#   m.save('autoencoder.h5')
    #m.load_weights('weights_best.hdf5')
    #runTest(True)
    m.load_weights('weights_best_sharp.hdf5')
    runTest(False)
