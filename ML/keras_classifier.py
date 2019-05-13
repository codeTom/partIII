#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 23:43:40 2019

@author: filip
"""
#random seeds
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
from sklearn.model_selection import StratifiedKFold
set_random_seed(2)



from scipy.misc import imsave

import sys

from sklearn.model_selection import StratifiedKFold
import tensorflow as tf


import tensorflow.contrib.slim as slim
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback, ModelCheckpoint
from tensorflow.keras.utils import Sequence
import numpy as np
import os
import math
from glob import glob
import time
from random import randint, shuffle
from scipy.misc import imread
from tensorflow.keras.layers import Conv2D,Conv3D,Dropout, Flatten, MaxPool2D, Softmax, Dense
from tensorflow.keras.layers import UpSampling2D, Reshape, Input, GaussianNoise
from tensorflow.keras.models import Model, load_model
from scipy.misc import imread
from tensorflow.keras.optimizers import SGD, Adam
import cv2 as cv
from preprocessing import loadCellsNpz
from keras_autoencoder2 import encoder, decoder, saveTestImages, model as autoenc_model
import random
import sklearn.metrics as skm
from matplotlib import pyplot as plt
from shutil import copyfile

tf.reset_default_graph()
tf.keras.backend.clear_session()

#this will probably be too large, will need to use a better method
def loadTrainData(smooth=False, datafolder="train_orig"):
    if smooth:
        s="_s"
    else:
        s=""
    pos=loadCellsNpz(f"{datafolder}/positive{s}.npz")
    neg=loadCellsNpz(f"{datafolder}/negative{s}.npz")
    #may=loadCellsNpz("train_orig/maybe.npz")
    res=np.concatenate((neg,pos))
    labs=np.array([0,1]).repeat([len(neg),len(pos)])
    #TODO
    #return labs, res
    lr=list(zip(labs,res))
    random.shuffle(lr)
    lr = [np.array(t) for t in zip(*lr)]
    return lr[0],lr[1]

def loadTestData(include_boring=True, smooth=True):
    if smooth:
        s="_s"
    else:
        s=""
    pos=loadCellsNpz(f"test_data/positive{s}.npz")
    neg=loadCellsNpz(f"test_data/negative{s}.npz")
    if include_boring:
        boring=loadCellsNpz(f"test_data/boring{s}.npz")
        res=np.concatenate((neg,boring,pos))
    else:
        boring=[]
        res=np.concatenate((neg,pos))
    labs=np.array([0,1]).repeat([len(neg)+len(boring),len(pos)])
    lr=list(zip(labs,res))
    random.shuffle(lr)
    lr = [np.array(t) for t in zip(*lr)]
    return lr[0],lr[1]

def loadBoringTestData(smooth=True):
    if smooth:
        imgs=loadCellsNpz(f"test_data/boring_s.npz")
    else:
        imgs=loadCellsNpz(f"test_data/boring.npz")

    labs=np.zeros(imgs.shape[0])
    return labs, imgs

def model(freeze=True, combined_loss=False, from_scratch=False):
    inp=Input(shape=(112,112,3))
    enc=encoder(inp, not freeze) #freeze the autoencoder part
#    x = Dropout(0.5)(enc)
    x = Dense(128, activation='relu', name='classifier/fc1')(enc)
    #x = Dropout(0.5)(x)
#    x = Dense(128, activation='tanh', name='classifier/cl2')(x)
    x = Dense(1, activation='sigmoid', name='classification')(x)
    outs=x
    if combined_loss:
       rec=decoder(enc, not freeze)
       outs=[x,rec]
    m=Model(inputs=inp, outputs=outs)
    if not from_scratch:
        m.load_weights('weights_best.hdf5', by_name=True)
    if combined_loss:
        tf.summary.image("reconstruction", rec*255.0)
        tf.summary.image("original", inp*255.0)
        tf.summary.image("abs_difference", tf.math.abs(inp-rec)*255.0)
    return m

def logloss(y_true, y_pred):
#        print(y_true)
#        print(y_pred)
    positive_weight=1.0
    return -(positive_weight*y_true*tf.log(y_pred+1e-9)+(1-y_true)*tf.log(1.0-y_pred+1e-9))

def runXval(n_folds=10):
    """
    Crossvalidation will only be run for the best model
    """
    smooth=True
    labs,imgs=loadTrainData(smooth)
#    imgs=imgs[12000:14000]
#    labs=labs[12000:14000]
    #test data disabled for the nih run
    testlabs,testimgs=loadTestData(False,smooth)
    labs = np.concatenate((labs,testlabs),axis=0)
    imgs = np.concatenate((imgs,testimgs),axis=0)
    print(f"X-validation on {len(imgs)} samples")
    i=0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    for train_index, test_index in skf.split(imgs, labs):
        #model(freeze=True, combined_loss=False, from_scratch=False):
        print(f"processing fold {i}")
        m=model(True, True, False)
        #prevent exploding encoder gradient
        train(m, True, True, name=f"xval_smooth_combined_rw500_fold{i}_step1", rw=25.0, epochs=20, traindata=(labs[train_index], imgs[train_index]))
        for layer in m.layers:
            layer.trainable=True
        train(m, True, True, name=f"xval_smooth_combined_rw500_fold{i}_step2", rw=800.0, epochs=20, traindata=(labs[train_index], imgs[train_index]))
        #model is compiled inside train so this should work
        train(m, True, True, name=f"xval_smooth_combined_rw500_fold{i}_step3", rw=25.0, epochs=250, traindata=(labs[train_index], imgs[train_index]))

        os.mkdir(f'xval{i}/')
        tf.keras.backend.clear_session()
        res=testCombinedLoss(f"xval{i}/", testdata=(labs[test_index], imgs[test_index]))
        print(f"ROCAUC: {res['fil']['roc_auc']}")
        os.rename("classifier_weights_best.hdf5", f"xval{i}/classifier.hdf5")
        tf.keras.backend.clear_session()
        i+=1


def train(model, smooth=True, combined_loss=False, name="", epochs=200, rw=1.0, traindata=None):
    if traindata is None:
        labs,imgs=loadTrainData(smooth)
    else:
        labs, imgs = traindata
    filepath="classifier_weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    if combined_loss:
        model.compile(optimizer='Adam', loss={'classification':logloss, 'reconstruction':'mse'},
                metrics={'classification':['mse', 'acc', tf.keras.metrics.Precision(),  keras.metrics.Recall()],
                         'reconstruction':['mse']
                         },
                loss_weights={'classification':1.0, 'reconstruction': rw})
    else:
        model.compile(optimizer='Adam', loss=logloss,
                metrics=['mse', 'acc', tf.keras.metrics.Precision(),  keras.metrics.Recall()],
                    )

    train_size=labs.shape[0]
    val_labs=labs[:train_size//15]
    val_imgs=imgs[:train_size//15]/255.0
    #remove all uncertain values from validation (testing only this does not really make sense)
    val_imgs=val_imgs[val_labs!=0.5]
    val_labs=val_labs[val_labs!=0.5]
    train_labs=labs[train_size//15:]
    train_imgs=imgs[train_size//15:]
    idg = keras.preprocessing.image.ImageDataGenerator(
                rotation_range = 179,
                zoom_range = 0.15,
                channel_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True
 #               rescale=1.0/255.0
                )
    def batch_gen(imgs,labs, batch_size):
        gen=idg.flow(imgs, labs, batch_size=batch_size)
        i=0
        while True:
            i+=batch_size
            res=gen.next()
            yield res[0]/255.0, {'reconstruction':res[0]/255.0, 'classification':res[1]}

    if combined_loss:
        merged_summary_op = tf.summary.merge_all()
        datagen=batch_gen(train_imgs, train_labs,64)
        test_batch=val_imgs
        callback=TensorBoard(log_dir=f"./log_dir/{name}"+str(int(time.time())))
        callback.set_model(model)
        sampls=np.arange(0,len(test_batch)).astype('int')
        def get_images(epoch, logs):
            s=tf.keras.backend.get_session()
            tests=test_batch[np.random.choice(sampls, 3)]
            #        print(tests)
            sums=s.run(merged_summary_op, feed_dict={'input_1:0':tests})
            callback.writer.add_summary(sums, epoch)
        get_images(0,None)
        image_callback=LambdaCallback(on_epoch_end=get_images)
        model.fit_generator(datagen,
                        epochs=epochs,
                        steps_per_epoch=len(train_imgs)//64,
                        validation_data=(val_imgs, {'reconstruction': val_imgs,
                                                    'classification': val_labs}),
                        verbose=1,
                        callbacks=[checkpoint,
                                   callback,
                                   image_callback
                                  ]
                        )
    else:
        model.fit_generator(idg.flow(train_imgs,train_labs, batch_size=64),
                        epochs=epochs,
                        validation_data=(val_imgs, val_labs),
                        verbose=1,
                        callbacks=[checkpoint,
                                   TensorBoard(log_dir=f"./log_dir/{name}"+str(int(time.time()))),
                                  ]
                        )
    #model.fit(x=train_imgs,y=train_labs, batch_size=64, epochs=200, validation_data=(val_imgs, val_labs), verbose=1, callbacks=[checkpoint, TensorBoard(log_dir="./log_dir/clas"+str(int(time.time())))])

def testCombinedLoss(outd='combined_test_smooth_new', testdata=None):
    clas=model(True, False, True) #classifier only
    clas.compile(optimizer='Adam', loss=logloss, metrics=['accuracy'])
    autoenc=autoenc_model()
    autoenc.compile(optimizer='Adam', loss='mean_squared_error')
    clas.load_weights('classifier_weights_best.hdf5', by_name=True)
    autoenc.load_weights('classifier_weights_best.hdf5', by_name=True)
#    os.mkdir(outd)
    return runTests(clas, autoenc, True, outd, testdata=testdata)

def runTests(model, autoenc=None, smooth=False, outf="", testdata=None):
    """
    Load test images and run tests
    use autoenc to provide autoencoder settting
    TODO
    """
    if testdata is None:
        data_wb=loadTestData(True,smooth)
        data_nb=loadTestData(False, smooth)
        res_wb=model.predict(data_wb[1]/255.0, batch_size=64)
    else:
        data_nb=testdata
        data_wb=(np.array([]),np.array([]))

    #this is insanely inefficient, could get just boring cells in the second ru
    res_nb=model.predict(data_nb[1]/255.0, batch_size=64)
    def getStats(p, r, title_prefix="", imgs=None, recons=None):
        """
        Get the statistics
        {ROC, PRC}+AUCs, {Accuracy, Recall, Precision} at thresholds
        TODO: save the data for plotting multiple in the same
        """
        all_data={}
        roc_fpr, roc_tpr, roc_th=skm.roc_curve(r,p)
        roc_auc=skm.auc(roc_fpr, roc_tpr)
        prc_p, prc_r, prc_th = skm.precision_recall_curve(r,p)
        prc_auc=skm.auc(roc_fpr, roc_tpr)
        acc_th=[]
        acc=[]
        conmat=skm.confusion_matrix(r, p.round())
        all_data['conmat']=conmat
        all_data['f1']=skm.f1_score(r,p.round())
        all_data['true_pred']=[r,p]
        fnfix=title_prefix.replace(' ','_').replace(',','')
        #now get the accuracy, balanced accuracy at different
        #thresholds
        for th in np.linspace(0,1,100):
            pred=np.zeros_like(p).astype('uint8')
            pred[p>th]=1
            acc_th.append(th)
            acc.append(skm.accuracy_score(r,pred))
        #plot the data
        plt.figure()
        plt.title(f"{title_prefix}ROC")
        plt.ylabel("True positive rate")
        plt.xlabel("False positive rate")
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.plot(roc_fpr,roc_tpr)
        all_data['roc'] = [roc_fpr, roc_tpr, roc_th]
        all_data['roc_auc'] = roc_auc
        plt.text(0.7,0.2,f"AUC: {roc_auc:.3f}",bbox={'facecolor':'black','alpha':0.2})
        plt.savefig(f"{outf}/{fnfix}_ROC.png", dpi=250)
        plt.gcf().clear()
        plt.figure()
        plt.title(f"{title_prefix}Precision-Recall Curve")
        plt.ylabel("Recall")
        plt.xlabel("Precision")
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.plot(prc_p,prc_r)
        all_data['prc'] = [prc_p,prc_r, prc_th]
        all_data['prc_auc']= prc_auc
        plt.text(0.7,0.2,f"AUC: {prc_auc:.3f}",bbox={'facecolor':'black','alpha':0.2})
        plt.savefig(f"{outf}/{fnfix}_PRC.png", dpi=250)
        plt.gcf().clear()

        plt.figure()
        plt.title(f"{title_prefix} Accuracy at thresholds")
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.ylabel("Accuracy")
        plt.xlabel("Threshold")
        plt.plot(acc_th,acc)
        all_data['atc'] = [acc_th, acc]
        plt.savefig(f"{outf}/{fnfix}_AT.png", dpi=250)
        plt.gcf().clear()

        print(f"ROC-AUC: {roc_auc}")
        print(f"PRC-AUC: {prc_auc}")
        pos=r.sum()
        total=len(r)
        print(f"Test set has positives: {pos} out of {total} ({(pos/total*100):.1}%)")
        closs=[]
        if autoenc is not None:
            for i in range(len(imgs)):
                closs.append(model.evaluate(np.array([imgs[i]]), np.array([r[i]]), batch_size=1))
            closs=np.array(closs)
            reloss=((imgs-recons)**2).mean(axis=(1,2,3))
            all_data['losses'] = np.concatenate((np.expand_dims(reloss, axis=1),closs),axis=1)
        return all_data
    all_data={}
    rec_nb=None
    rec_wb=None
    if testdata is None:
        data_bo=loadBoringTestData(smooth)
    else:
        data_bo = (np.array([]),np.array([]))
    rec_bo=None
    if autoenc is not None:
        rec_nb = autoenc.predict(data_nb[1]/255.0)
        if testdata is None:
            rec_wb = autoenc.predict(data_wb[1]/255.0)
            rec_bo = autoenc.predict(data_bo[1]/255.0)
    print("Stats excluding boring:")
    all_data['fil'] = getStats(res_nb, data_nb[0], "Filtered set, ", imgs=data_nb[1]/255.0, recons=rec_nb)
    if testdata is None:
        print("Stats including boring:")
        all_data['all'] = getStats(res_wb, data_wb[0], "All cells, ", imgs=data_wb[1]/255.0, recons=rec_wb)
        res_bo=model.predict(data_bo[1]/255.0, batch_size=64)
        print("Stats boring only:")
        all_data['boring'] = getStats(res_bo, data_bo[0], "Removed cells only, ", imgs=data_bo[1]/255.0, recons=rec_bo)

        if autoenc is not None:
            if smooth:
                s="_s"
            else:
                s=""
            all_data['rec']={}
            #TODO get the reconstruction loss vs classification error plot
            os.mkdir(f"{outf}/positive/")
            all_data['rec']['p']=saveTestImages(autoenc, f"test_data/positive{s}.npz", f"{outf}/positive/")
            os.mkdir(f"{outf}/negative/")
            all_data['rec']['n']=saveTestImages(autoenc, f"test_data/negative{s}.npz", f"{outf}/negative/")
            os.mkdir(f"{outf}/boring/")
            all_data['rec']['b']=saveTestImages(autoenc, f"test_data/boring{s}.npz", f"{outf}/boring/")
    elif autoenc is not None:
        #generate images from the test dataset
        os.mkdir(f"{outf}/images")
        saveTestImages(autoenc, outfolder=f"{outf}/images/", testdata=(testdata[0]/255.0, testdata[1]/255.0))
#    print(all_data)
    np.save(f"{outf}/data.npy",all_data)
    return all_data


#model(freeze=True, combined_loss=False, from_scratch=False)
#runTests(model, autoenc=None, smooth=False, outf=""):
#testCombinedLoss(outd='combined_test_smooth_new')
def getFailTestData():
    files=glob("failtest/*.png")
    labs=[]
    imgs=[]
    for f in files:
        print("loading {f}")
        if "p.png" in f:
            labs.append(1)
        else:
            labs.append(0)
        imgs.append(imread(f))
    return (np.array(labs),np.array(imgs))

'''
copyfile("weights_best_smooth.hdf5","weights_best.hdf5")
m=model(True, False, False)
train(m,True,name="frozen_smooth")
os.mkdir('frozen_smooth/')
runTests(m, autoenc=None, smooth=True, outf="frozen_smooth/")
os.rename("classifier_weights_best.hdf5", "trained/frozen_smooth.hdf5")
tf.keras.backend.clear_session()

copyfile('weights_best_sharp.hdf5', 'weights_best.hdf5')
m=model(True, False, False)
train(m,False, name="frozen_sharp")
os.mkdir('frozen_sharp/')
runTests(m, autoenc=None, smooth=False, outf="frozen_sharp/")
os.rename("classifier_weights_best.hdf5", "trained/frozen_sharp.hdf5")
tf.keras.backend.clear_session()

copyfile("weights_best_smooth.hdf5","weights_best.hdf5")
m=model(False, False, False)
train(m,True,name="unfrozen_smooth")
os.mkdir('unfrozen_smooth/')
runTests(m, autoenc=None, smooth=True, outf="unfrozen_smooth/")
os.rename("classifier_weights_best.hdf5", "trained/unfrozen_smooth.hdf5")
tf.keras.backend.clear_session()

copyfile('weights_best_sharp.hdf5', 'weights_best.hdf5')
m=model(False, False, False)
train(m,False, name="unfrozen_sharp")
os.mkdir('unfrozen_sharp/')
runTests(m, autoenc=None, smooth=False, outf="unfrozen_sharp/")
os.rename("classifier_weights_best.hdf5", "trained/unfrozen_sharp.hdf5")
tf.keras.backend.clear_session()

m=model(False, False, True)
train(m,True,name="scratch_smooth", epochs=500)
os.mkdir('scratch_smooth/')
runTests(m, autoenc=None, smooth=True, outf="scratch_smooth/")
os.rename("classifier_weights_best.hdf5", "trained/scratch_smooth.hdf5")
tf.keras.backend.clear_session()

m=model(False, False, True)
train(m,False, name="scratch_sharp", epochs=500)
os.mkdir('scratch_sharp/')
runTests(m, autoenc=None, smooth=False, outf="scratch_sharp/")
os.rename("classifier_weights_best.hdf5", "trained/scratch_sharp.hdf5")
tf.keras.backend.clear_session()


#combined loss
for i in range(1,3000,200):
    rw=i*1.0
    copyfile("weights_best_smooth.hdf5","weights_best.hdf5")
    m=model(False, True, False)
    train(m, True, True, name=f"combined_smooth_rw_{i}", epochs=200)
    os.mkdir(f'combined_smooth_rw{i}/')
    testCombinedLoss(f"combined_smooth_rw{i}/")
    os.rename("classifier_weights_best.hdf5", f"trained/combined_smooth_rw{i}.hdf5")
    tf.keras.backend.clear_session()

    copyfile('weights_best_sharp.hdf5', 'weights_best.hdf5')
    m=model(False, True, False)
    train(m,False, True, name=f"combined_sharp_rw{i}_", epochs=300, rw=rw)
    os.mkdir(f'combined_sharp_rw{i}/')
    testCombinedLoss(f"combined_sharp_rw{i}/")
    os.rename("classifier_weights_best.hdf5", f"trained/combined_sharp_rw{i}.hdf5")
    tf.keras.backend.clear_session()

m=model(False, True, True)
train(m,True,True,name="scratch_combined_smooth", epochs=500)
os.mkdir('scratch_combined_smooth/')
testCombinedLoss("scratch_combined_smooth/")
os.rename("classifier_weights_best.hdf5", "trained/scratch_combined_smooth.hdf5")
tf.keras.backend.clear_session()

m=model(False, True, True)
train(m,False, True, name="scratch_combined_sharp", epochs=500)
os.mkdir('scratch_combined_sharp/')
testCombinedLoss("scratch_combined_sharp/")
tf.keras.backend.clear_session()
os.rename("classifier_weights_best.hdf5", "trained/scratch_combined_sharp.hdf5")
'''

#os.rename('weights_best.hdf5', 'weights_best_sharp.hdf5')
#os.rename('weights_best_smooth.hdf5', 'weights_best.hdf5')

#m=model(False, True, False)
#train(m, True)
#testCombinedLoss()
#os.mkdir('classifier_testing')
sets=['classifier_sharp_frozen',
'classifier_sharp_unfroz',
'classifier_smooth_frozen',
'classifier_smooth_unfrozen',
'classifier_from_scratch']
#runXval(10)
#or s in sets:
#   m.load_weights(f"{s}.hdf5", by_name=True)
#   os.mkdir(f"classifier_testing/{s}")
#   runTests(m, autoenc=None, smooth='smooth' in s, outf=f"classifier_testing/{s}")

#.load_weights('classifier_from_scratch.hdf5')
#s.mkdir('classifier_testing/fromscratch')
#unTests(m, autoenc=None, smooth=False
#m.load_weights('classifier_weights_best.hdf5', by_name=True)
#runTests(m)
#train(m)
#m.save_weights('class_final.hdf5')
#os.mkdir("failuretest/")
testCombinedLoss(f"failuretest/",testdata=getFailTestData())
