#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:40:06 2019

@author: filip
"""

import imutils.convenience as imutils
import numpy as np
import cv2 as cv
import sys
import random
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import scipy.optimize as opt
import itertools
from glob import glob
import re
import os
import json
import scipy.stats as stats
from scipy.optimize import minimize
from shapely.geometry import Point, Polygon
from scipy.misc import imread,imsave

def makePolys(boxes=(5,1), maxx=3280, maxy=2464):
    polys=[]
    xstep=maxx//boxes[0]
    ystep=maxy//boxes[1]
    for i in range(boxes[0]):
        for j in range(boxes[1]):
            polys.append(Polygon([(i*xstep,j*ystep), ((i+1)*xstep,j*ystep),
                          ((i+1)*xstep, (j+1)*ystep), (i*xstep, (j+1)*ystep)]))
    return polys

def selectShifts(regions, data, calfn=None):
    """
    Select the best offset for each region based on calibration data in the for
    [[x,y,z]]
    regions are a list of polygons
    this is a simple implementation and could be made faster
    """
    res=[]
    stds=[]
    for poly in regions:
        vals=[]
        if calfn is not None:
            res.append(calfn(poly.centroid.x,poly.centroid.y))
            stds.append(0)
            continue

        for d in data:
            if poly.contains(Point((d[0],d[1]))):
                vals.append(d[2])
        vals=np.array(vals)
        res.append(vals.mean())
        stds.append(vals.std())
    return np.array(res), np.array(stds)

def fittedFn(x,y):
    x=x-3280//2
    y=y-2464//2
    return -3.85833674e-02*x-6.08879819e-02*y+9.66707822e-06*(y**2+x**2)

def selectCompensation(stepsize=2,boxes=(5,2)):
    res=np.loadtxt('rawbeadxyz.npy')
    regions=makePolys(boxes=boxes)
    shifts,devs = selectShifts(regions, res)#@, calfn=fittedFn)
    rounded=np.round(shifts/stepsize)*stepsize
    return regions,rounded

def getMask(image, poly):
    #this only works for rectangular grid aligned with the axes
    cords=np.array(poly.exterior.coords)
    image[int(cords[0,1]):int(cords[2,1]),int(cords[0,0]):int(cords[1,0]),:]=1
    #print(image.sum())
    return image

def assemble(folder="/media/filip/data1/project_data/focus_zstack3/", startz=244550, noshift=False,boxes=(5,2)):
    #find range of zs available
    files=glob(f"{folder}/*z*.jpg")
    files=sorted(files)
    if len(files)==0:
        print(folder)
        print(files)
        raise Exception("Invalid")
    zr=re.compile(".*z(\d+).jpg")
    zmin=int(zr.search(files[0]).group(1))
    zmax=int(zr.search(files[-1]).group(1))
    print(f"min:{zmin}, max: {zmax}")
    if startz < zmin or startz>zmax:
       # print("Invalid data")
        return False
    regions,compensations=selectCompensation(boxes=boxes)
    baseimg=imread(glob(f"{folder}/*z{startz}.jpg")[0])
    #print(baseimg.shape)
    final=np.zeros_like(baseimg)
    for r,c in zip(regions,compensations):
        c*=-1
        if startz+c > zmax:
            print(f"Z>max, using maxz")
            z=zmax
        elif startz+c < zmin:
            print(f"Z<min, using minz")
            z=zmin
        else:
            z=startz+c
        if np.isnan(z):
            print("got NaN, using base")
            z=startz
        z=int(z)
        flist=glob(f"{folder}/*z{z}.jpg")
        if len(flist)==0:
            print(f"failed to find file: {z}")
            return
        img=imread(glob(f"{folder}/*z{z}.jpg")[0])
        if not noshift:
            shift=cv.estimateRigidTransform(img,baseimg,False)
            shifted=cv.warpAffine(img, shift, (img.shape[1],img.shape[0]))#, b[, dst[, flags[, borderMode[, borderValue]]]]	)
        else:
            shifted=img
        mask=np.zeros_like(final)
        mask=getMask(mask,r)
        final+=mask*shifted
        #get the offset
    #load the start image
    return final

def saveImages(bx=5,by=2): 
    original=imread(glob("/media/filip/data1/project_data/focus_test_zstack1/*z245423*.jpg")[0])
    shifted=assemble(folder="/media/filip/data1/project_data/focus_test_zstack1/", startz=245423,noshift=False, boxes=(bx,by))
    unshifted=assemble(folder="/media/filip/data1/project_data/focus_test_zstack1/", startz=245423, noshift=True, boxes=(bx,by))
    folder=f"assemble_{bx}{by}"
    os.mkdir(folder)
    #cv.imshow('orig', cv.resize(original,(600,800)))
    imsave(f"{folder}/orig.jpeg",original)
    #cv.imshow('shifted',cv.resize(shifted,(600,800)))
    imsave(f"{folder}/shifted.jpeg",shifted)
    imsave(f"{folder}/unshifted.jpeg",unshifted)

def testBoxes(bx):
    for i in range(1,bx+1,2):
        saveImages(bx,i)

testBoxes(3)
testBoxes(5)
testBoxes(7)