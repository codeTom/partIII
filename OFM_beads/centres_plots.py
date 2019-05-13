#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 00:26:59 2019

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
import gc
from frst.frst import frst

files=["x0.000_y0.000",
           "x0.000_y1000.000",
           "x0.000_y250.000",
           "x0.000_y500.000",
           "x0.000_y750.000",
            "x1000.000_y0.000",
            "x1000.000_y1000.000",
            "x1000.000_y250.000",
            "x1000.000_y500.000",
            "x1000.000_y750.000",
            "x250.000_y0.000",
            "x250.000_y1000.000",
            "x250.000_y250.000",
            "x250.000_y500.000",
            "x250.000_y750.000",
            "x500.000_y0.000",
            "x500.000_y1000.000",
            "x500.000_y250.000",
            "x500.000_y500.000",
            "x500.000_y750.000",
            "x750.000_y0.000",
            "x750.000_y1000.000",
            "x750.000_y250.000",
            "x750.000_y500.000",
            "x750.000_y750.000"
]

def cutRect(rect, image):
    return image[int(rect[0][0]):int(rect[1][0]),int(rect[0][1]):int(rect[1][1])]

def grayData(image,rect):
    return cutRect(rect, cv.cvtColor(image, cv.COLOR_BGR2GRAY))

#A*exp(-((x-x0)^2+(y-y0)^2)/s^2)+B
def gauss(p, A, B, x0, y0, s):
 #   s=15 #TODO remove
    res=A*np.exp(-((p[1]-x0)*(p[1]-x0)+(p[0]-y0)*(p[0]-y0))/s/s)+B
    return res.ravel()

#A*exp(-(x-x0)^2/sx^2-(y-y0)^2/sy^2)+B
def gaussEl(p, A, B, x0, y0, sx, sy):
    res=A*np.exp(-(p[1]-x0)*(p[1]-x0)/sx/sx-(p[0]-y0)*(p[0]-y0)/sy/sy)+B
    return res.ravel()

#fit gaussian, cyl=True to force cylindrical symmetry
def fitGauss(data,cyl=False,xc=None, yc=None, direction="auto"):
    xx, yy = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    poss=np.array(list(zip(np.reshape(xx,(-1)),
                           np.reshape(yy,(-1)))))

    ys = np.reshape(data,(-1)) #image values
    #start at max poss has [y,x] format as all of opencv
    maxp=poss[ys==np.max(ys)][0]
    poss=np.transpose(poss)
    sign=-1
    if xc:
        #compare center to edge brightness
        if yc>=0 and xc>=0 and yc<data.shape[0] and xc<data.shape[1] and data[int(yc)][int(xc)] >= data[1][1]:
            sign = 1
        elif yc>=0 and xc>=0 and yc<data.shape[0] and xc<data.shape[1]:
            sign = -1
        maxp[1]=xc
        maxp[0]=yc
    if direction != "auto":
        sign=direction
    if(cyl):
        initial_guess = (sign*20,data.mean(),maxp[1],maxp[0],
                         data.shape[0]//5)
        popt, pcov = opt.curve_fit(gauss, poss,ys, p0 = initial_guess,
                                   maxfev=2000)
    else:
        initial_guess = (sign*20,data.mean(),maxp[1],maxp[0],
                         data.shape[0]//5,data.shape[1]//5)
        popt, pcov = opt.curve_fit(gaussEl, poss,ys, p0 = initial_guess,
                                   maxfev=2000)
    return popt,pcov

def procImage(image,rect, smooth):
    data = grayData(image,rect)
    data = cv.blur(data, (smooth,smooth))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xx, yy = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    ax.scatter3D(xx, yy, data)

def distCentres(image, draw=True, blur=2, pn=None):
    image=image.copy()
    if pn is None:
        return np.array(distCentres(image, draw, blur,1)+distCentres(image, draw, blur,-1))
    if image.shape[0] > 500: #the bottom of ofm images is weird
        img=cv.blur(image[:-150,:],(blur,blur))
    else:
        img=cv.blur(image,(blur,blur))
    mn=img.mean()
    if pn == 1:
        thresh_val = mn+11
        ret3, th = cv.threshold(img,thresh_val,255,cv.THRESH_BINARY)
    else:
        thresh_val = mn-11
        ret3, th = cv.threshold(img,thresh_val,255,cv.THRESH_BINARY_INV)
    #print(f"mean:{mn}")

    dist=cv.distanceTransform(th, cv.DIST_L2, 3)
    dist*=255.0/dist.max()
    ret,th=cv.threshold(dist.astype('uint8'),0,255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    cnts = cv.findContours(th.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    centres=[]
    for c in cnts:
        moms=cv.moments(c)
        if moms["m00"] == 0:
            #print("Zero moment, skipping")
            continue
        cx = moms["m10"] / moms["m00"]
        cy = moms["m01"] / moms["m00"]
        sz=int(cx-c[:,:,0].min())
        #print(f"{cx}:{cy}:{sz}")
        if draw:
            cv.circle(image, (int(cx),int(cy)), sz, 0, 2)
        centres.append([cx,cy,sz]) #rough estimate of size
    if draw:
        cv.imshow('th', cv.resize(th ,(800,600)))
        cv.imshow('dist',cv.resize(dist.astype('uint8'), (800,600)))
        cv.imshow('image', cv.resize(image ,(800,600)))
        cv.waitKey(0)
        cv.destroyAllWindows()
    return centres

def houghCentres(image, draw=True, blur=1, min_radius=2, max_radius=12, min_distance=4, threshold=False, houghparam1=None, thresh_val=158):

    """
    Detect circles using Hough Circles in opencv, reduce noise by appling blur
    before to improve performance
    """
    grey=True
    if len(image.shape) == 3:
        grey=False
    #hough needs grayscale
    if not grey:
        img=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        img=image
    img=cv.blur(img, (blur,blur))
    if threshold:
        #th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C , cv.THRESH_BINARY, 9, 0)
        #img=cv.blur(img, (3,3))
        ret3, th = cv.threshold(img,thresh_val,255,cv.THRESH_BINARY_INV)
        #cv.imshow('th', cv.resize(th ,(800,600)))
        #cv.waitKey(0)
        img = th
    if houghparam1 is None:
        houghparam1=18 if threshold else 14

    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, min_distance,
                              param1=24, param2=houghparam1, minRadius=min_radius, maxRadius=max_radius)

    if circles is None:
        return None

    if not draw:
        return circles[0,:]

    if grey:
        cenc=0
        circ=255
    else:
        cenc=(0,0,255)
        circ=(0,255,0)
    for i in circles[0,:]:
        #draw the outer circle
        cv.circle(image,(i[0],i[1]),i[2],circ,2)
        #draw the center of the circle
        cv.circle(image,(i[0],i[1]),2,cenc,3)
    #cv.imshow('locations', cv.resize(image,(800,600)))
    #if threshold:
        #cv.imshow('th', cv.resize(th ,(800,600)))
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    return circles[0,:]

def gaussCentre(image, draw=False,xc=None, yc=None, direction="auto"):
    if(len(image.shape)==3):
        img=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        cenc=(0,0,255)
    else:
        img=image
        cenc=0
    res,cov=fitGauss(img,True, xc,yc, direction=direction)
    if draw:
        cv.circle(image,(int(round(res[2])),int(round(res[3]))),2, cenc, 3)
    return res

def frstCentre(image, draw=False):
    if image.ndim>2:
        img=image.mean(axis=2)
    else:
        img=image.copy()
    #Performs fast radial symmetry transform
    #img: input image, grayscale
    #radii: integer value for radius size in pixels (n in the original paper); also used to size gaussian kernel
    #alpha: Strictness of symmetry transform (higher=more strict; 2 is good place to start)
    #beta: gradient threshold parameter, float in [0,1]
    #stdFactor: Standard deviation factor for gaussian kernel
    #mode: BRIGHT, DARK, or BOTH
    #def frst(img, radii, alpha, beta, stdFactor, mode='BOTH')
    r=12
    rtran=np.abs(frst(img, r, 0, 0.01,2))
    #returns image with size input+2*rad
    maxp=np.unravel_index(np.argmax(rtran), rtran.shape)
    if draw and maxp[0]>r and maxp[1]>r and maxp[0]<r+img.shape[0] and maxp[1]<r+img.shape[1]:
        cv.circle(image, (maxp[1]-r,maxp[0]-r), 2, 0, 3)
    return [maxp[1]-r,maxp[0]-r]
    
def testGaussFit():
    img = cv.imread("testgaussel.jpg")
    data = grayData(img,np.array([(0,0),(340,500)]))
    res=gaussCentre(data)
    cv.imshow('image',data)
    cv.waitKey()
    cv.destroyAllWindows()
    return res

# get I(r) data, binned through binsize
def intensitiesRadius(image, x0, y0, binsize, maxr=-1 ,raw=False, scale=1.0):
    if(len(image.shape)==3):
        img=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        img=image
    img=img*scale
    #get distances and values
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    poss=np.array(list(zip(np.reshape(xx,(-1)),
                           np.reshape(yy,(-1)))))
    ys = np.reshape(img,(-1))
    diffs=poss-[y0,x0]
    rs=np.sqrt(np.sum(diffs**2,axis=1))
    data=[]

    cdata=[]
    step=binsize
    rawd={}

    if maxr == -1:
        maxr=min(image.shape[0]/2,image.shape[1]/2)

    while step<maxr:
        cdata=ys[np.logical_and(rs>=step-binsize,rs<step)]
        rawd[step]=cdata
        std=np.std(cdata)
        n=cdata.shape[0]
        if(n>0):
            data.append([step-binsize/2, np.average(cdata), std, std/np.sqrt(n-0.999999), n])
        else:
            data.append([step-binsize/2, np.nan, np.nan, np.nan, 0])
        step=step+binsize
    if raw:
        return rawd, np.array(data)
    return np.array(data)

def testGaussIntensitiesRadius():
    img = cv.imread("testgaussx.jpg")
    data = grayData(img,np.array([(0,0),(340,500)]))
    res=gaussCentre(data)
    y0=res[3]
    x0=res[2]
    cv.imshow('image',data)
    cv.waitKey()
    cv.destroyAllWindows()
    data=intensitiesRadius(img,x0,y0,2)
    plt.scatter(data[:,0],data[:,1],marker='.')
    plt.errorbar(data[:,0],data[:,1],yerr=data[:,2],linestyle="None")
    plt.show()
    return data

#cut out a rectagle containing the circle (square if inside image)
#behaviour undefined for large r edge cases
def cutByCircle(image, xc, yc, r, return_img=True):
    if xc-r >= 0:
        x0=xc-r
    else:
        x0=0

    if xc+r <=image.shape[1]:
        x1=xc+r
    else:
        x1=image.shape[1]

    if yc-r >= 0:
        y0=yc-r
    else:
        y0=0

    if yc+r<=image.shape[0]:
        y1=yc+r
    else:
        y1=image.shape[0]

    img=cutRect(np.array(
                    [(y0,x0),
                     (y1,x1)]),image)
    if return_img:
        return {'xc':xc-x0,
                'yc':yc-y0,
                'img':img,
                'rect':np.array(
                        [(y0,x0),
                         (y1,x1)])}
    else:
        return {'xc':xc-x0,
                'yc':yc-y0,
                'rect':np.array(
                        [(y0,x0),
                         (y1,x1)])}



#centres=gauss|hough
#ebars=std|err for standard deviation, std/sqrt(n-1)
def plotAllIntensitiesRadius(img, centres="gauss", average=False, binsize=1,
                             ebars='std', draw=False):
    if(ebars=='std'):
        eb=2
    else:
        eb=3
    if len(img.shape)==3:
        img=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    circles=houghCentres(img, False)
    rawds=[]
    lines=[]
    for circle in circles:
        invalid=False
        cutout=cutByCircle(img,circle[0],circle[1],5*circle[2])
        #The nikon image has a lot of weak ghost images
        #this should get rid of them
        rect=cutout['img']
        testr=cutByCircle(img, circle[0],circle[1],3+circle[2])['img']
        if testr.max()-testr.min() < 35:
            print("Ignoring due to small intensity spread")
            continue
        xc=cutout['xc']
        yc=cutout['yc']
        #find the radius if using gaussian
        if centres=="gauss":
            print("gaussfitting")
            res=gaussCentre(rect)
            centre=[res[2],res[3]]
            if res[2]-xc>50 or res[3]-yc>50:
                print("WARN: large hough-gauss difference")
                invalid=True
        else:
            centre=[xc,yc]
        #now we get the I(r) data, raw data used if we want to average
        irraw, irdata = intensitiesRadius(rect, centre[0], centre[1], binsize,
                                      5*circle[2], True)
        #need to filter the data, we can use the standard deviation to check if
        #its well detected (only really checks if its cylindrically sym.)
        error_score = irdata[0:int(2*circle[2]),2].mean()+0.2*irdata[int(2*circle[2]):,2].mean() #further away has less weight in the score
        if error_score > 15:
            print(f"Large error detected ({error_score}), ignoring circle")
            continue

        if draw:
            #draw the outer circle
            cv.circle(img,(circle[0],circle[1]),int(circle[2]),255,2)
            #draw the center of the circle
            cv.circle(img,(circle[0],circle[1]),2,0,3)

        rawds.append(irraw)
        lines.append(irdata)

    if not average:
        marker = itertools.cycle((',', '+', '.', 'o', '*'))
        for data in lines:
            plt.scatter(data[:,0],data[:,1],marker=next(marker))
            plt.errorbar(data[:,0],data[:,1],yerr=data[:,eb],linestyle="None")
        plt.show()
        data=lines
    else:
        nonempty=True
        data=[]
        cdata=[]
        step=binsize
        while nonempty:
            nonempty=False
            for raw in rawds:
                if step in raw:
                    cdata=np.append(cdata,raw[step])
                    nonempty=True
            cdata=np.array(cdata)
            std=np.std(cdata)
            n=cdata.shape[0]
            if(n>0):
                data.append([step-binsize/2, np.average(cdata), std, std/np.sqrt(n-0.999999), n])
            step=step+binsize
        data=np.array(data)
        plt.scatter(data[:,0],data[:,1],marker='.')
        plt.errorbar(data[:,0],data[:,1],yerr=data[:,eb],linestyle="None")
        plt.show()
    return data

def drawLocations(image, locs):
    print(f"Drawing {len(locs)} locations")
    for l in locs:
        z = list(l.keys())[0]
        xc=l[z]['rect'][0][1]+l[z]['centre'][0]
        yc=l[z]['rect'][0][0]+l[z]['centre'][1]
        print(f"xc:{xc}; yc:{yc}")
        cv.circle(image,(int(xc),int(yc)),20,255,2)
        #draw the center of the circle
        cv.circle(image,(int(xc),int(yc)),2,0,3)


ofm_zstacks=[
        "/media/filip/391eb46e-99ae-4ce0-a26f-d29d3f783595/project/ofm/beads/beads_zstack6/",
        #"/media/filip/391eb46e-99ae-4ce0-a26f-d29d3f783595/project/ofm/beads/beads_zstack5/",
        "/media/filip/391eb46e-99ae-4ce0-a26f-d29d3f783595/project/ofm/beads/beads_zstack4/",
        "/media/filip/391eb46e-99ae-4ce0-a26f-d29d3f783595/project/ofm/beads/beads_zstack3/",
        "/media/filip/391eb46e-99ae-4ce0-a26f-d29d3f783595/project/ofm/beads/beads_zstack7/",
        "/media/filip/391eb46e-99ae-4ce0-a26f-d29d3f783595/project/ofm/beads/beads_zstack8/"
        ]

def getImageShiftEstimate(im1, im2):
    shift=cv.estimateRigidTransform(im1,im2,False)
    if shift is None:
        print("Failed to find transform")
        return 0,0
    #get rotation angle to check, this should be very low
    #rot=np.arctan(-1.0*shift[0][1]/shift[0][0])
    #print("{} {} {} {}".format(i,shift[0][2],shift[1][2], rot*180.0/np.pi))
    return shift[0][2],shift[1][2]

def procZstackOFM(zstack, shift=0.0, debug=False, save_img=True):
    """
    Uses a distance-transform based system for finding the beads
    Similar to nikon processing but some constants different
    and we need to recentre at every point as it shifts slightly
    intensities normalised by the mean of entire z-stack
    """
    binsize=2
    files=glob(zstack+"*.jpg")
    pattern=re.compile(".*z([-0-9.]+).jpg")
    #first we get the locations from the first image
    files.sort()
    if len(files)==0:
        print("No files found!")
        return
    img=cv.imread(files[len(files)//2]).mean(axis=2).astype('uint8')
    im0=img
    circles=distCentres(img, draw=False)
    #houghCentres(img, True, blur=2,  min_radius=10, max_radius=32, min_distance=15,  threshold=True, houghparam1=18, thresh_val=110)#32)
    locations=[]
    testrs=[]
    centres="dist"
    #find min z
    minz=9999999999
    for f in files:
        z=float(pattern.match(f).group(1))
        if z < minz:
            minz=z

    shift-=minz
    for circle in circles:
        maxr=80 #5*circle[2]
        cutout=cutByCircle(img,circle[0],circle[1],maxr, True) #was 5*circle[2]
        testr=cutout
        #testr=cutByCircle(img, circle[0],circle[1],25+circle[2])
        if testr['img'].max()-testr['img'].min() < 15 and testr['img'].std() < 40:
            print("Ignoring due to small intensity spread")
            continue
        #find the radius if using gaussian
        if centres=="gauss":
            #print("gaussfitting")
            try:
                res=gaussCentre(testr['img'], False, testr['xc'],testr['yc'])
                #centre=[res[2],res[3]]
                if (res[2]-testr['xc'])**2+(res[3]-testr['yc'])**2>3200:
                    print(f"WARN: large hough-gauss difference {res[2]-testr['xc']},{res[3]-testr['yc']}")
                    continue
                else:
                    print(f"Correcting centre by {res[2]-testr['xc']},{res[3]-testr['yc']}")
                    cutout['xc']=res[2]#-testr['xc']
                    cutout['yc']=res[3]#-testr['yc']
                    #testr['xc']=res[2]
                    #testr['yc']=res[3]
            except RuntimeError:
                print("WARN: failed to fit gaussian.")
                continue
        irraw, irdata = intensitiesRadius(cutout['img'], cutout['xc'], cutout['yc'], 2,
                                      maxr, True)
        #need to filter the data, we can use the standard deviation to check if
        #its well detected (only really checks if its cylindrically sym.)
        error_score = np.nanmean(irdata[0:int(maxr//2)//2,2])+np.nanmean(0.2*irdata[int(maxr//2)//2:,2]) #further away has less weight in the score
        ivar = np.nanstd(irdata[0:2*int(maxr//2)//3,1])
        if error_score > 10 or error_score > ivar:
            print(f"Large error detected ({error_score}), ignoring circle!")
            continue
        elif ivar<9.5:
            print(f"Low variation detected ({ivar}), ignoring circle!")
            continue
        else:
            print(f"Continuing with var: {ivar:.2f}, err: {error_score:.2f}")
        locations.append(cutout)
        testrs.append(testr)
    print(f"Obtained {len(locations)} locations.")
    if debug:
        for l in locations:
            xc=l['rect'][0][1]+l['xc']
            yc=l['rect'][0][0]+l['yc']
            print(f"xc:{xc}; yc:{yc}")
            cv.circle(im0,(int(xc),int(yc)),20,255,2)
            #draw the center of the circle
            cv.circle(im0,(int(xc),int(yc)),2,0,3)
        cv.imshow('locations', cv.resize(im0,(800,600)))
        cv.waitKey(0)
        cv.destroyAllWindows()

    #this will process the first image twice but whatever...
    location_data=[]
    for i in range(len(locations)):
        location_data.append({})

    #we read all data twice but whatever
    savg=0.0
    for f in files:
        img = cv.imread(f)
        savg+=img.mean()
    savg=savg/len(files)
    print(f"Obtained average {savg}")
    prevcentres={}
    q=-1
    startim=None
    lastim=None
    for f in files[len(files)//2:]+list(reversed(files[0:len(files)//2])):
        #if not pattern.match(f):
        #    print(f)
        #    return location_data
        z=float(pattern.match(f).group(1))
        img=cv.imread(f)
        if startim is None:
            lastim = img
            startim = img
        i=0
        j=-1
        q+=1
        if q == len(files)//2:
            prevcentres={}
            lastim = startim
        shiftimg=getImageShiftEstimate(lastim, img)
        print(f"Detected shift: ({shiftimg[0]}, {shiftimg[1]})")
        for l,testr in zip(locations,testrs):
            j+=1
            xc=l['xc']
            yc=l['yc']
            cutout=cutRect(l['rect'], img)
            try:
                if j in prevcentres:
                    pc=prevcentres[j]
                else:
                    pc=np.array([l['xc'],l['yc']])
                #apply the computed shift
                pc[0]+=shiftimg[0]
                pc[1]+=shiftimg[1]
                if centres=="gauss":
                    #ress=[]
                    #xyshift=10
                    #ress.append(np.array(gaussCentre(cutout.mean(axis=2).astype('uint8'), False, pc[0],pc[1])))
                    #ress.append(ressshiftimg[0]) #twice the weight
                    #ress.append(np.array(gaussCentre(cutout.mean(axis=2).astype('uint8'), False, pc[0]+xyshift,pc[1]-xyshift)))
                    #ress.append(np.array(gaussCentre(cutout.mean(axis=2).astype('uint8'), False, pc[0]+xyshift,pc[1]+xyshift)))
                    #ress.append(np.array(gaussCentre(cutout.mean(axis=2).astype('uint8'), False, pc[0],pc[1]+xyshift)))
                    #ress.append(np.array(gaussCentre(cutout.mean(axis=2).astype('uint8'), False, pc[0],pc[1]-xyshift)))
                    #ress.append(np.array(gaussCentre(cutout.mean(axis=2).astype('uint8'), False, pc[0]+xyshift,pc[1])))
                    #ress.append(np.array(gaussCentre(cutout.mean(axis=2).astype('uint8'), False, pc[0]-xyshift,pc[1])))
                    #ress.append(np.array(gaussCentre(cutout.mean(axis=2).astype('uint8'), False, pc[0]-xyshift,pc[1]-xyshift)))
                    #ress.append(np.array(gaussCentre(cutout.mean(axis=2).astype('uint8'), False, pc[0]-xyshift,pc[1]+xyshift)))
                    #res=np.array(ress)
                    #cx = res[:,2].mean()
                    #cy = res[:,3].mean()
                    res = np.array(gaussCentre(cutout.mean(axis=2).astype('uint8'), False, pc[0], pc[1]))
                elif centres=="dist":
                    res_p1=np.array(distCentres(cutout.mean(axis=2).astype('uint8'), False, 1, 1))
                    res_n1=np.array(distCentres(cutout.mean(axis=2).astype('uint8'), False, 1, -1))

                    if len(res_p1) == 0 and len(res_n1) == 0:
                        print("Could not find bead, skipping position")
                        continue
                    else:
                        if len(res_p1) > 0:
                            res1=res_p1[np.argmin(((res_p1[:,0:2]-pc)**2).sum(axis=1),axis=0)]
                            irdata1 = intensitiesRadius(cutout, res1[0], res1[1], 2, 40, False)
                        if len(res_n1) > 0:
                            res2=res_n1[np.argmin(((res_n1[:,0:2]-pc)**2).sum(axis=1),axis=0)]
                            irdata2 = intensitiesRadius(cutout, res2[0], res2[1], 2, 40, False)
                        
                        if len(res_n1) == 0 or irdata1[:,2].mean() >  irdata2[:,2].mean():
                            res = gaussCentre(cutout.mean(axis=2).astype('uint8'), False, res2[0], res2[1], direction=-1)
                        else:
                            res = gaussCentre(cutout.mean(axis=2).astype('uint8'), False, res1[0], res1[1], direction=1)
                        #select one with minimal distance to previous
                        #[0,0,res[0], res[1]]
                        #res = np.array(gaussCentre(cutout.mean(axis=2).astype('uint8'), False, res[0], res[1]))
                elif centres=="frst":
                    #combined frst+gauss
                    frstC=frstCentre(cutout, draw=False)
                    if (frstC[0]-pc[0])**2+(frstC[1]-pc[1])**2 > 625:
                        print("Large frst shift detected, ignoring frst results")
                        frstC=pc
                    res=np.array(gaussCentre(cutout.mean(axis=2).astype('uint8'), False, frstC[0], frstC[1]))
                else:
                    res=[0,0,pc[0]+shiftimg[0],pc[1]+shiftimg[1],0]
                #centre=[res[2],res[3]]
                if (res[2]-pc[0])**2+(res[3]-pc[1])**2>3000:  #the images are massively shifted
                    print(f"WARN: large centre difference {res[2]-pc[0]},{res[3]-pc[1]}")
                    continue
                else:
                    print(f"Correcting centre by {res[2]-pc[0]},{res[3]-pc[1]}")
                    xc=res[2]#cutout['xc']+res[2]-testr['xc']
                    yc=res[3]#cutout['yc']+res[3]-testr['yc']
                    prevcentres[j]=[res[2],res[3]]
                    #testr['xc']=res[2]
                    #testr['yc']=res[3]
            except RuntimeError:
                print("WARN: failed to fit gaussian.")
                continue
            #obtain the centre again
            #now we get the I(r) data, raw data used if we want to average
            irraw, irdata = intensitiesRadius(cutout, xc, yc, binsize,
                                      maxr, True, scale=1.0/savg)
            if save_img:
                location_data[j][z+shift]={'raw':irraw, 'data':irdata, 'rect':l['rect'], 'centre':[xc,yc], 'img':cutout.astype('uint8'), 'scale':1.0/savg}
            else:
                location_data[j][z+shift]={'raw':irraw, 'data':irdata, 'rect':l['rect'], 'centre':[xc,yc]}
            i+=1
    #now remove invalid beads
    filtered=[]
    for bead in location_data:
        dsetsl=[]
        if len(bead) < 150:
            print("Fewer than 150 z positions, dropping set")
            continue
        for z,d in bead.items():
            irdata=d['data']
            dsetsl.append(irdata)
        dsets=np.array(dsetsl)
        #
        error_score = np.nanmean(dsets[50:-50,0:maxr//2//binsize,2])+0.3*np.nanmean(dsets[50:-50,maxr//2//binsize:,2])
        zsvar=np.nanmean(np.nanstd(dsets[80:-80,:,1],axis=0))
        if error_score > 0.07:
            print(f"Rejecting sequence due to large error on set {error_score}.")
            continue
        if zsvar < 0.04:
            print(f"Rejecting sequence due to zstack variation {zsvar}")
            continue
        #print("Keeping set")
        filtered.append(bead)
    print(f"Have {len(filtered)} beads")
    location_data=None
    if debug:
        drawLocations(im0, filtered)
        cv.imshow('locations', cv.resize(im0,(800,600)))
        cv.waitKey(0)
        cv.destroyAllWindows()
    return filtered


   
def plotBead(bead,zstep=1):
    zs=list(bead.keys())
    i=-1
    for z in zs:
        i+=1
        if not i % zstep == 0:
            continue

        data=bead[z]['data']
        plt.figure()
        #plt.xlim(0,35)
        plt.xlabel("Distance from centre/pix")
        #plt.ylim(70,260)
        plt.ylabel("Intensity/A.U.")
        #plt.plot(ints[:,0],ints[:,1],lw=2)
        plt.scatter(data[:,0],data[:,1], marker='+')
        #plt.show()
        plt.errorbar(data[:,0],data[:,1],yerr=data[:,3],linestyle="None", capsize=2)
        plt.text(50,110,"z="+str(z-zs[0]),bbox={'facecolor':'black','alpha':0.5})

def procZstackNikonBeads(zstack, shift=0.0, debug=False):
    """
    provide shift parameter to offset returned z-heights z->z+shift
    """
    files=glob(zstack+"*_*.tiff")
    pattern=re.compile(".*_z?([-0-9.]+)_0+.tiff")
    #first we get the locations from the first image
    files.sort()
    if len(files)==0:
        print("No files found!")
        return
    img=cv.imread(files[0]).mean(axis=2).astype('uint8')
    im0=img
    circles=houghCentres(img, False, threshold=True)
    locations=[]
    centres="gauss"
    for circle in circles:
        maxr=35 #5*circle[2]
        cutout=cutByCircle(img,circle[0],circle[1],maxr, True) #was 5*circle[2]
        testr=cutByCircle(img, circle[0],circle[1],3+circle[2])
        if testr['img'].max()-testr['img'].min() < 15:
            print("Ignoring due to small intensity spread")
            continue
        #find the radius if using gaussian
        if centres=="gauss":
            #print("gaussfitting")
            try:
                res=gaussCentre(testr['img'], False, testr['xc'],testr['yc'])
                #centre=[res[2],res[3]]
                if np.abs(res[2]-testr['xc'])>4 or np.abs(res[3]-testr['yc'])>4:
                    print(f"WARN: large hough-gauss difference {res[2]-testr['xc']},{res[3]-testr['yc']}")
                    continue
                else:
                    print(f"Correcting centre by {res[2]-testr['xc']},{res[3]-testr['yc']}")
                    cutout['xc']+=res[2]-testr['xc']
                    cutout['yc']+=res[3]-testr['yc']
            except RuntimeError:
                print("WARN: failed to fit gaussian.")
                continue
        irraw, irdata = intensitiesRadius(cutout['img'], cutout['xc'], cutout['yc'], 2,
                                      maxr, True)
        #need to filter the data, we can use the standard deviation to check if
        #its well detected (only really checks if its cylindrically sym.)
        error_score = irdata[0:int(maxr//2),2].mean()+0.2*irdata[int(maxr//2):,2].mean() #further away has less weight in the score
        if error_score > 21:
            print(f"Large error detected {error_score}, ignoring circle!")
            continue
        locations.append(cutout)
    print(f"Obtained {len(locations)} locations.")
    #this will process the first image twice but whatever...
    location_data=[]
    for i in range(len(locations)):
        location_data.append({})

    for f in files:
        #if not pattern.match(f):
        #    print(f)
        #    return location_data
        z=float(pattern.match(f).group(1))
        img=cv.imread(f)
        i=0
        for l in locations:
            xc=l['xc']
            yc=l['yc']
            rect=cutRect(l['rect'], img)

            #now we get the I(r) data, raw data used if we want to average
            irraw, irdata = intensitiesRadius(rect, xc, yc, 1,
                                      maxr, True)
            location_data[i][z+shift]={'raw':irraw, 'data':irdata, 'rect':l['rect'], 'centre':[xc,yc],'img':rect}
            i+=1
    #now remove invalid beads
    filtered=[]
    for bead in location_data:
        dsets=[]
        for z,d in bead.items():
            irdata=d['data']
            dsets.append(irdata)
        dsets=np.array(dsets)
        #
        error_score = dsets[25:65,0:10,2].mean()+0.2*dsets[25:65,10:,2].mean()
        if error_score > 141:
            print(f"Droping sequence due to large error on set {error_score}.")
            continue

        zsvar=dsets[:,:,1].std(axis=0).mean()
        if zsvar < 3:
            print(f"Rejecting sequence due to zstack variation {zsvar}")
            continue
        #print("Keeping set")
        filtered.append(bead)
    print(f"Have {len(filtered)} beads")
    if debug:
        drawLocations(im0, filtered)
        cv.imshow('locations', cv.resize(im0,(800,600)))
        cv.waitKey(0)
        cv.destroyAllWindows()
    return filtered

def getDataStats(data):
    out=[]
    for r,d in data.items():
        out.append([r, d.mean(), d.std(), d.std()/np.sqrt(len(d)-0.9999), len(d)])

    return np.array(out)

def testZmeasurement(calibration, testdata, zposs=None, plt_name="plot.png"):
    """
    Produced z-measurement calibration graph
    test data: beads as produced by procZstackNikonBeads
    calibration, data as produced by plotAvgZstackNikonBeads
    """
    if zposs is None:
        zposs=list(range(len(testdata[0])))
        dif=1
    else:
        dif=np.abs(zposs[1]-zposs[0])
    predictions=[]
    #for every position, get z for every bead
    for z in zposs:
        pz=[]
        predictions.append(pz)
        for bead in testdata:
            pred_z,s = findZ(calibration, bead[z]['data'])
            pz.append(pred_z)
    predictions=np.array(predictions)*dif#+zposs[0]
    p_m=predictions.mean(axis=1)
    p_e=predictions.std(axis=1)/np.sqrt(predictions.shape[1]-1)
    p_s=predictions.std(axis=1)
    plt.figure()
    #plt.xlim(0,35)
    plt.xlabel("Z stack position/um")
    #plt.ylim(70,260)
    plt.ylabel("Predicted z position/um")
    #plt.plot(ints[:,0],ints[:,1],lw=2)
    plt.scatter(zposs,predictions.mean(axis=1), marker='+')
    #plt.show()
    plt.errorbar(zposs,predictions.mean(axis=1),yerr=predictions.std(axis=1)/np.sqrt(predictions.shape[1]-1),linestyle="None", capsize=2)
    plt.savefig(plt_name, dpi=350)
    print("#meas_z pred_z stdev stderr")
    for i in range(len(zposs)):
        print(f"{zposs[i]} {p_m[i]} {p_s[i]} {p_e[i]}")
    #plt.gcf().clear()
    #print(np.array(predictions))

def plotAvgZstackNikonBeads(data, name="f1_", no_plot=False):
    #TODO: should we normalize intensity separately?
    collected={}

    for z in np.linspace(-5,5,81):
        #collect data from all beads
        values = {}
        for bead in data:
            #due to shifts, not all sets will have all -5,5 range
            if not z in bead:
                continue
            for i,vs in bead[z]['raw'].items():
                #print(i)
                #print(values[i])procZstackNikonBeads(f"{prefix}{f}")
                #print(vs)
                if i in values:
                    values[i] = np.concatenate((vs,values[i]))
                else:
                    values[i] = vs
        collected[z]=values
    allstats=[]
    i=0
    for z in np.linspace(-5,5,81):
        data=collected[z] #ensure right order for output allstats
        stats=getDataStats(data)
        allstats.append(stats)
       # print(f"Processing z={z}")
        if not no_plot:
            plt.figure()
            plt.xlim(0,35)
            plt.xlabel("r/pix")
            plt.ylim(70,260)
            plt.ylabel("Mean intensity/a.u.")
            #plt.plot(ints[:,0],ints[:,1],lw=2)
            plt.scatter(stats[:,0],stats[:,1], marker='+')
            plt.text(25,90,"z="+str(z),bbox={'facecolor':'black','alpha':0.5})
            #plt.show()
            plt.errorbar(stats[:,0],stats[:,1],yerr=stats[:,3],linestyle="None", capsize=2)
            plt.savefig(f"nikon_bead_plots1/{name}{i:02d}_z{z:+.3f}.png", dpi=250)
            plt.gcf().clear()
        i+=1
    return allstats


def plotAvgOFMBeads(data, name="f1_", no_plot=False, plot_every=5):
    #TODO: should we normalize intensity separately?
    collected={}
    #beads will beshifted at this point
    allzs=list(data[0].keys())
    allzs.sort()
    zstart=-1
    for z in allzs:
        if zstart == -1:
            zstart=z
        #collect data from all beads
        values = {}
        for bead in data:
            if not z in bead:
                continue
            for i,vs in bead[z]['raw'].items():
                #print(i)
                #print(values[i])procZstackNikonBeads(f"{prefix}{f}")
                #print(vs)
                if i in values:
                    values[i] = np.concatenate((vs,values[i]))
                else:
                    values[i] = vs
        collected[z]=values
    allstats=[]
    i=0
    for z in allzs:
        data=collected[z] #ensure right order for output allstats
        stats=getDataStats(data)
        allstats.append(stats)
       # print(f"Processing z={z}")
        if not no_plot and i%plot_every ==0:
            plt.figure()
            plt.xlim(0,70)
            plt.xlabel("r/pix")
            plt.ylim(0,2)
            plt.ylabel("Mean intensity/a.u.")
            #plt.plot(ints[:,0],ints[:,1],lw=2)
            plt.scatter(stats[:,0],stats[:,1], marker='.')
            plt.text(50,0.8,"z="+str(z-zstart),bbox={'facecolor':'black','alpha':0.5})
            plt.errorbar(stats[:,0],stats[:,1],yerr=stats[:,3],linestyle="None", capsize=2)
            #plt.show()
            plt.savefig(f"{name}{i:02d}_z{z:+.3f}.png", dpi=250)
            plt.gcf().clear()
        i+=1
    return allstats

def plotDatasetIntensities(folder, rect):
    files=glob(folder+"*.jpg")
    eb=3
    images=[]
    p=re.compile(".*_z(\d+).jpg")
    z0=-1
    markers = itertools.cycle((',', '+', '.', 'o', '*'))
    for file in files:
        image=cv.imread(file)
        z=int(p.match(file).group(1))
       # if z0==-1:
       #    z0=z#-500 #make the numbers more readable
       # z-=z0
        data=grayData(image,rect)
        circles=houghCentres(data, False)
        if circles is None:
            print("No circle detected in "+file)
            continue
        circle=circles[0]
        ints=intensitiesRadius(data,circle[0],circle[1],1,maxr=100)
        ints=np.array(ints)
        plt.xlim(0,100)
        plt.xlabel("r/pix")
        plt.ylim(80,250)
        plt.ylabel("Mean intensity/a.u.")
        #plt.plot(ints[:,0],ints[:,1],lw=2)
        plt.scatter(ints[:,0],ints[:,1],marker='+')
        plt.text(70,100,"z="+str(z),bbox={'facecolor':'black','alpha':0.5})
        #plt.show()
        plt.errorbar(ints[:,0],ints[:,1],yerr=ints[:,eb],linestyle="None", capsize=2)
        print(z)
        plt.savefig("testfig"+str(z)+".png", dpi=400)
        plt.gcf().clear()

def curveDifference(c1, c2):
    """
    Get 2 different curves as array of [[r,mean, std, stderr, n],..]
    Must be using the same binsize, ignores errors
    """
    if c1.shape[0] > c2.shape[0]:
        return np.nanmean((c1[0:c2.shape[0],1]-c2[:,1])**2)
    elif c1.shape[0] < c2.shape[0]:
        return np.nanmean((c1[:,1]-c2[0:c1.shape[0],1])**2)
    else:
        return np.nanmean((c1[:,1]-c2[:,1])**2)

def zstackCurvesDifference(data1, data2, shift):
    """
    data in the shape (zsteps, rsteps, 5)
    shift in the number of steps within this array
    """
    data1=np.array(data1)
    data2=np.array(data2)
    if shift > 0:
        return ((data1[shift:,:,1]-data2[:-shift,:,1])**2).sum(axis=1).mean()
    elif shift<0: #swap sign and data1<->data2
        return ((data2[-shift:,:,1]-data1[:shift,:,1])**2).sum(axis=1).mean()
    return ((data2[:,:,1]-data1[:,:,1])**2).sum(axis=1).mean()

def findZ(calibration, data):
    """
        iterate through all possible shifts and get the best score
        TODO: can we interpolate???
    """
    best_shift=0
    best_score=99999999
    for shift in range(len(calibration)):
        score = curveDifference(calibration[shift], data)
        if score < best_score:
            best_shift=shift
            best_score=score
    return best_shift, best_score

def findBestShift(data1, data2, min_overlap=10):
    """
    test all
    """
    best_shift=0
    best_score=99999999
    #keep at least 10 overlapping
    for i in range(-len(data1)+min_overlap,len(data1)-min_overlap):
        sc=zstackCurvesDifference(data1,data2,i)
        if sc < best_score:
            best_shift=i
            best_score=sc
    print(f"Obtained shift {best_shift}, score {best_score}")
    return best_shift

def shiftBeads(beads, shift):
    """
    shift the z-values of the set by a given value
    """
    out=[]
    for bead in beads:
        beaddata={}
        for z,d in bead.items():
            beaddata[z+shift]=d
        out.append(beaddata)
    return out

prefix="/media/filip/391eb46e-99ae-4ce0-a26f-d29d3f783595/project/nikon/beads1/tiffs/beads1"

def processAllNikonSets():
    #get the first set separately and use it as reference for shifts
    beads=procZstackNikonBeads(f"{prefix}{files[0]}",0.0, False)
    data1=plotAvgZstackNikonBeads(beads,"",False) #do not plot just return data
    print(f"Obtained {len(beads)} beads")
    for f in files[1:]: #TODO all
        locs=procZstackNikonBeads(f"{prefix}{f}", 0.0, False)
        data2=plotAvgZstackNikonBeads(locs, "", True)
        print(f"Obtained {len(locs)} beads")
        if len(locs) == 0:
            continue
        shift=findBestShift(data1, data2)
        beads+=shiftBeads(locs, shift*0.125)
        #print(f"Processed {f}")
    all_data=plotAvgZstackNikonBeads(beads,'all')
    print(f"Obtained {len(beads)} beads in total")
    return all_data, beads

def presentationPlot():
    rect=np.array(
                    [(1452,800),
                     (1760,1232)])
    zs=[227311, 227381, 227491]
    src="/media/filip/data1/project_data/z3/image_0*_x21376_y103424_z"
    plt.xlim(0,100)
    plt.xlabel("r/pix")
    plt.ylim(90,250)
    plt.ylabel("Mean intensity/a.u.")
    eb=3
    for z in zs:
        file=glob(src+str(z)+".jpg")[0]
        image=cv.imread(file)
        data=grayData(image,rect)
        circles=houghCentres(data, True)
        if circles is None:
            print("No circle detected in "+str(z))
            continue
        circle=circles[0]
        ints=intensitiesRadius(data,circle[0],circle[1],1,maxr=100)
        ints=np.array(ints)

        #plt.plot(ints[:,0],ints[:,1],lw=2)
        plt.scatter(ints[:,0],ints[:,1],marker='+',label="z="+str(z-zs[0]))
        #plt.text(70,100,"z="+str(z),bbox={'facecolor':'black','alpha':0.5})
        #plt.show()
        plt.errorbar(ints[:,0],ints[:,1],yerr=ints[:,eb],linestyle="None", capsize=2)
        print(z)
    plt.legend()
    plt.savefig("Ir_presentation.png", dpi=300)
    #plt.gcf().clear()

def testNikonZMeasurement():
    beads = procZstackNikonBeads(f"{prefix}{files[5]}",0.0, False)
    cal=np.load('nikon_beads_hough.npy')
    testZmeasurement(cal, beads, np.linspace(-5,5,81), "nikon_z_measurement.png")


def alignBeadStacks(beads, cal=None):
    """
    Align beads z-stacks with calibration data, or use the first bead for calibration,
    return shifts
    """
    shifts=[]
    shifted=[]
    if cal is None:
        dsets=[]
        for z,d in beads[0].items():
            irdata=d['data']
            dsets.append(irdata)
        cal = np.array(dsets)

    for bead in beads:
        dsets=[]
        zvals=[]
        for z,d in bead.items():
            irdata=d['data']
            zvals.append(z)
            dsets.append(irdata)
        data=np.array(dsets)
        z=findBestShift(cal, dsets, 100)
        if np.abs(z) > 190:
            print(f"Large z shift:{z}, skipping")
            continue
        shifts.append(z*(zvals[1]-zvals[0]))
        shifted+=shiftBeads([bead], z*(zvals[1]-zvals[0]))
        print(z*(zvals[1]-zvals[0]))
    return shifts, shifted


def getOFMBeads(img, debug=False):
    """
    get list of beads from openflexure image, based on procOFMZstack
    """
    im0=img
    img=img.copy()
    circles=distCentres(img, draw=False)
    #houghCentres(img, True, blur=2,  min_radius=10, max_radius=32, min_distance=15,  threshold=True, houghparam1=18, thresh_val=110)#32)
    locations=[]
    centres="gauss"
    avg=img.mean()

    for circle in circles:
        maxr=80 #5*circle[2]
        cutout=cutByCircle(img,circle[0],circle[1],maxr+20, True) #was 5*circle[2]
        if cutout['img'].max()-cutout['img'].min() < 17 and cutout['img'].std() < 45:
            print("Ignoring due to small intensity spread")
            continue
        #find the radius if using gaussian
        if centres=="gauss":
            #print("gaussfitting")
            try:
                res=gaussCentre(cutout['img'], False, cutout['xc'],cutout['yc'])
                #centre=[res[2],res[3]]
                if (res[2]-cutout['xc'])**2+(res[3]-cutout['yc'])**2>100:
                    print(f"WARN: large hough-gauss difference {res[2]-cutout['xc']},{res[3]-cutout['yc']}")
                    continue
                else:
                    print(f"Correcting centre by {res[2]-cutout['xc']},{res[3]-cutout['yc']}")
                    cutout['xc']=res[2]
                    cutout['yc']=res[3]
            except RuntimeError:
                #print("WARN: failed to fit gaussian.")
                continue
        binsize=2
        irraw, irdata = intensitiesRadius(cutout['img']/avg, cutout['xc'], cutout['yc'], binsize,
                                      maxr, True)
        #need to filter the data, we can use the standard deviation to check if
        #its well detected (only really checks if its cylindrically sym.)
        error_score = irdata[0:int(maxr//2//binsize),2].mean()+0.2*irdata[int(maxr//2//binsize):,2].mean() #further away has less weight in the score
        if error_score > 0.055:
            print(f"Large error detected {error_score}, ignoring circle!")
            continue
        if irdata[0:(maxr-30)//binsize,1].std() < max(1.5*error_score, 0.07):
            print("Low standard deviation, dropping")
            continue
        print(f"Error score {error_score}, std: {irdata[0:maxr-30,1].std()}")
        locations.append({'raw':irraw, 'data':irdata, 'rect':cutout['rect'], 'centre':[cutout['xc'],cutout['yc']]})
    if debug:
        for l in locations:
            xc=l['rect'][0][1]+l['centre'][0]
            yc=l['rect'][0][0]+l['centre'][1]
            print(f"xc:{xc}; yc:{yc}")
            cv.circle(im0,(int(xc),int(yc)),20,255,2)
            #draw the center of the circle
            cv.circle(im0,(int(xc),int(yc)),2,0,3)
        cv.imshow('locations', cv.resize(im0,(800,600)))
        cv.waitKey(0)
        cv.destroyAllWindows()
    return locations

def getBeadsXYZ(files, cal):
    """
        Get xy, z (relative to centre) of all detected beads in the image
        Beads the average of a central area will be used as z=0
    """
    res=np.empty((0,4))
    #this center area is ~1/6
    center_h=2464//8
    center_w=3280//8

    cx1=3280//2-center_w
    cx2=3280//2+center_w
    cy1=2464//2-center_h
    cy2=2464//2+center_h
    raws=[]
    for f in files:
        unshifted=[]
        center_zs=[]
        img=cv.imread(f).mean(axis=2).astype('uint8')
        beads = getOFMBeads(img, False) #set debug here
        print(f"Extracted {len(beads)} beads.")
        for bead in beads:
            xc=bead['rect'][0][1]+bead['centre'][0]
            yc=bead['rect'][0][0]+bead['centre'][1]
            zc,score = findZ(cal, bead['data'])
            zc*=4 #convert to steps
            #check if it's in the centre
            if xc > cx1 and xc < cx2 and yc>cy1 and yc<cy2:
                center_zs.append(zc)
                print(f"Center bead at shift {zc}")
            unshifted.append([xc,yc,zc, score])
        raws.append(unshifted)
        if(len(center_zs)<1):
            print("No center beads found, skipping")
            continue
        center_zs=np.array(center_zs)
        center_z=center_zs.mean()
        center_std=center_zs.std()
        if center_zs.shape[0] > 1:
            print(f"Center std: {center_std}")
        shifted=np.array(unshifted)-[0,0,center_z, 0]
        res=np.concatenate((res,shifted))
    return res, raws

def saveBeads(beads, folder, drawc=True):
    j=0
    for bead in beads:
        i=0
        os.mkdir(f"{folder}/{j}/")
        for z,p in bead.items():
            img=p['img'].copy()#                                      3     2
            cv.circle(img, (int(p['centre'][0]),int(p['centre'][1])), 1, 0, 1)
            cv.imwrite(f'{folder}/{j}/{z}.png',img)
            i+=1
        j+=1

def getPlane(data, nooffset=False):
    """
    """
    def err(a, data):
        if nooffset:
            return ((a[0]*data[:,0]+a[1]*data[:,1]-data[:,2])**2).sum()/400000
        return ((a[0]*data[:,0]+a[1]*data[:,1]+a[2]-data[:,2])**2).sum()/400000
    if nooffset:
        a0 = [1e-2,1e-2]
    else:
        a0 = [1,1,1]
    res = minimize(err, a0, args = data, options={'disp':True, 'maxiter': 2000})
    resd=data.copy()
    a=res.x
    resd[:,2]-=a[0]*data[:,0]+a[1]*data[:,1]
    #return res.x, resd
    return res.x

def getPlanePlus(data):
    """
    fit a plane + a*r
    """
    def err(a, data):
            return ((a[0]*data[:,0]+a[1]*data[:,1]+a[2]*np.sqrt(data[:,1]**2+data[:,0]**2)-data[:,2])**2).sum()/2e6
    a0 = [1e-2,1e-2,1e-2]
    res = minimize(err, a0, args = data, options={'disp':True, 'maxiter': 2000})
    #get residuals
    resd=data.copy()
    a=res.x
    resd[:,2]-=a[0]*data[:,0]+a[1]*data[:,1]+a[2]*np.sqrt(data[:,1]**2+data[:,0]**2)
    return res.x, resd

def getParab(data):
    def err(a, data):
            return ((a[0]*(data[:,0]**2+data[:,1]**2)-data[:,2])**2).sum()/5e6
    a0 = [1e-2]
    res = minimize(err, a0, args = data, options={'disp':True, 'maxiter': 2000})
    #get residuals
    resd=data.copy()
    a=res.x
    resd[:,2]-=a[0]*(data[:,0]**2+data[:,1]**2)
    return res.x, resd

def fitWithOffsets(data, centre=[1640,1232]):
    filtered=[]
    for d in data:
        #the z-measurement only works at ~300-1100 positions, discard others
        ftd=[]
        for bead in d:
            if bead[2]<80 or bead[2]>800:
                print("removing bead")
                continue
            ftd.append(np.array(bead))
        if len(ftd) < 3:
            continue
        dat=np.array(ftd)
        dat[:,0]-=centre[0]
        dat[:,1]-=centre[1]
        filtered.append(dat)
    data=filtered
    def err(a, data):
        total=0
        for i in range(len(data)):
            d=data[i]
            total+=((a[2]*(d[:,0]**2+d[:,1]**2)+
                     d[:,0]*a[0]+d[:,1]*a[1]+a[3+i]-d[:,2])**2).sum()/5e6
        return total
    #plane+parabolic
    a0 = [0,0,1e-2]
    #add the offsets
    for i in range(len(data)):
        a0.append(np.random.random())
    res = minimize(err, a0, args = data, options={'disp':True, 'maxiter': 2000})
    #get residuals on the shifted set
    resd=[]
    a=res.x
    shifted=[]
    for i in range(len(data)):
        d=data[i].copy()
        d[:,2]-=a[3+i]
        shifted.append(d.copy())
        d[:,2]-=a[2]*(data[i][:,0]**2+data[i][:,1]**2)+data[i][:,0]*a[0]+data[i][:,1]*a[1]
        resd+=list(d)
    return res.x, np.array(resd), np.array(shifted)

def findZMultiple(cals, data):
    zs=[]
    errs=[]
    for cal in cals:
        z, err = findZ(cal, data)
        zs.append(z)
        errs.append(err)
    zs=np.array(zs)
    errs=np.array(errs)
    #TODO: weight based on err?
    return zs.mean(), zs.std()

def getHeightR(data):
    center=np.array([3280.0/2, 2464.0/2, 0,0])
    rel = data-center
    res = np.concatenate((np.expand_dims(np.sqrt(rel[:,0]**2+rel[:,1]**2), axis=1),
                          np.expand_dims(rel[:,2], axis=1)), axis=1)
    plt.figure()
    plt.ylim(-5,300)
    plt.scatter(res[:,0],res[:,1])
    return res


def plot3D(res, fname="notiltmap1.dat"):
    bins=10
    means = stats.binned_statistic_2d(res[:,0],res[:,1],res[:,2],statistic='mean', bins=bins)
    devs  = stats.binned_statistic_2d(res[:,0],res[:,1],res[:,2],statistic='std', bins=bins)
    devs = devs.statistic
    yc=(means.y_edge[1:]+means.y_edge[:-1])/2.0
    xc=(means.x_edge[1:]+means.x_edge[:-1])/2.0
    mesh=np.array(np.meshgrid(xc,yc))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-300,300)
    ax.scatter(np.reshape(mesh[0,:,:]/3280.0,(-1)), np.reshape(mesh[1,:,:]/2464.0,(-1)), np.reshape(means.statistic,(-1)), marker='+')
    axc=np.expand_dims(np.reshape(mesh[0,:,:]/3280.0,(-1)),axis=1)
    ayc=np.expand_dims(np.reshape(mesh[1,:,:]/2464.0,(-1)),axis=1)
    azc=np.expand_dims(np.reshape(means.statistic,(-1)),axis=1)
    dvs=np.expand_dims(np.reshape(devs,(-1)),axis=1)
    data=np.concatenate((axc,ayc,azc,dvs),axis=1)
    np.savetxt(fname, data)
    return data

def prepForManualAlign(i=0, save_img=True):
    beads = []
    total=0
    f=ofm_zstacks[i]
    res=procZstackOFM(f, shift=0.0, debug=False, save_img=save_img)
    #s, b = alignBeadStacks(res, cal)
    #shifts+=r
    #beads += res
    os.mkdir(f"ofm_beads_unaligned/{i}")
    saveBeads(res, f"ofm_beads_unaligned/{i}")
    np.save(f"ofm_beads_unaligned{i}.npy", res)
    #i+=1
    total+=len(res)
    print(f"Have {total} in total")
    #np.save("ofm_beads_unaligned.npy", beads)
    return total
cx, cy = -1, -1
def mouseCB(event, x, y,flags, param):
    global cx, cy
    if event == cv.EVENT_LBUTTONUP:
        cx,cy=x,y

def prepManual():
    cv.namedWindow('image')
    cv.setMouseCallback('image',mouseCB)

def getManualCentre(image):
    global cx, cy
    cx = -1
    cy = -1
    #already have the window ready
    cv.imshow('image', image)
    while cx == -1 and cy == -1:
        key = cv.waitKey(20)
        if key == ord('s'):
            return [-1,-1]
    return [cx, cy]

def manualCentreCorrect(bead, correct_from, correct_to):
    for z, b in bead.items():
        if z >= correct_from and z <= correct_to:
            img=b['img'].copy()
            cv.circle(img, (int(b['centre'][0]),int(b['centre'][1])), 3, 0, 2)
            cc = getManualCentre(img)
            if cc[0] > 0 and cc[1] > 0:
                b['centre']=cc
                olderr=b['data'][:,2].mean()
                irraw, irdata = intensitiesRadius(b['img'], cc[0], cc[1], 2,
                                      80, True, scale=b['scale'])
                b['raw']=irraw
                b['data']=irdata
                newerr=b['data'][:,2].mean()
                print(f"Corrected centre to {cc[0]},{cc[1]}, error decreased by {olderr-newerr}")
    return bead

def procManualAligned():
    #only first 1-2 beads in the datasets seem usable
    bead_shifts={
        '0':[548.0,724.0,708.0,-1,-1,-1,632],
        '1': [-1,580,-1,-1,572, -696.0, -1, 572, -1, -1, 536,500, -1], #second needs more corrections up to 100+
        '2': [-1, -1, 772, 768,748,744,-1,720, 708],
        '3': [712, -1, 736, 700, 728, 712,  680, 712, 700, 708, 652]
        }
    #load and align beads
    shifted=[]
    prepManual()
    for i, shifts in bead_shifts.items():
        beads = np.load(f"ofm_beads_unaligned{i}.npy")
        for j, shift in enumerate(shifts):
            if shift < 0:
                continue
            corrected=manualCentreCorrect(beads[j], shift-16, shift+16)
            shifted+=shiftBeads([corrected], 732-shift)

    print(f"Have {len(shifted)} beads")
    cal = plotAvgOFMBeads(beads, name="ofm_plots/ofm", no_plot=False, plot_every=1)
    np.save('ofm_beads_cal.npy', cal)
    np.save('ofm_beads_aligned.npy', shifted)
    return shifted

def testOFMZ():
    cal=np.load('ofm_beads_cal.npy')
    #print(cal.shape)
    testbeads=np.load("ofm_beads_unaligned3.npy")
    testZmeasurement(cal, [testbeads[2]], zposs=list(testbeads[2].keys()), plt_name="ofm_test_z2.png")
    print("\n")
    testZmeasurement(cal, [testbeads[0]], zposs=list(testbeads[0].keys()), plt_name="ofm_test_z0.png")
    print("\n")
    testZmeasurement(cal, [testbeads[3]], zposs=list(testbeads[3].keys()), plt_name="ofm_test_z3.png")
    print("\n")
    testZmeasurement(cal, [testbeads[10]], zposs=list(testbeads[10].keys()), plt_name="ofm_test_z10.png")


def getFieldMap():
    cal=np.load('ofm_beads_cal.npy')
    fs=glob("/media/filip/391eb46e-99ae-4ce0-a26f-d29d3f783595/project/ofm/beads/beads/*.jpg")
    #res,raws=getBeadsXYZ(fs,cal)
    #np.savetxt("rawbeadxyz.npy", res)
    res=np.loadtxt('rawbeadxyz.npy')
    #np.save("fieldmap_beads.npy", np.array(raws))
    raws=np.load('fieldmap_beads.npy')
    plot3D(res, "fieldmap1.dat")
    #remove tilt
    plane = getPlane(res)
    print("Obtained plane:")
    print(plane)
    zs = res[:,0]*plane[0]+res[:,1]*plane[1]+plane[2]
    notilt=res.copy()
    notilt[:,2] = res[:,2]-zs
    plot3D(notilt, "notiltmap1.dat")
    return getHeightR(notilt), raws, res
#getFieldMap()
#def recalculateIFMBeads(beads):
#    out=[]
#    for bead in beads:

#testOFMZ()
#beads=prepForManualAlign(5)

cv.setNumThreads(3)
#beads=procZstackNikonBeads("/media/filip/391eb46e-99ae-4ce0-a26f-d29d3f783595/project/nikon/beads1/tiffs/beads1x0.000_y0.000")


#res=distCentres(testim.mean(axis=2).astype('uint8'), False, 1,None)
#beads=prepForManualAlign()
#
#for i in range(5):
#    beads=np.load(f"ofm_beads_unaligned{i}.npy")
#    os.mkdir(f"ofm_beads_unaligned/{i}")
#    saveBeads(beads, f"ofm_beads_unaligned/{i}/")
#    beads=None
#     prepForManualAlign(i, save_img=True)
#     gc.collect()

#procManualAligned()
#testOFMZ()
#getFieldMap()


#img=cv.imread('/media/filip/391eb46e-99ae-4ce0-a26f-d29d3f783595/project/nikon/beads1/tiffs/beads1x0.000_y1000.000_0.000000_000000.tiff').mean(axis=2).astype('uint8')
#cv.imshow('orig',cv.resize(img.copy(),(800,600)))
#data=plotAllIntensitiesRadius(img, 'hough',average=True, draw=True, ebars='err')
#cv.imshow('img',cv.resize(img,(800,600)))
#locs=procZstackNikonBeads("/media/filip/391eb46e-99ae-4ce0-a26f-d29d3f783595/project/nikon/beads1/tiffs/beads1x0.000_y0.000")
#col=plotAvgZstackNikonBeads(locs)
#data,beads=processAllNikonSets()
#np.save('nikon_beads_hough.npy', data)-
#testNikonZMeasurement()
#lotBead(res[0],20)
#cal = None
#use one of the beads in the first image for calibration
#testbeads=procZstackOFM(ofm_zstacks[0], shift=0.0, debug=False)
#print(f"Len(res): {len(res)}")
#pl=res[5] #has the minimum error
#cal=plotAvgOFMBeads([pl], no_plot=True)
#saveBeads(beads, "ofm_beads_unaligned")

#ofm_zstacks=[]


#print("Plotting")
#with open('ofm_beads.json', 'w') as jsonf:
#    json.dump(beads, jsonf)
#cal = plotAvgOFMBeads(beads, name="ofm_plots/ofm", no_plot=False, plot_every=1)
#np.save('ofm_beads_cal2.npy', cal)
#cal=np.load('ofm_beads.npy')
#fs=glob("/media/filip/391eb46e-99ae-4ce0-a26f-d29d3f783595/project/ofm/beads/beads/*.jpg")
#testZmeasurement(cal, [testbeads[5]], zposs=list(testbeads[0].keys()), plt_name="ofm_test_z.png")
#res=getBeadsXYZ(fs,cal)
#np.savetxt("rawbeadxyz.npy", res)
#2464//8


#plot3D(res, "fieldmap1.dat")


#    center_w=3280//8

#remove tilt
#notilt=res.copy()
#notilt[:,2] = res[:,2]-zs
#plot3D(notilt, "notiltmap1.dat")
#getHeightR(notilt)
#, c=c, marker=m)

#avgs=plotAvgOFMBeads(beads, name="ofm_plots/ofm", no_plotzs = res[:,0]*plane[0]+res[:,1]*plane[1]+plane[2]=False, plot_every=1)
#cv.waitKey()
#cv.destroyAllWindows()
#presentationPlot()
#plotDatasetIntensities("/media/filip/data1/project_data/z3/",np.array(
 #                   [(1452,800),
 #                    (1760,1232)]))
#res=testGaussFit()
#data=testGaussIntensitiesRadius()
#img = cv.imread("test.jpg")
#plotAllIntensitiesRadius(img,"hough", False, 8,'err')
#houghCentres(img)
#img=cv.resize(img,None,fx=0.3,fy=0.3)
#cv.imshow('image',img)
#cv.waitKey()
#cv.destroyAllWindows()

#popt,pcov=fitGauss(data, False)
#res=houghCentres(data)

#procImage(img, np.array([(437,780,),(1628,970)]),5)