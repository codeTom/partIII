#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:57:42 2019

@author: filip
"""

import numpy as np
import cv2 as cv
import sys
import random
from matplotlib import pyplot as plt
import scipy.optimize as opt
import itertools
from glob import glob
import re
from scipy.misc import imread
from scipy import ndimage
from scipy.ndimage.measurements import variance
from ntpath import basename
import multiprocessing as mp
import time
import os

#segmentation
scalesize=1
MIN_SIZE=25**2*3.14
MIN_SIZE_FILTER=25**2*3.14
MAX_SIZE=46**2*3.14

def removeSmallRegions(image, min_size,connectivity=8):
    
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(
                                            image, connectivity=connectivity)
    sizes=stats[1:, -1]
    filtered=np.zeros((output.shape), dtype=np.uint8)
    for i in range(0, nb_components-1):
        if sizes[i]>min_size:
            filtered[output==i+1] = 255
    return filtered

def boundingRect(markers,val):
    """
        get a box containing all instances of val in markers
    """
    vals=np.where(markers==val)
    lc=(vals[0].min(),vals[1].min())
    rc=(vals[0].max(),vals[1].max())
    return np.array([lc,rc])


def cutRect(rect, image):
    """
        TODO: handle edges
    """
    return image[rect[0][0]:rect[1][0],rect[0][1]:rect[1][1]]

def centreImage(background, image):
    background[(background.shape[0]-image.shape[0])//2:(background.shape[0]+image.shape[0])//2,
               (background.shape[1]-image.shape[1])//2:(background.shape[1]+image.shape[1])//2] =image
    return background


def extractImage(background, image, markers, val, shape=(200,200),
                 allow_edge=True, name=""):
    rect=boundingRect(markers,val)
    rw=rect[1][0]-rect[0][0]
    rh=rect[1][1]-rect[0][1]
    #grow the rectangle to shape
    needw=(shape[0]-rw)/2
    needh=(shape[1]-rh)/2
    #check for image edges
    edge1=True
    if(rect[0][0]<needw):
        rect[0][0]=0
        rect[1][0]=shape[0]
    elif(rect[1][0]>image.shape[0]-needw):
        rect[1][0]=image.shape[0]-1
        rect[0][0]=image.shape[0]-shape[0]
    else:
        rect[0][0]-=needw
        rect[1][0]+=needw
        edge1=False
    edge2=True
    if(rect[0][1]<needh):
        rect[0][1]=0
        rect[1][1]=shape[1]
    elif(rect[1][1]>image.shape[1]-needw):
        rect[1][1]=image.shape[1]-1
        rect[0][1]=image.shape[1]-shape[1]
    else:
        rect[0][1]-=needh
        rect[1][1]+=needh
        edge2=False
    if (edge1 or edge2) and not allow_edge:
        return False
    
    #report ID->location
    if name:
        print(f"EXTRACTED,{name},{rect[0][0]},{rect[0][1]},{rect[1][0]},{rect[1][1]}")
    back=cutRect(rect,background)
    img=cutRect(rect,image)
    marks=cutRect(rect,markers)
    maskgs = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    maskgs[marks==val]=255
    maskgs=cv.bitwise_not(maskgs)
    #clear some black areas
    maskgs=removeSmallRegions(maskgs, 100)
    maskgs=cv.bitwise_not(maskgs)
    #TODO: Add option of keeping surroundings here
    #simply disable the mask multiplication
    mask1=np.zeros(img.shape, dtype=np.uint8)
    mask1[maskgs==255]=[1,1,1]
    back[marks==val]=[0,0,0]
    back=back+np.multiply(mask1,img)
    #hack to fix small size deviation due to rounding (111->112 at the edges)
    return cv.resize(back, shape)

def markCells(img, draw=False):
    cv.imwrite('segment_start.png', img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite('segment_grayscale.png', gray)
    ret, thresh = cv.threshold(gray, 0 ,255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    cv.imwrite('segment_thresh1.png', thresh)
    thresh=removeSmallRegions(thresh,MIN_SIZE_FILTER)
    thresh=cv.bitwise_not(thresh)
    thresh=removeSmallRegions(thresh,MIN_SIZE_FILTER)
    thresh=cv.bitwise_not(thresh)
    cv.imwrite('segment_removed_small.png', thresh)
    #cv.imshow('thresh', cv.resize(thresh,(1024,768)))
    dist=cv.distanceTransform(thresh, cv.DIST_L2, 3)
    #cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    #dist=dist*255
    cv.imwrite('segment_dist.png', (dist*5.0).astype('uint8'))
    dist=dist.astype('uint8')
    
    ret, dist = cv.threshold(dist, 22, 255, cv.THRESH_BINARY)
    cv.imwrite('segment_dist_thresh.png', dist)
   # cv.imshow('thresh', cv.resize(thresh,(1024,768)))
    dist_8u = dist.astype('uint8')
    #cv.imshow('dist', cv.resize(dist,(1024,768)))
    # Find total markers
    _, contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)
    # Draw the foreground markers
    for i in range(len(contours)):
        cv.drawContours(markers, contours, i, (i+1), -1)
    cv.imwrite('segment_markers_contours.png', markers)
    #sharp = cv.filter2D(img, -1, kernel)
    thresh3=np.zeros((thresh.shape[0],thresh.shape[1],3), dtype=int)
    thresh3[thresh==255]=np.array([1,1,1], dtype=int)
    res=img*thresh3
    cv.imwrite('segment_threshmultip.png', res)
    markers = cv.watershed(res.astype('uint8'),markers)
    if draw:
        img[markers == -1] = [255,0,0]
        #for i in circles[0,:]:
        #    if i[0] >= markers.shape[1] or i[1] >=markers.shape[0]:
        #        print("Location at the edge, skipping")
        #        continue
        #    cv.circle(img,(i[0],i[1]),i[2],(0,255,0),5)
        #    #draw the center of the circle
        #    cv.circle(img,(i[0],i[1]),2,(0,0,255),5)
    cv.imwrite('segment_img.png', img)
    cv.imwrite('segment_markers_watershed.png', markers)
    return markers

def getSize(markers,val):
    ze=np.zeros(markers.shape)
    ze[markers==val]=1
    return ze.sum()

def getCells(image, shape, allow_edge=False, name=""):
    background=np.zeros(image.shape)
    img=image.copy()
    print("Marking cells")
    markers=markCells(img, True)
    marker_ids=np.unique(markers)
    marker_ids=marker_ids[marker_ids>0]
    total=0
    cells=[]
    for mid in marker_ids:
        size=getSize(markers,mid)
        intensity=image[markers==mid].mean()
        if size>MAX_SIZE or size<MIN_SIZE:
            print("Marker {} rejected due to size: {}".format(mid,size))
            continue

        if intensity > 200:
            print("Marker {} rejected due to brightness: {}".format(mid,intensity))
            continue

        procim=extractImage(background,
                            image, markers, mid, shape, allow_edge, f"{name}_{total}")

        if type(procim) == type(False):
            print("Marker {} rejected (too edgy): {}".format(mid,size))
            continue
        print("Extracted cell: id: {} size: {} intensity: {}".format(mid,size,
                                  intensity))
        cells.append(procim)
        total+=1
        #cv.imshow('orig',procim.astype(np.uint8))
        #cv.waitKey(0)
    print("Extracted {} cells".format(total))
    return cells

def readZstack(fname):
    zstack=glob(fname+"_z*.tiff")
    data={}
    for file in zstack:
        ma = re.search('_z([-\d.]+).tiff', file)
        z=float(ma.group(1))
        data[z]=imread(file)
    return data

def sharpness(image):
    image_bw=np.mean(image,2)
    image_lap=ndimage.filters.laplace(image_bw)
    return np.mean(image_lap.astype(np.float)**4)

def zstackSharpness(zstack):
    sharps=[]
    zs=[]
    for z,img in zstack.items():
        zs.append(z)
        sharps.append(sharpness(img))
    return zs,sharps

def labelCell(cell):
    """
    Show the cell and wait for key 1,2,3 for negative, possible, positive
    """
    cv.imshow('img',cv.resize(cell.astype(np.uint8),(3*112,3*112)))
    resp=cv.waitKey(0)
    return resp-48

def labelSet(cells, ids=None):
    pos=[]
    neg=[]
    may=[]
    if ids is None:
        ids = np.arange(len(cells))
    
    for i,cell in zip(ids,cells):
        lab = labelCell(cell)
        if lab == 1:
            neg.append(cell)
        elif lab == 2:
            may.append(cell)
        elif lab == 3:
            pos.append(cell)
        else:
            print(f"Invalid label {lab}, skipping")
        print(f"ID_LABEL:{i}:{lab}")
    cv.destroyAllWindows()
    return neg, may, pos

def testLabelling():
    cells=loadCellsNpz("cells/2_3.npz")
    print(f"Loaded {len(cells)} cells")
    n,m,p = labelSet(cells)
    print(f"Positive: {len(p)}, Maybe: {len(m)}, Negative: {len(n)}")

def plotZstackSharpness(zstack):
    zs, ss = zstackSharpness(zstack)
    #plt.xlim(0,100)
    plt.xlabel("z/um")
    #plt.ylim(90,250)
    plt.ylabel("Sharpness score/a.u.")
    plt.scatter(zs,ss)
    #plt.show()
 #   print(z)
    #plt.savefig("testfig"+str(z)+".png", dpi=400)
#    plt.gcf().clear()

def saveCellsPng(cells, prefix, ids=None):
    if ids is None:
        ids = np.arange(len(cells))
    for i,cell in zip(ids,cells):
        print('Writing:'+"{}_{}.png".format(prefix,i))
        cv.imwrite("{}_{}.png".format(prefix,i),cell)
        #index+=1
        
def saveCellsNpz(cells, filename):
    """
        save the cells as npz file for easier processing later
    """
    np.savez_compressed(filename, cells)
    
def loadCellsNpz(filename):
    res=np.load(filename)
    return res['arr_0'].astype('uint8')

def augment(cells):
    """
        Return cells with augmentations applied (including 0 augmentation)
        rotate by random angle
        flip (horizontal/vertical)
        zoom
        TODO: translate
    """
    angles=range(5,355)
    aug_cells=[]
    for cell in cells:
        for zoom in [0.90,1.0,1.1]:
            rotations=list(np.random.choice(angles,8))
            rotations.append(0) #always include no-modification
            for rotation in rotations:
                for fl in [-2, -1, 0, 1]:
                    new_cell = cv.flip(cell, fl) if fl>-2 else cell.copy()
                    new_cell = ndimage.rotate(new_cell, rotation, reshape=False)
                    new_cell = cv.resize(new_cell, (0,0), fx=zoom, fy=zoom)
                    if zoom<1:
                        new_cell = np.pad(new_cell,
                                         [(round((1-zoom)/2*cell.shape[0]),round((1-zoom)/2*cell.shape[0])),
                                          (round((1-zoom)/2*cell.shape[1]),round((1-zoom)/2*cell.shape[1])),
                                          (0,0)],
                                           'constant',
                                           constant_values=0
                                          )
                        #fix small deviations due to rounding
                        new_cell=cv.resize(new_cell, (cell.shape[0],cell.shape[1]))
                        if not new_cell.shape == cell.shape:
                            print("Failed to pad cell: new:{}, target:{}".format(
                                    new_cell.shape, cell.shape))
                    elif zoom>1:
                        ch=new_cell.shape[0]//2
                        cw=new_cell.shape[1]//2
                        w=cell.shape[1]//2
                        h=cell.shape[0]//2
                        new_cell=new_cell[ch-h:ch+h,cw-w:cw+w]
                        if not new_cell.shape == cell.shape:
                            print("Failed to cut cell: new:{}, target:{}".format(
                                    new_cell.shape, cell.shape))
                    aug_cells.append(new_cell)
    return aug_cells

def checkCellBoring(cell):
    return not checkCell(cell)

def checkCell(cell):
    """
    We want to discard boring cells
    TODO: also check correctly segmented
    """
    if not type(cell) is tuple:
        devs=imgStdev(cell)
        mm=imgMinMax(cell)
    else:
        devs=imgStdev(cell[1])
        mm=imgMinMax(cell[1])
    #|devs| >7  or any channel > 5
    return (np.sqrt((devs**2).sum()) > 16 or len(devs[devs > 8]) > 0
            or mm.max() > 90 or mm.mean() > 70)

def proc_zstack(fname, output_npz, output_folder=None, name=""):
    print(f"Processing {fname}->{output_npz}")
    zstack=readZstack(fname)
    zs,shs=zstackSharpness(zstack)
    max_z = np.array(zs)[shs==max(shs)][0]
    #if the the last one is best, we do not use this
    if max_z < -2.9 or max_z > 2.9:
        print(f"Skipping {fname} (edge), max sharp={max(shs)}")
        return
    #elif max(shs) < 9000:
    #    print(f"Skipping {fname} (value), max sharp={max(shs)}")
    #    return
    print(f"Selected z={max_z}, max sharp={max(shs)}")
    best_img = zstack[max_z]
    #allow gc to free
    zstack=None
    cells=getCells(best_img,(112,112), False, f"{fname}_z{max_z:.3f}.tiff{name}")
    saveCellsNpz(cells, output_npz)
    if output_folder:
        saveCellsPng(cells, output_folder+basename(output_npz).replace('.npz',''))
    #augment([cells[0]])
    #saveCellsPng(cells,"/home/filip/Documents/project/ml/autoencoder/test/1")
#    markers=markCells(best_img, True)
#    cv.imshow('orig',cv.resize(best_img.astype(np.uint8),(1024,768)))
#    cv.waitKey(0)
#    cv.destroyAllW indows()
#    variances
#    for z,img in zstack:

# TESTING SEGMENTATION
##########
#proc_zstack("/home/filip/Documents/project/ml/autoencoder/test_zstack/_x1200.000_y1600.000", "/dev/null", "/home/filip/Documents/porject/ml/autoencoder/test2/")
#img=imread("/home/filip/Documents/project/ml/autoencoder/test_zstack/_x1200.000_y1600.000_z-3.000.tiff")
#markCells(img, False)
#cs=getCells(img, (112,112), False)
#cs=list(filter(checkCell,cs))
#saveCellsPng(cs, '/home/filip/Documents/project/ml/autoencoder/test2/')
#cv.imshow('orig',cv.resize(img.astype(np.uint8),(1024,768)))
#cv.waitKey(0)
#cv.destroyAllWindows()
#exit(0)


#########################
    
#proc_zstack(
#        "/media/filip/827E47CE7E47B9A51/malaria_data/nikon/zstack_grid2/tiffs/_x300.000_y0.000",
#        "testc.npz"
#        )    
#

# We load the entire z-stack into memory 41*~12MB =~500MB/stack
def extractAllCells():
    """
    Extract all cells from zstack_grids
    Save pngs and npz files with cells
    """
    pool=mp.Pool(8) #8 threads
    tasks=[]
    for gr in range(2,11):
        i=0
        root=f"/media/filip/391eb46e-99ae-4ce0-a26f-d29d3f783595/project/nikon/zstack_grid{gr}/"
        for stack in glob(root+"tiffs/*_z-0.00*.tiff"):
            print(f"Queuing: {stack}")
            stack_name=stack.replace('_z-0.000.tiff','')
            tasks.append((stack_name,
                        f"/home/filip/Documents/project/ml/autoencoder/cells/{gr}_{i}.npz",
                        f"/home/filip/Documents/project/ml/autoencoder/cell_pngs/",
                        f"->{gr}_{i}"))
            i+=1
    #tasks=tasks[0]
    #proc_zstack(tasks[0],tasks[1],tasks[2])
    pool.starmap(proc_zstack, tasks)
    
def saveCellBatch(cells, folder, batch_size=10000, index=0):
    """
    Save first batch_size cells from list of cells
    """
    if len(cells) < batch_size:
        return index
    #random.shuffle(cells)
    batch=cells[0:batch_size]
    saveCellsNpz(batch, f"{folder}{index}.npz")
    del cells[0:batch_size]
    index+=1
    return saveCellBatch(cells, folder, batch_size, index)

def relabelTestSetMaybes():
    out_test="./test_data/"
    maybes=loadCellsNpz(f"{out_test}maybe.npz")
    print(f"Labelling {len(maybes)} uncertain cells")
    n,m,p=labelSet(maybes, ids=None)
    print(f"Positive: {len(p)}, Maybe: {len(m)}, Negative: {len(n)}")
    saveCellsNpz(n, f"{out_test}negative2.npz")
    saveCellsNpz(m, f"{out_test}maybe2.npz")
    saveCellsNpz(p, f"{out_test}positive2.npz")

def relabelTrainSetMaybes():
    out_test="./train_orig/"
    maybes=loadCellsNpz(f"{out_test}maybe.npz")
    print(f"Labelling {len(maybes)} uncertain cells")
    n,m,p=labelSet(maybes, ids=None)
    print(f"Positive: {len(p)}, Maybe: {len(m)}, Negative: {len(n)}")
    saveCellsNpz(n, f"{out_test}negative2.npz")
    saveCellsNpz(m, f"{out_test}maybe2.npz")
    saveCellsNpz(p, f"{out_test}positive2.npz")

def splitDatasets():
    src="./cells/"
    out_train="./train_orig/"
    out_test="./test_data/"
    test_png="./test_data_png/"
    npzs=glob(src+"*.npz")
    #testing
    #random.shuffle(npzs)
    #npzs=npzs[0:5]
    
    train_set=[]
    test_set=[]
    test_index=0
    index=0
    total_train=0
    total_test=0
    total_cells=0
    total_train_lab=0
    test_ids=[]
    train_lab_ids=[]
    lab_train_set=[]
    for npz in npzs:
        cells=list(loadCellsNpz(npz))
        cell_ids = np.arange(len(cells))
        total_cells+=len(cells)
        cids = list(zip(cell_ids, cells))
        random.shuffle(cids)
        cids = [list(t) for t in zip(*filter(checkCell,cids))]
        if not cids or len(cids[1]) == 0:
            print(f"WARN: No useful cells found in {npz}")
            continue
        cells=cids[1]
        cell_count=len(cells)
        set_id = npz.replace('.npz','').split('/')[-1]
        ids=list(map(lambda x: f"{set_id}_{x}", cids[0]))
        tsc=cell_count//60+1 #~25000 cells -> ~500 labelled test
        lsc=cell_count//30+1 if cell_count > 1 else 0
        total_test+=tsc
        total_train+=len(cells)-tsc-lsc
        total_train_lab+=lsc
        test_set+=cells[0:tsc]
        lab_train_set+=cells[tsc:tsc+lsc]
        train_lab_ids+=ids[tsc:tsc+lsc]
        test_ids+=ids[0:tsc]
        train_set+=cells[tsc+lsc:]
        index=saveCellBatch(train_set, out_train, 50000, index)
        #test_index=saveCellBatch(test_set, out_test, 50000, test_index)
        #print(f"In memory: {len(train_set)}, {len(test_set)}")
    
    print("Cells loaded")
    print(f"Total cells found: {total_cells}")
    print(f"Train unlabelled: {total_train}")
    print(f"Train labelled: {len(lab_train_set)}")
    print(f"Test: {len(test_set)}")
    #save the remainder of train set
    saveCellsNpz(train_set, f"{out_train}rem.npz")
    print("TEST_SET_LABELS")
    n,m,p=labelSet(test_set, test_ids)
    print(f"Positive: {len(p)}, Maybe: {len(m)}, Negative: {len(n)}")
    saveCellsNpz(n, f"{out_train}negative.npz")
    saveCellsNpz(m, f"{out_train}maybe.npz")
    saveCellsNpz(p, f"{out_train}positive.npz")
    if not os.path.exists("labelled_train/"):
        os.mkdir(f"labelled_train/")
        os.mkdir(f"labelled_train/p/")
        os.mkdir(f"labelled_train/m/")
        os.mkdir(f"labelled_train/n/")
    saveCellsPng(n,"labelled_train/n/")
    saveCellsPng(m,"labelled_train/m/")
    saveCellsPng(p,"labelled_train/p/")
    print("TRAIN_SET_LABELS")
    n,m,p=labelSet(lab_train_set, train_lab_ids)
    print(f"Positive: {len(p)}, Maybe: {len(m)}, Negative: {len(n)}")
    saveCellsNpz(n, f"{out_test}negative.npz")
    saveCellsNpz(m, f"{out_test}maybe.npz")
    saveCellsNpz(p, f"{out_test}positive.npz")
    if not os.path.exists(f"{test_png}/p/"):
        os.mkdir(f"{test_png}/p/")
        os.mkdir(f"{test_png}/n/")
        os.mkdir(f"{test_png}/m/")
    saveCellsPng(n,f"{test_png}/n/")
    saveCellsPng(m,f"{test_png}/m/")
    saveCellsPng(p,f"{test_png}/p/")
    #saveCellsNpz(test_set, f"{out_test}rem.npz")
    #saveCellsPng(test_set, test_png+"t")
    
    #saveCellsNpz(test_set, out_test)


def augmentBatch(batch_no):
    src=f"./train_orig/{batch_no}.npz"
    out=f"./train_data/{batch_no}_"
    train_set=[]
    test_set=[]
    index=0
    count=0
    cells=list(loadCellsNpz(src))
    cell_count=len(cells)
    print(f"Loaded {cell_count} cells.")
    if cell_count == 0:
        return 0 
    for c in cells:
        augd=augment([c]) 
        train_set+=augd  #this produces a lot of cells, cant keep in memory
        count+=len(augd)
        index=saveCellBatch(train_set, out, 1000, index)
#            saveCellsPng(train_set, "/home/filip/Documents/project/ml/autoencoder/test/")
        print(f"Batch:{batch_no}: saved {index}, total:{count}")
    
    print(f"Batch: {batch_no} saving remainder {len(train_set)}")
    saveCellBatch(train_set, out, len(train_set), 999) #index 999 for remainder
    return count

def augmentTrainDataset():
    p=mp.Pool(12) #12 threads because IO delays
    ins=range(0,77)
 #   ins=[76,76]
    results=[]
    p.map_async(augmentBatch, ins, callback=results.append)
    p.close()
    p.join()
    total = 0
    print(results)
    for i in results[0]:
        total+=i
    print(f"Augmented dataset has {total} cells")

def imgMinMax(img):
    sums = img.sum(axis=2)
    nz=img[sums!=0,:]
    mins=nz.min(axis=0)
    maxs=nz.max(axis=0)
    diff=maxs-mins
    return diff #TODO: do we also want the absolute values?
    

def imgStdev(img):
    #batch
    sums = img.sum(axis=2)
    nz=img[sums!=0,:]
    return nz.std(axis=0)

def getStdevs(data):
    """
    Get standard deviations in the 3 channels
    """
    res = []
    for i in data:
        res.append(imgStdev(i))
    return res

def getMinMaxs(data):
    res = []
    for i in data:
        res.append(imgMinMax(i))
    return res

def minMaxHistogramAllData(fromSplit=True):
    """
    Get value ranges for all and plot as histogram
    """
    n_bins=200
    bins=np.arange(-0.5,201.5,1)
    data = []
    if fromSplit:
        traind = loadCellsNpz("train_orig/0.npz")
        testd = loadCellsNpz("test_data/0.npz")
        data = getMinMaxs(testd)
        testd=None
        data += getMinMaxs(traind)
        traind=None #free some memory
    else:
        for f in glob("cells/*.npz"):
            data+=getMinMaxs(loadCellsNpz(f))
            print(f"Processed {f}")
    data=np.array(data)
    fig, axs = plt.subplots(3,1, sharey=False, sharex=True, tight_layout=True)
    axs[0].set_xlim(0,200)
    axs[1].set_xlim(0,200)
    axs[2].set_xlim(0,200)
    axs[0].set_ylabel("Count")
    axs[1].set_ylabel("Count")
    axs[2].set_ylabel("Count")
    axs[2].set_xlabel("Pixel value range")
    axs[0].hist(data[:,0], bins, color='r')
    axs[1].hist(data[:,1], bins, color='g')
    axs[2].hist(data[:,2], bins, color='b')

def stdevsHistogramAllData(fromSplit=True):
    """
    Get stdevs for all and plot as histogram
    """
    n_bins=500
    data = []
    if fromSplit:
        traind = loadCellsNpz("train_orig/0.npz")
        testd = loadCellsNpz("test_data/0.npz")
        data = getStdevs(testd)
        testd=None
        data += getStdevs(traind)
        traind=None #free some memory
    else:
        for f in glob("cells/*.npz"):
            data+=getStdevs(loadCellsNpz(f))
            print(f"Processed {f}")
    
    data=np.array(data)
    mods = np.sqrt((np.array(data)**2).sum(axis=1))
    fig, axs = plt.subplots(2,2, sharey=True, sharex=False, tight_layout=True)
    axs[0][0].set_ylabel("Count")
    axs[0][0].set_xlim(0,20)
    axs[0][1].set_xlim(0,20)
    axs[1][0].set_xlim(0,20)
    axs[1][1].set_xlim(0,40)
    axs[0][0].hist(data[:,0], bins=n_bins, color='r')
    axs[0][1].hist(data[:,1], bins=n_bins, color='g')
    axs[1][0].hist(data[:,2], bins=n_bins, color='b')
    axs[1][0].set_xlabel("Standard deviation/a.u.")
    axs[1][0].set_ylabel("Count")
    axs[1][1].hist(mods, bins=n_bins, color = 'k')
    axs[1][1].set_xlabel("Standard deviation/a.u.")
    fig.savefig("stdev_histograms.png", dpi=300)

def exampleStdevs():
    #load manually selected examples from test set
    cells=loadCellsNpz("test_data/0.npz")
    interesting=[59, 77, 121, 140, 143, 156,3830, 1584, 1593, 3137, 3219, 3229, 3392, 3498, 3399]
    print("#something in cell;")
    i_cells = cells[interesting]
    for dev in getStdevs(i_cells):
        m=(dev**2).sum()
        print(f"{dev[0]} {dev[1]} {dev[2]} {m}")
    print("#empty cell")
    boring = [3390,3430,100,3379, 3273, 3322] #3322 bad segment
    i_cells = cells[boring]
    for dev in getStdevs(i_cells):
        m=(dev**2).sum()
        print(f"{dev[0]} {dev[1]} {dev[2]} {m}")

def splitBoringSet():
    src="./cells/"
    out_train="./train_orig/"
    out_test="./test_data/"
    npzs=glob(src+"*.npz")
    #testing
    #random.shuffle(npzs)
    #npzs=npzs[0:5]
    
    test_set=[]
    boring_set=[]
    boring_ids=[]
    for npz in npzs:
        cells=list(loadCellsNpz(npz))
        cell_ids = np.arange(len(cells))
        cids = list(zip(cell_ids, cells))
        random.shuffle(cids)
        cids = [list(t) for t in zip(*filter(checkCellBoring,cids))]
        if not cids or len(cids[1]) == 0:
            print(f"WARN: No boring cells found in {npz}")
            continue
        cells=cids[1]
        set_id = npz.replace('.npz','').split('/')[-1]
        ids=list(map(lambda x: f"{set_id}_{x}", cids[0]))
        boring_set+=cells
        boring_ids+=ids
        #index=saveCellBatch(train_set, out_train, 50000, index)
        #test_index=saveCellBatch(test_set, out_test, 50000, test_index)
        #print(f"In memory: {len(train_set)}, {len(test_set)}")
    
    print("Boring cells loaded")
    #split, need 2468 cells for the test set, select randomly
    ids = list(zip(boring_ids, boring_set))
    random.shuffle(ids)
    cids = [list(t) for t in zip(*ids)]
    test_set=cids[1][0:2468]
    print("TEST SET IDS:")
    for i in cids[0][0:2468]:
        print(f"{i}")
    print(f"SAVING {len(test_set)} boring test cells.")
    saveCellsNpz(test_set, f"{out_test}boring.npz")
    #save the remainder of train set
    print(f"SAVING {len(cids[1][2468:])} boring train cells.")
    saveCellsNpz(cids[1][2468:], f"{out_train}boring.npz")
    #saveCellsNpz(test_set, f"{out_test}rem.npz")
    #saveCellsPng(test_set, test_png+"t")
    
    #saveCellsNpz(test_set, out_test)

def smoothCellEdge(cell, edge_only=False):
    """
    The sharp edge has plenty of high frequency componenets
    By smoothing it we can make reconstruction easier
    """
    #cv.imshow('orig',cell)
    gr =  cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
    ret, th = cv.threshold(gr, 1 ,255, cv.THRESH_BINARY)
    dist=cv.distanceTransform(th, cv.DIST_L1, 3)
    #cv.imshow('dist',dist.astype('uint8'))
    cp=cell.copy()
    bl=cv.blur(cell,(5,5))
    #bl=cv.blur(bl,(7,7))
    #replace edges with the blured values (this might not work well when single channel is 0)
    if edge_only:
        cp[cp==0]=bl[cp==0]
    else:
        reps=np.repeat(dist[:,:,np.newaxis],3, axis=2)<4 #replace all within 4 pixels of edge
        cp[reps]=bl[reps]
    return cp
    #cv.imshow('blurred',cp)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    
def smoothDatasets(sets=None):
    if sets is None:
        sets=["test_data/positive.npz",
          "test_data/negative.npz",
          "train_orig/positive.npz",
          "train_orig/negative.npz",
          "train_orig/rem.npz"]
    
    boring_sets=["train_orig/boring.npz",
                 "test_data/boring.npz"]
    for s in sets:
        res=[]
        new_name=s.replace(".npz","_s.npz")
        for c in loadCellsNpz(s):
            res.append(smoothCellEdge(c))
        saveCellsNpz(res, new_name)
    
if __name__ == "__main__":
    #print("here")#_x300.000_y400.000
    testz=f"/media/filip/391eb46e-99ae-4ce0-a26f-d29d3f783595/project/nikon/zstack_grid2/tiffs/_x300.000_y400.000"
    zs= proc_zstack(testz, "/dev/null")# stack.replace('_z-0.000.tiff','')
    #splitBoringSet()
 #   cs=loadCellsNpz("test_data/positive.npz")
 #   sd=smoothCellEdge(cs[41])
#    smoothDatasets()
    #relabelTestSetMaybes()
#    stdevsHistogramAllData(False)
#    minMaxHistogramAllData(False)
    #extractAllCells()
    #cells=loadCellsNpz("test_data/0.npz")
    #imgStdev(cells)
    #ugmentTrainDataset()
    #splitDatasets()
    #testLabelling()
#     exampleStdevs()
#    stdevsHistogramAllData()
#augmentBatch(0)
#data_src="/media/filip/827E47CE7E47B9A51/malaria_data/nikon/zstack_grid2/tiffs/_x0.000_y0.000"
#data_src="/home/filip/Documents/project/ml/autoencoder/test_zstack/_x0.000_y0.000"
#proc_zstack(data_src)
#cells=loadCellsNpz('/home/filip/Documents/project/ml/autoencoder/test/c.npz')
#cells=augment([cells[0]])
#saveCellsPng(cells,"/home/filip/Documents/project/ml/autoencoder/test/1")
