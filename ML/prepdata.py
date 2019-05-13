#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:17:39 2019

@author: filip
"""

import numpy as np
from glob import glob
from os import listdir
import os
import sys
import pandas as pd

def extractOverallData(models=["classifier/model1"]):
    alldata={}
    #print(folders)
    for t in ['fil','all']:
        alldata[f'{t}_rocauc']=[]
        alldata[f'{t}_maxacc']=[]
        alldata[f'{t}_acc5']=[]
        alldata[f'{t}_maxaccth']=[]
        alldata[f'{t}_losscorel']=[]
        alldata[f'{t}_lacorrel']=[]
        alldata[f'{t}_prcauc']=[]
        alldata[f"{t}_spec"]=[]
        alldata[f"{t}_sens"]=[]
        alldata[f"{t}_f1"]=[]
    for model in models:
        folders=listdir(f"{model}/")
        for folder in folders:
            if not os.path.isdir(f'{model}/{folder}') or not os.path.isfile(f"{model}/{folder}/data.npy"):
                continue
            data=np.load(f"{model}/{folder}/data.npy").all()
            folder=f"{model}_{folder}"
            print(f"processing {folder}")
            for t in ['fil']:#, 'all']:
                alldata[f'{t}_rocauc'].append([folder, data[t]['roc_auc']])
                atc=np.array(data[t]['atc'])
                alldata[f'{t}_acc5'].append([folder, atc[1,50]])
                alldata[f'{t}_maxacc'].append([folder, atc[1,:].max()])
                alldata[f'{t}_maxaccth'].append([folder, atc[0,atc[1].argmax()]])
                alldata[f'{t}_prcauc'].append([folder, data[t]['roc_auc']])
                alldata[f"{t}_f1"].append([folder, data[t]['f1']])
                conf=np.array(data[t]['conmat'])
                alldata[f"{t}_spec"].append([folder, conf[1,1]/(conf[1].sum())])
                alldata[f"{t}_sens"].append([folder, conf[0,0]/(conf[0].sum())])
                #prc=np.array(data[t]['prc'])
                #roc=np.array(data[t]['roc'])
                #center position in ROC array is not neccessarily the 0.5 threshold value!!!
                #alldata[f'{t}_sensitivity'].append([folder, prc[1,prc.shape[1]//2]])
                #alldata[f'{t}_specificity'].append([folder, 1-roc[0, roc.shape[1]//2]])
                if 'losses' in data[t]:
                    alldata[f"{t}_losscorel"].append([folder,np.corrcoef(data[t]['losses'][:,0],data[t]['losses'][:,1])[0,1]])
                    alldata[f"{t}_lacorrel"].append([folder,np.corrcoef(data[t]['losses'][:,0],data[t]['losses'][:,2])[0,1]])
                else:
                    alldata[f"{t}_losscorel"].append([folder,''])
                    alldata[f"{t}_lacorrel"].append([folder,''])
    return alldata


def formatToLatex(data):
    df=pd.DataFrame(data)
    df[3]= df[3].apply(lambda x:round(100.0*float(x), 1) if x != '' and x!='-' else x)
    for i in range(4,7):
        df[i]=df[i].apply(lambda x:round(float(x), 3) if x != '' and x!='-' else x)
    df.columns = ['Model type', 'Data', 'Reconstruction loss weight', 'Accuracy/%', 'Area under ROC', 'Area under PRC', 'Correlation coefficition']
    return df

def tonumber(s):
    if s == '' or s == '-':
        return 0.0
    return float(s)

def getRWPlotData(data, model=["combined","unfrozen","smooth"]):
    # rw fil_acc
    print('#rw acc roc corel')
    res=[]
    for row in data:
        if row[0] in model and row[1] in model:
           # print(f"{row[0]}+{row[1]}")
            res.append([tonumber(row[2]), tonumber(row[3]), tonumber(row[4]), tonumber(row[6])])
    res=sorted(res, key=lambda x: x[0])
    res=np.array(res)
    np.savetxt(sys.stdout, res)
    return res

def getModelComparisonTable(folders=['model1']):
    alldata = extractOverallData(folders)
    output_keys=['fil_acc5', 'fil_rocauc','fil_prcauc', 'fil_losscorel']
    rownames=np.array(sorted(alldata['fil_acc5'], key=lambda x: x[0]))[:,0]
    rowtitles=[]
    for row in rownames:
        rowtitle=[]
        exploded=row.split('_')
        if(len(folders)>1):
            rowtitle.append(exploded[0])
        if 'rw'  in row:
            weight=exploded[-1].replace('rw','')
            exploded.pop()
        else:
            weight='-'
        datatype=exploded.pop()
        modeltype=exploded.pop()
        if len(exploded) > 1:
            modeltype = f"{exploded[1]}{modeltype}"
        rowtitle.append(modeltype)
        rowtitle.append(datatype)
        rowtitle.append(weight)
        rowtitles.append(rowtitle)
    datatable=np.array(rowtitles)
    for k in output_keys:
        #sort the array
        srtd=sorted(alldata[k], key=lambda x: x[0])
        datatable=np.concatenate((datatable, np.expand_dims(np.array(srtd)[:,1], axis=1)), axis=1)
    datatable
    return datatable,alldata

def getROCs():
    #TODO: add the deeper combined version once it works
    sets=['model1/frozen_smooth', 'model1/combined_smooth_rw300', 'model1/combined_sharp_rw300', 'model1/combined_smooth_rw1100', 'model1/unfrozen_smooth', 'deeper7/frozen_smooth', 'deeper7/unfrozen_smooth', 'deeper7_fix3/']
    for s in sets:
        d=np.load(f"{s}/data.npy").all()
        roc=np.transpose(d['fil']['roc'])
        print(f"#{s}")
        print("#FPR TPR")
        np.savetxt(sys.stdout,roc)
        print("\n")
getROCs()
#data,alldata=getModelComparisonTable(['model1'])
#print("#smooth")
#getRWPlotData(data)
#print("\n")
#print("#sharp")
#getRWPlotData(data, model=['combined','sharp','unfrozen'])