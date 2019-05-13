#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:49:48 2019

@author: filip
"""
import scipy
import sys
import numpy as np
from scipy.misc import comb

def exactBi(x,n,p):
    return comb(n,x)*p**x*(1.0-p)**(n-x)

def approxBi(x,n,p):
    return 1/np.sqrt(2*np.pi*n*p*(1-p))*np.exp(-(x-n*p)**2/(2*n*p*(1-p)))

def P_fp(fp, k, fpr):
    return approxBi(fp, k, fpr)

#i will be small so may want to use full binomial
def P_tp(tp, i, tpr):
    if tp>i:
        return 0
    return exactBi(tp, i, tpr)

ppcache={}
def P_P(p,k,i, tpr, fpr):
    """
    Probability p positive cells given k negative cells and i positive cells
    """
    total=0
    key=f"{p}_{k}_{i}_{tpr}_{fpr}"
    if key in ppcache:
        return ppcache[key]

    for tp in range(0, p+1):
        total+=P_tp(tp, i, tpr)*P_fp(p-tp, k, fpr)
    ppcache[key]=total
    return total

def PPR(th, k, i, tpr, fpr):
    total=0
    for p in range(0,th):
        total+=P_P(p,k,i,tpr,fpr)
    return 1-total

def roc(k,i,tpr,fpr, maxth=25000):
    print("#fpr tpr th")
    ths=list(range(0,maxth))
    data=[]
    for th in ths:
        ttpr=PPR(th,k,i,tpr,fpr)
        tfpr=PPR(th,k,0,tpr,fpr)
        data.append([tfpr,ttpr,th])
        print(f"{tfpr} {ttpr} {th}")
        if ttpr < 1e-10 and tfpr < 1e-10:
            break
    data=np.array(data)
    auc=np.trapz(data[:,1],data[:,0])
    print(f"#AUC={auc}")
    return auc,data

tpr=0.975
fpr=0.00173
N=500000
rates=[1e-4, 1e-3, 1e-5, 3e-5, 6e-5]
rates=np.logspace(-5.5,-3,50)
aucs=[]
for rate in rates:
    i=int(N*rate)
    k=N-i
    print(f"#rate:{rate}")
    auc, data=roc(k,i,tpr,fpr,maxth=int((N*fpr+i)*5))
    print("\n")
    aucs.append([rate, auc])

print("#rate auc")
np.savetxt(sys.stdout, np.array(aucs))