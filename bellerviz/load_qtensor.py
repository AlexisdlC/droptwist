# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:01:34 2015

@author: zoeydavidson
"""


import csv
import numpy as np

def load_paramfile(label, path=None, rel=True):
    """
    Load a parameter file from Beller Qtensor simulations
    Inputs
    ------
    label : is the filelabel from the simulation unless rel=False
    rel : defaults True. If false, label must be an absolute path
    Returns
    -------
    dictionary of parameters of simulation from parameters_label.txt
    """
    pdict = {}
    flatten = lambda z: [x for y in z for x in y]
    if rel:
        if path:
            reader = csv.reader(open(path+'parameters_'+label+'.txt', "rb"))
        else:
            reader = csv.reader(open('parameters_'+label+'.txt', "rb"))
    else:
        reader = csv.reader(open(label, "rb"))
    for i, rows in enumerate(reader):
        if i == 1: continue
        k = rows[0].split()[0]
        v = rows[0].split()[1:]
        if not k in pdict:
            pdict[k] = flatten([v])
        else:
            pdict[k].append(v)
    return pdict


def load_dfield(label, path=None, rel=True, Lx=None, Ly=None, Lz=None, outputskiprods=None):
    """
    Load a qtensor file from Beller Qtensor simulations
    Inputs
    ------
    label : is the filelabel from the simulation unless rel=False
    rel : defaults True. If false, label must be an absolute path
    Lx, Ly, Lz, outputskiprods : must be defined if rel=False
    Returns
    -------
    tuple of (S,nx,ny,nz,boundary)
    note: non sim points are zeroed
    """
    if rel:
        if path:
            params = load_paramfile(label,path=path)
        else:
            params = load_paramfile(label)
        Lx = params['Lx'][0]
        Ly = params['Ly'][0]
        Lz = params['Lz'][0]
        outputskiprods = params['outputskiprods'][0]
        qt_label = 'Qtensor_'+Lx+'x'+Ly+'x'+Lz+'_'+label+'.dat'
    else:
        if (Lx>0 and Ly>0 and Lz>0 and (outputskiprods>=0)):
            qt_label = label
        else:
            raise ValueError('Need integer values for Lx,Ly,Lz > 0 & outputskiprods >=0')
    data = np.genfromtxt(path+qt_label)
    #we just want to get rid of that 1 in nz # 999 means empty voxel
    key = np.array([999, 0, 0, 1, 2])
    data = np.where(key == data, [0,0,0,0,0], data)
    #return data
    if outputskiprods > 0:
        Lxyz = [int(i)/int(outputskiprods) for i in [Lx,Ly,Lz]]
    else:
        Lxyz = [int(i) for i in [Lx,Ly,Lz]]
    data_s = np.reshape(data[:,0],(Lxyz[0],Lxyz[1],Lxyz[2]),order='F')
    datax = np.reshape(data[:,1],(Lxyz[0],Lxyz[1],Lxyz[2]),order='F')
    datay = np.reshape(data[:,2],(Lxyz[0],Lxyz[1],Lxyz[2]),order='F')
    dataz = np.reshape(data[:,3],(Lxyz[0],Lxyz[1],Lxyz[2]),order='F')
    data_bound = np.reshape(data[:,4],(Lxyz[0],Lxyz[1],Lxyz[2]),order='F')
    return (data_s,datax,datay,dataz,data_bound)
