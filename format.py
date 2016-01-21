# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 20:17:36 2015

@author: hutch
"""

import numpy
from scipy import io
import cPickle

def mat2pkl(file_path):
    file_name = file_path.split('/')[-1]
    buffer = io.loadmat(file_path)
    x = buffer['fea']
    #label->class index
    cls = numpy.unique(buffer['gnd']); D = dict(zip(cls, range(len(cls))))
    y = numpy.array([D[label[0]] for label in buffer['gnd']], dtype=numpy.uint32)
    cPickle.dump((x, y), file('../../data/pkl/'+file_name, 'w'))
    

def loadpkl(filepath):
    return cPickle.load(file(filepath))
    
def loadmat(filepath):
    D = io.loadmat(filepath)
    return (D['fea'],numpy.array(D['gnd'].reshape(-1), dtype=numpy.uint32))
    
   
if __name__ == '__main__':
    for file_name in ['mnist', 'pca-rbf', 'coil-20', 'coil-100']:
        try:
            mat2pkl('../../data/mat/'+file_name)
        except Exception, excp:
            print excp
            continue