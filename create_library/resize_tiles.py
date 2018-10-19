# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:28:38 2018

@author: RDCRLDDH
"""

from glob import glob
import cv2

from tile_utils import *
import os, sys
if sys.version[0]=='3':
   from tkinter import Tk, Toplevel 
   from tkinter.filedialog import askopenfilename
   import tkinter
   import tkinter as tk
   from tkinter.messagebox import *   
   from tkinter.filedialog import *
else:
   from Tkinter import Tk, TopLevel
   from tkFileDialog import askopenfilename
   import Tkinter as tkinter
   import Tkinter as tk
   from Tkinter.messagebox import *   
   from Tkinter.filedialog import *   
   

from joblib import Parallel, delayed
from scipy.misc import imsave


def resize_images(im,outpath,f):
   dat = cv2.imread(im)
   dat_rshp = cv2.resize(dat,dsize=(96,96),  interpolation=cv2.INTER_CUBIC)
   outfile = id_generator()+'.jpg'
   fp = os.path.normpath(outpath+ os.sep + 'tile_96_resize' +os.sep + f + os.sep + outfile)
   imsave(fp,dat_rshp)



if __name__ == '__main__':
    
    
   Tk().withdraw()
   direc = askdirectory(initialdir=os.getcwd(),title='Please select a containing small tiles')
   #direc = r'C:\workspace\git_clones\dl_tools\data\train\tile_48'
    
   labels = [i.split(os.sep)[-1] for i in glob(direc+os.sep+'*')]
   outpath = os.path.dirname(direc)
   try:   
      os.mkdir(os.path.normpath(outpath+os.sep+'tile_96_resize'))
   except:
      pass
   
   for f in labels:
      try:
         os.mkdir(os.path.normpath(outpath+os.sep+'tile_96_resize' + os.sep + f))
      except:
         pass    
   
   for f in labels:
      files = glob(direc + os.sep + f + os.sep +'*.jpg')
      w = Parallel(n_jobs=-1, verbose=0, pre_dispatch='2 * n_jobs', max_nbytes=None)(delayed(resize_images)(im,outpath,f) for im in files)
    
    
