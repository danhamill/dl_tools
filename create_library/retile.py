## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

## This function will take an directory of images and their associated .mat files (which contain dense pixelwise labels), create image tiles, and sort them into folders based on class

#general
from __future__ import division
from joblib import Parallel, delayed
from glob import glob
import numpy as np
from imageio import imread
from scipy.io import loadmat
import sys, getopt, os
from tile_utils import *
import csv
from scipy.stats import mode as md
from scipy.misc import imsave

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

import os.path as path


# =========================================================
def writeout(tmp, cl, labels, outpath, thres):
   '''
   tmp = tile from jpg image
   cl = tile from label image_names
   labels = list of label strings
   outpath = directory to write organize tiles in
   '''
   np.unique(cl)
   #look for most common classification in tile
   l, cnt = md(cl.flatten())
   l = np.squeeze(l)

   if type(thres) == dict:
      if cnt/len(cl.flatten()) > thres[int(l)]:
         outfile = id_generator()+'.jpg'
         fp = os.path.normpath(outpath+os.sep+labels[l]+os.sep+outfile)
         imsave(fp, tmp)
   else:
      if cnt/len(cl.flatten()) > thres:
         outfile = id_generator()+'.jpg'
         fp = os.path.normpath(outpath+os.sep+labels[l]+os.sep+outfile)
         imsave(fp, tmp)

#==============================================================
if __name__ == '__main__':

   direc = ''; tile = ''; thres = ''; thin=''

   argv = sys.argv[1:]
   try:
      opts, args = getopt.getopt(argv,"ht:a:b:")
   except getopt.GetoptError:
      print('python retile.py -t tilesize -a threshold -b proportion_thin')
      sys.exit(2)

   for opt, arg in opts:
      if opt == '-h':
         print('Example usage: python retile.py -t 96 -a 0.9 -b 0.5')
         sys.exit()
      elif opt in ("-t"):
         tile = arg
      elif opt in ("-a"):
         thres = arg
      elif opt in ("-b"):
         thin = arg

   if not direc:
      direc = r'data\\train'
   if not tile:
      tile = 96
   if not thres:
      thres = .9
   if not thin:
      thin = 0

   tile = int(tile)

   #thres = os.path.normpath('data' + os.sep +'thresh.txt')
   try:
      thres = float(thres)
      print("Single threshold applied to all classes")
   except:
      try:
         with open(thres) as f: #'labels.txt') as f:
            reader = csv.reader(f)
            thres = {int(rows[0]):float(rows[1]) for rows in reader}
            print("Found dictonary of thresholds")
      except:
         thres=float(0.9)
         print('Could not parse threshold csv, reverting to 0.9 threshold')
   thin = float(thin)

   #===============================================
   # Run main application
   Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
   files = askopenfilename(filetypes=[("pick mat files","*.mat")], multiple=True)

   direc = imdirec = os.path.dirname(files[0])##'useimages'

   #=======================================================
   outpath = os.path.normpath(direc+os.sep+'tile_'+str(tile))
   ##files = sorted(glob(direc+os.sep+'*.mat'))

   print("Searching for labels in all %i files" % (len(files)))
   L = []
   for f in files:
      dat = loadmat(files[0])
      if 'labels' in dat.keys():
         labels = dat['labels']
         labels = [label.replace(' ','') for label in labels]
         L.extend(labels)

   labels = np.unique(L).tolist()
   #=======================================================

   #=======================================================
   try:
      os.mkdir(outpath)
   except:
      pass

   for f in labels:
      try:
         os.mkdir(os.path.normpath(outpath+os.sep+f))
      except:
         pass
   #=======================================================

   types = (os.path.normpath(direc+os.sep+'*.jpg'), os.path.normpath(direc+os.sep+'*.jpeg'), \
            os.path.normpath(direc+os.sep+'*.tif'), os.path.normpath(direc+os.sep+'*.tiff'), \
            os.path.normpath(direc+os.sep+'*.png')) # the tuple of file types
   files_grabbed = []
   for f in types:
      files_grabbed.extend(glob(f))

   #=======================================================
   for f in files:
      print('Working on %s' % f)
      dat = loadmat(f)
      if 'labels' in dat.keys():
         labels = dat['labels']
         labels = [label.replace(' ','') for label in labels]

      res = dat['class']
      del dat
      core = f.split('/')[-1].split('_mres')[0]

      ##get the file that matches the above pattern but doesn't contain 'mres'
      fim = [e for e in files_grabbed if e.find(core)!=-1 if e.find('mres')==-1 ]
      if fim:
         fim = fim[0]
         print('Generating tiles from dense class map ....')
         Z,ind = sliding_window(imread(fim), (tile,tile,3), (int(tile/2), int(tile/2),3))

         C,ind = sliding_window(res, (tile,tile), (int(tile/2), int(tile/2)))

         w = Parallel(n_jobs=-1, verbose=0, pre_dispatch='2 * n_jobs', max_nbytes=None)(delayed(writeout)(Z[k], C[k], labels, outpath, thres) for k in range(len(Z)))
      else:
         print("correspodning image not found")

   print('thinning files ...')
   if thin>0:
      for f in labels:
         files = glob(outpath+os.sep+f+os.sep+'*.jpg')
         if len(files)>60:
            usefiles = np.random.choice(files, int(thin*len(files)), replace=False)
            rmfiles = [x for x in files if x not in usefiles.tolist()]
            for rf in rmfiles:
               os.remove(rf)

   for f in labels:
      files = glob(outpath+os.sep+f+os.sep+'*.jpg')
      print(f+': '+str(len(files)))
