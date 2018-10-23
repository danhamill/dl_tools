#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:17:32 2018

@author: dhamill
"""

from glob import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import itertools

import numpy as np
from scipy.io import loadmat, savemat
import pandas as pd

import tensorflow as tf
from numpy.lib.stride_tricks import as_strided as ast
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.segmentation import slic
from imageio import imread
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import swifter

#====================================================================


def build_time_series(date,c):
    g = pd.value_counts(c[~np.isnan(c)].flatten())
    pic_date = pd.to_datetime(date,format='%y%m%d%H%M')
    if g.index.isin([1.0]).any():
        return {pic_date:1}
    else:
        return {pic_date:0}

#====================================================================


def build_truth(true_dates,false_dates):
    a=[pd.to_datetime(date,format='%y%m%d%H%M') for date in true_dates]
    b=[pd.to_datetime(date,format='%y%m%d%H%M') for date in false_dates]

    out = {}
    [out.update({date:1}) for date in a]
    [out.update({date:0}) for date in b]
    return out


#====================================================================

def read_mat(i,pred):
    chan_dat = loadmat(i)
    c = chan_dat['river_class'].astype('float')
    p = chan_dat['prob'].astype('float')

    c[np.isneginf(c)] = np.nan
    c[np.isinf(c)] = np.nan
    c[c<0] = np.nan
    name, ext = os.path.splitext(i)
    date = name.split('\\')[-1].split('_')[0][4:]
    pred.update(build_time_series(date,c))
    return pred

#====================================================================

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#====================================================================

def norm_shape(shap):
   '''
   Normalize numpy array shapes so they're always expressed as a tuple,
   even for one-dimensional shapes.
   '''
   try:
      i = int(shap)
      return (i,)
   except TypeError:
      # shape was not a number
      pass

   try:
      t = tuple(shap)
      return t
   except TypeError:
      # shape was not iterable
      pass

   raise TypeError('shape must be an int, or a tuple of ints')


#====================================================================

def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    '''
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    # convert ws, ss, and a.shape to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shap = np.array(a.shape)
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shap),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shap):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shap - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    a = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return a
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    #dim = filter(lambda i : i != 1,dim)

    return a.reshape(dim), newshape

#====================================================================
def norm_im(img): ##, testimage):
   input_mean = 0 #128
   input_std = 255 #128

   #img = imread(image_path)
   nx, ny, nz = np.shape(img)

   float_caster = tf.cast(img, tf.float32)

   dims_expander = tf.expand_dims(float_caster, 0);
   normalized = tf.divide(tf.subtract(dims_expander, [input_mean]), [input_std])
   sess = tf.Session()
   ans = np.squeeze(sess.run(normalized))
   sess.close()
   return ans

def img_pad(img, tile=96):
    try:
        nxo, nyo, nzo = np.shape(img)
        pad_img = norm_im(img)
        pad_img = np.vstack((np.hstack((pad_img,np.fliplr(pad_img))), np.flipud(np.hstack((pad_img,np.fliplr(pad_img))))))
        pad_img = pad_img[:nxo+(nxo % tile),:nyo+(nyo % tile), :]
    except:
        nxo, nyo = np.shape(img)
        pad_img = np.vstack((np.hstack((img,np.fliplr(img))), np.flipud(np.hstack((img,np.fliplr(img))))))
        pad_img = pad_img[:nxo+(nxo % tile),:nyo+(nyo % tile)]
    return pad_img

def get_channel_mask(dat):
    c = dat['river_class'].astype('float')
    c[np.isneginf(c)] = np.nan
    c[np.isinf(c)] = np.nan
    c[c<0] = np.nan
    return c


def img_coords(a,tile,ind):
    grid_pos = np.unravel_index(a, (ind[0], ind[1]))
    t,b,l,r = get_win_pixel_coords(grid_pos,(int(tile/1),int(tile/1)), (int(tile/2),int(tile/2)))
    return t,b,l,r

def slic_infer(i):
    date_str = i['index'][0].strftime('%y%m%d%H%M')
    dat_path = r"C:\workspace\git_clones\dl_tools\RileyCreek_0001\rlc1" +date_str +'_test_96.mat'
    jpg_path = r"D:\PendOriellePhotos\RileyCreek" + os.sep + str(i['index'][0].year)+os.sep + 'no_ice\\rlc1' + date_str + '.jpg'
    img = img_as_float(imread(jpg_path))
    dat = loadmat(dat_path)
    c = dat['class'].astype('float')
    p = dat['prob'].astype('float')
    idx = dat['idx'].astype('int')
    chan_mask= get_channel_mask(dat)



    tile = 96
    pad_img = img_pad(img)
    pad_c = img_pad(c)
    nx, ny, nz = np.shape(pad_img)

    Z,ind = sliding_window(pad_img, (int(tile/1),int(tile/1),3), (int(tile/2),int(tile/2),3))
    Zc,_ = sliding_window(pad_c,(int(tile/1),int(tile/1)), (int(tile/2),int(tile/2)))

    windows = np.unique(idx[chan_mask==1])
    nxo, nyo, nzo = img.shape

    count = 0
    for a in windows:
        grid_pos = np.unravel_index(a, (ind[0], ind[1]))
        t,b,l,r = get_win_pixel_coords(grid_pos,(int(tile/1),int(tile/1)), (int(tile/2),int(tile/2)))
        slic_img = img[t:b, l:r,:]
        slic_i = slic(slic_img, n_segments=9, compactness=10, sigma=1)
        sigma =np.std(np.unique(slic_i, return_counts=True)[1])
        #print('Window ' + str(a) + ' sigma is ' + str(sigma))
        if sigma < 75:
            count+=1
            pad_c[t:b, l:r] = 4
        elif b==144:
            count+=1
            pad_c[t:b, l:r] = 4
        pad_c = pad_c[:nxo,:nyo]
    pad_c[np.isnan(chan_mask)] = np.nan
    
    
    dat.update({'river_class':pad_c})
    savemat(dat_path, dat, do_compression=True)
    #print('[i] Changed '+str(count) + ' of ' + str(len(windows)) + ' windows for ' +str(i.index[0])  )
    del slic_i, slic_img, img, dat, pad_img, pad_c, a, t,b,l,r,c,p,idx,chan_mask
    if len(windows) == count:
        return "fixed"
    else:
        return ""


#====================================================================

def get_win_pixel_coords(grid_pos, win_shape, shift_size=None):
    if shift_size is None:
        shift_size = win_shape
    gr, gc = grid_pos
    sr, sc = shift_size
    wr, wc = win_shape
    top, bottom = gr * sr, (gr * sr) + wr
    left, right = gc * sc, (gc * sc) + wc

    return top, bottom, left, right

def main():
    truth_ice = r"D:\PendOriellePhotos\RileyCreek\*\images_with_ice\*.jpg"
    false_ice = r"D:\PendOriellePhotos\RileyCreek\*\no_ice\*.jpg"
    mat_files = r"C:\workspace\git_clones\dl_tools\RileyCreek_0001\*.mat"

    true_files = glob(truth_ice)
    false_files = glob(false_ice)
    true_dates = [i.split('\\')[-1].split('.')[0][4:] for i in true_files]
    false_dates = [i.split('\\')[-1].split('.')[0][4:] for i in false_files]

    mat_paths = glob(mat_files)

    pred = {}
    #i = mat_paths[0]
    print('[i] Reading memory mapped files...')
    pred = Parallel(n_jobs=-1, verbose=0)(delayed(read_mat)(i, pred ) for i in mat_paths)
    pred = dict((key,d[key]) for d in pred for key in d)
    true = build_truth(true_dates,false_dates)

    df = pd.Series(true).to_frame()
    df1 = pd.Series(pred).to_frame()

    result = df.merge(df1, left_index=True, right_index=True, how='left').dropna()
    result.columns = ['true','pred']
    classes = ['no_ice','ice']

    mis_class = result[(result['true'] == 0) & (result['pred'] == 1)]

    print('[i] Peforming SLIC inference...')
    fixes = 0
    
    mis_class['status'] = mis_class[['index'], ['true'], ['pred']].swifter.apply(slic_infer)
    
    
    for i in range(len(mis_class)):
        row = mis_class.iloc[i:i+1,]
        a = slic_infer(row)
        fixes += a
    print(a)


if __name__ == '__main__':

    main()
'''
mask = result.index.month.isin([4,5,6,7,8,9,10,11])
result.pred[mask] = 0

result.loc[:,'Date'] = result.index.date
g = pd.pivot_table(result,values=['true','pred'],index='Date', aggfunc=lambda x: x.mode().iat[0])
e =precision_recall_fscore_support(g.true,g.pred)
p = np.max(e[0])
r = np.max(e[1])
f = np.max(e[2])
print('Precision : ', str(p))
print('Recall : ', str(r))
print('F-score : ', str(f))



morning = result[result.index.hour< 12]
morning_lumped = pd.pivot_table(morning,values=['true','pred'],index='Date', aggfunc=lambda x: x.mode().iat[0])


cm = confusion_matrix(morning.true,morning.pred)
plot_confusion_matrix(cm, classes,True)


grouped = result.groupby(result.index.hour)
plt.tight_layout()
plot_confusion_matrix(confusion_matrix(g.true, g.pred), classes, True, 'Average')
plt.savefig('Time_Series_Confusion_matrix.png')

for name, group in grouped:
    print(name)
    cm = confusion_matrix(group.true, group.pred)
    plot_confusion_matrix(cm, classes, True, name)
    e =precision_recall_fscore_support(group.true,group.pred)
    p = np.max(e[0])
    r = np.max(e[1])
    f = np.max(e[2])
    print('Precision : ', str(p))
    print('Recall : ', str(r))
    print('F-score : ', str(f))



for name, group in grouped:
    fig,ax = plt.subplots()
    group.plot(subplots=True)
    ax.set_title(name)
    '''
