## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

from __future__ import division
import os, time, sys
from glob import glob
from imageio import imread

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#numerical
import tensorflow as tf
import numpy as np
from scipy.io import savemat, loadmat
from numpy.lib.stride_tricks import as_strided as ast
import random, string

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore")

# suppress divide and invalid warnings
np.seterr(divide='ignore')
np.seterr(invalid='ignore')
np.seterr(all='ignore')
import matplotlib as mpl
mpl.use('agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from scipy.misc import imresize
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

## =========================================================
#def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
#   return ''.join(random.choice(chars) for _ in range(size))

# =========================================================
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


# =========================================================
# Return a sliding window over a in any number of dimensions
# version with no memory mapping
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

# =========================================================
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

# =========================================================
def getCP(tmp,graph):

   #graph = load_graph(classifier_file)

   input_name = "import/Placeholder" #input"
   output_name = "import/final_result"

   input_operation = graph.get_operation_by_name(input_name);
   output_operation = graph.get_operation_by_name(output_name);

   with tf.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: np.expand_dims(tmp, axis=0)})
   results = np.squeeze(results)

   # Sort to show labels of first prediction in order of confidence
   top_k = results.argsort()[-len(results):][::-1]

   return top_k[0], results[top_k[0]] ##, results[top_k] #, np.std(tmp[:,:,0])


# =========================================================
def norm_im(img): ##, testimage):
   input_mean = 0 #128
   input_std = 255 #128

   input_name = "file_reader"
   output_name = "normalized"
   #img = imread(image_path)
   nx, ny, nz = np.shape(img)

   theta = np.std(img).astype('int')
   #try:
   #   file_reader = tf.read_file(testimage, input_name)
   #   image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
   #                                     name='jpeg_reader')
   #   float_caster = tf.cast(image_reader, tf.float32)
   #except:
   float_caster = tf.cast(img, tf.float32)

   dims_expander = tf.expand_dims(float_caster, 0);
   normalized = tf.divide(tf.subtract(dims_expander, [input_mean]), [input_std])


   sess = tf.Session()
   return np.squeeze(sess.run(normalized))



#=======================
def get_semseg(img, tile, decim, classifier_file,chan_dat_file, prob_thres, prob, cmap1, name, out_dir):

   winprop = 1.0

    #===========================================================================
   nxo, nyo, nzo = np.shape(img)
   result = norm_im(img)

    ## pad image so it is divisible by N windows with no remainder
   result = np.vstack((np.hstack((result,np.fliplr(result))), np.flipud(np.hstack((result,np.fliplr(result))))))
    #np.shape(result)
    #result = result[:nxo+np.mod(nxo,tile),:nyo+np.mod(nyo,tile), :]
   result = result[:nxo+(nxo % tile),:nyo+(nyo % tile), :]

   nx, ny, nz = np.shape(result)
   gridy, gridx = np.meshgrid(np.arange(ny), np.arange(nx))
   Zx,_ = sliding_window(gridx, (int(tile/1),int(tile/1)), (int(tile/2),int(tile/2)))#int(tile/1),int(tile/1)
   Zy,_ = sliding_window(gridy, (int(tile/1),int(tile/1)), (int(tile/2),int(tile/2)))#int(tile/1),int(tile/1)
    #len(Zx)

   if decim>1:
      Zx = Zx[::decim]
      Zy = Zy[::decim]

   graph = load_graph(classifier_file)

   #print('CNN ... ')

   w1 = []
   Z,ind = sliding_window(result, (int(tile/1),int(tile/1),3), (int(tile/2),int(tile/2),3))#int(tile/1),int(tile/1)

    #np.shape(Z[3])
   if decim>1:
      Z = Z[::decim]

   #from joblib import Parallel, delayed
   #w1= Parallel(n_jobs=-1, verbose=0)(delayed(getCP)(Z[i],classifier_file) for i in range(len(Z)))

   for i in range(len(Z)):
      w1.append(getCP(Z[i], graph))

    ##C=most likely, P=prob, PP=all probs
   C, P = zip(*w1)

   C = np.asarray(C)
   P = np.asarray(P)

    #np.unique(C)
    #PP = np.asarray(PP)

   C = C+1 #add 1 so all labels are >=1
   
   allones = (Z == np.array(np.ones(Z[i].shape))).all(axis=(1,2))
   idx = np.where(allones)[0][0]
   
   
       #PP = np.squeeze(PP)

       ## create images with classes and probabilities
   Lc = np.zeros((nx, ny,2))
   Lc1 = np.zeros((nx, ny,2))
   Lc2 = np.zeros((nx, ny,2))
   Lc3 = np.zeros((nx, ny,2))

   Lp = np.zeros((nx, ny,2))
   Lp1 = np.zeros((nx, ny,2))
   Lp2 = np.zeros((nx, ny,2))
   Lp3 = np.zeros((nx, ny,2))

   mn = np.int(tile-(tile*winprop)) #tile/2 - tile/4)
   mx = np.int(tile+(tile*winprop)) #tile/2 + tile/4)
   row_count=0
   for row_count in range(15):
      if row_count %2 == 0:
         for k in range(len(Zx))[row_count*21:(row_count+1)*21][::2]:
            Lc[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0] = Lc[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0]+C[k]
            Lp[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0] = Lp[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0]+P[k]
            Lc[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1] = Lc[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1]+k
            Lp[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1] = Lp[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1]+k
         for k in range(len(Zx))[row_count*21:(row_count+1)*21-1][::2]:
            Lc1[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0] = Lc1[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0]+C[k]
            Lp1[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0] = Lp1[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0]+P[k]
            Lc1[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1] = Lc1[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1]+k
            Lp1[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1] = Lp1[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1]+k
      else:
         for k in range(len(Zx))[row_count*21:(row_count+1)*21][::2]:
            Lc2[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0] = Lc2[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0]+C[k]
            Lp2[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0] = Lp2[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0]+P[k]
            Lc2[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1] = Lc2[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1]+k
            Lp2[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1] = Lp2[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1]+k
         for k in range(len(Zx))[row_count*21:(row_count+1)*21-1][::2]:

            Lc3[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0] = Lc3[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0]+C[k]
            Lp3[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0] = Lp3[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],0]+P[k]
            Lc3[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1] = Lc3[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1]+k
            Lp3[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1] = Lp3[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx],1]+k
      row_count +=1

#
# plt.imshow(Lc);plt.colorbar()
# plt.imshow(Lc1);plt.colorbar()
# plt.imshow(Lc2);plt.colorbar()
# plt.imshow(Lc3);plt.colorbar()
# plt.imshow(Lp);plt.colorbar()
# plt.imshow(Lp1);plt.colorbar()
# plt.imshow(Lp2);plt.colorbar()
# plt.imshow(Lp3);plt.colorbar()

   Lp = Lp[:nxo, :nyo,:2]
   Lc = Lc[:nxo, :nyo,:2]
   Lp1 = Lp1[:nxo, :nyo,:2]
   Lc1 = Lc1[:nxo, :nyo,:2]
   Lp2 = Lp2[:nxo, :nyo,:2]
   Lc2 = Lc2[:nxo, :nyo,:2]
   Lp3 = Lp3[:nxo, :nyo,:2]
   Lc3 = Lc3[:nxo, :nyo,:2]
   #np.shape(Lp1[:,:,0])
    #lets do some integer array indexin

   p_stack = np.stack((Lp[:,:,0].copy(),Lp1[:,:,0].copy(),Lp2[:,:,0].copy(),Lp3[:,:,0].copy()))
   p1_stack = np.stack((Lp[:,:,1].copy(),Lp1[:,:,1].copy(),Lp2[:,:,1].copy(),Lp3[:,:,1].copy()))
   c_stack = np.stack((Lc[:,:,0],Lc1[:,:,0],Lc2[:,:,0],Lc3[:,:,0]))
   index = np.argmax(p_stack, axis=0)



   xx = np.arange(nyo)
   aa= np.tile(xx,(nyo,1))
   bb = np.column_stack(tuple(aa))[:nxo,:]
   aa= np.tile(xx,(nyo,1))[:nxo,:]
   Lc_f = c_stack[index,bb,aa]
   Lp_f = p_stack[index,bb,aa]

   #image Lk contains the index from the sliding window
   Lk_k = p1_stack[index,bb,aa]
   #np.max(p_stack, axis=0)
   #np.array_equal(Lp_f,np.max(p_stack, axis=0))
   #plt.imshow(Lp);plt.colorbar()
   Lcorig = Lc_f.copy()

   #look for classificaitons that are at least 75% probabilities
   Lcorig[Lp_f < prob_thres] = np.nan
   #plt.imshow(Lcorig);plt.colorbar()

   chan_dat = loadmat(chan_dat_file)['chan_mask']
   #plt.imshow(chan_dat);plt.colorbar()
   LcChan = Lcorig.copy()
   LcChan[chan_dat ==0 ] = np.nan
   #plt.imshow(img);plt.imshow(LcChan, alpha=0.5);plt.colorbar()


   #np.shape(imgr)
   #plt.imshow(Lcr);plt.colorbar()

   '''
   #print("Writing png file")
   fig = plt.figure(figsize=(18,15))
   fig.subplots_adjust(wspace=0.1)
   ax1 = fig.add_subplot(121)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)
   im = ax1.imshow(img)

   plt.title('a) Input', loc='left', fontsize=10)
   ax1 = fig.add_subplot(122)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)

   im = ax1.imshow(img)
   plt.title('b) CNN prediction', loc='left', fontsize=10)
   im2 = ax1.imshow(LcChan-1, cmap=cmap1, alpha=0.5, vmin=0, vmax=len(labels))
   divider = make_axes_locatable(ax1)
   cax = divider.append_axes("right", size="5%")
   cb=plt.colorbar(im2, cax=cax)
   cb.set_ticks(.5+np.arange(len(labels)+1))
   cb.ax.set_yticklabels(labels)
   cb.ax.tick_params(labelsize=6)
   plt.axis('tight')

   plt.savefig(out_dir+ os.sep + name+'_overlapping_'+str(tile)+'.png', dpi=300, bbox_inches='tight')
   plt.close('all'); del fig

   ###==============================================================
   print("Writing mat file")
   '''
   #os.getcwd()
   #out_dir = '/media/dhamill/My Passport/PendOriellePhotos/ice_class2'
   savemat(out_dir+ os.sep + name+'_test_'+str(tile)+'.mat', {'river_class': LcChan.astype('int'), 'class':Lcorig.astype('float'), 'idx':Lk_k.astype('int'),'prob': Lp_f.astype('float'), 'labels': labels}, do_compression = True) ##'Lpp': Lpp,

   #print("Done!")


def eval_img(imfile, tile, fct, prob, prob_thres, decim, classifier_file,chan_dat_file,out_dir,cmap1):
   tile = np.int(tile)
   fct = np.float(fct)
   prob = np.float(prob)
   prob_thres = np.float(prob_thres)
   decim = np.int(decim)
   #imfile = images[0]
   #print('Image file: '+imfile)
   #try:
   img = imread(imfile)


   name, ext = os.path.splitext(imfile)
   name = name.split(os.sep)[-1]

   get_semseg(img, tile, decim, classifier_file, chan_dat_file, prob_thres, prob, cmap1, name, out_dir)
   return None
#==============================================================
if __name__ == '__main__':
   script, imdirec, out_dir, classifier_file, class_file, colors_path, chan_dat_file, tile, prob_thres, prob, decim, fct = sys.argv

   imdirec, classifier_file, class_file, colors_path, chan_dat_file, tile, prob_thres, prob, decim, fct=r'C:\workspace\git_clones\dl_tools\data\test', \
                                                                                                        r"C:\workspace\git_clones\dl_tools\RileyCreek_96_1000_001.pb", \
                                                                                                        r"C:\workspace\git_clones\dl_tools\labels.txt" ,\
                                                                                                        r"C:\workspace\git_clones\dl_tools\label_colors.txt",\
                                                                                                        r"C:\workspace\git_clones\dl_tools\mres_chanubuntu.mat" ,\
                                                                                                        96, 0.5, 0.5, 1, 0.25



   images = glob(imdirec + os.sep + '*.jpg')
   print('[i] Graph file: '+classifier_file)
   print('[i] Labels file: '+class_file)
   print('[i] Colors file: '+colors_path)
   print('[i] Chan dat file: '+chan_dat_file)
   print('[i] Tile size: '+str(tile))
   print('[i] Prob threshold: '+str(prob_thres))
   print('[i] Probability of DCNN class: '+str(prob))
   print('[i] Decimation factor: '+str(decim))
   print('[i] Image resize factor for CRF: '+str(fct))
   try:
       os.mkdir(out_dir)
   except:
       pass



	## Loads label file, strips off carriage return
   labels = [line.rstrip() for line
                in tf.gfile.GFile(class_file)]

   code= {}
   for label in labels:
      code[label] = [i for i, x in enumerate([x.startswith(label) for x in labels]) if x].pop()


   with open(colors_path) as f: #'labels.txt') as f:
      cols = f.readlines()
   cmap1 = [x.strip() for x in cols]

   ##classes = dict(zip(labels, cmap1))

   cmap1 = colors.ListedColormap(cmap1)
   if os.name=='posix': # true if linux/mac
      start = time.time()
   else: # windows
      start = time.clock()
   #imfile = images[0]
   from joblib import cpu_count
   cpus=cpu_count()
   print("[i] cpu_count", str(cpus))
   w = Parallel(n_jobs=cpus-1, verbose=0)(delayed (eval_img)(imfile, tile, fct, prob, prob_thres, decim, classifier_file,chan_dat_file,out_dir,cmap1) for imfile in images[653:])
   print("[i] Finished the parallel loop")
   if os.name=='posix': # true if linux/mac
      elapsed = (time.time() - start)
   else: # windows
      elapsed = (time.clock() - start)
   print("[i] Processing took "+ str(elapsed/60) + " minutes")
