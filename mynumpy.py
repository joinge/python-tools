from numpy import *

import numpy as np
from gfuncs import isNpScalarType, isNpArrayType

import scipy.interpolate as interpolate

#from numpy import pi, cos, sin

# Dynamic class
class Ndarray(np.ndarray):
   def __str__(self):
      
#      if self.is_transposed:
#         data = self.T
#      else:
#      data = self.base

      dim_size = "["

      if self.shape == (1,) or self.shape == ():
         if issubclass(self.dtype.type, np.integer):
            dim_size = '%d'%self#self.__str__()
         else:
            dim_size = '?' #%f'%self
#             self.__str__()
      else:
         for i in range(0,self.shape.__len__()):
            if i>0:
               dim_size += 'x'
                
            dim_size += "%d"%self.shape[i]
         dim_size += ']'
      
      try:
         return '%-30s\n%s' %(dim_size, np.ndarray.__str__(self))
      except:
         return ''
   

   def __new__(cls, input_array=None, **kwargs):
     
      if input_array is None:
         return np.asarray([]).view(cls)
      
      # To simplify, let scalars become 1D arrays
#      if isNpScalarType(type(input_array)):
#         obj = np.asarray(input_array).reshape((1,)).view(cls)
#      else:

      obj = np.asarray(input_array).view(cls)
      
#      obj.is_transposed = False
      
#     
#     # Add the supplied attributes to the created instance
#     for key in kwargs:
#       setattr(obj, key, kwargs[key])
#
##     obj.__dict__['base'] = obj.base
##     obj.desc = desc
##     obj.shape_desc = shape_desc

      # Finally, we must return the newly created object:
      return obj
      
      def __array_finalize__(self, obj):
         # see InfoArray.__array_finalize__ for comments
         if obj is None:
            return
#
#     if type(obj) == Ndarray or issubclass(type(obj),Ndarray):
#      
#       obj_dict = deepcopy(obj.__dict__)
#       if 'shape_desc' not in obj_dict:
#         return
#       
#       if self.shape.__len__() == obj.shape.__len__()+1:
#         offset = 0
#         for i in range(obj.shape.__len__()):
#            if obj.shape[i] != self.shape[i]:
#              if obj.shape[i] == self.shape[i+1]:
#                break
#         shape_desc = obj_dict.pop('shape_desc')
#         shape_desc = list(shape_desc)
#         shape_desc.insert(i,'1')
#         self.shape_desc = tuple(shape_desc)
#              
#       elif self.shape.__len__() == obj.shape.__len__():
#         self.shape_desc = obj_dict.pop('shape_desc')
#         
#       elif self.shape.__len__() == obj.shape.__len__()-1:
#            shape_desc = obj_dict.pop('shape_desc')
#            shape_desc = list(shape_desc)
#            for i in range(self.shape.__len__()):
#              if obj.shape[i] != self.shape[i]:
#                shape_desc.pop(i)
#                break
#              elif obj.shape[i+1] == self.shape[i]:
#                shape_desc.pop(i+1)
#                break
#            
#            self.shape_desc = tuple(shape_desc)
#            
#         
#       for key in obj_dict:
#         setattr(self, key, getattr(obj, key))
#
#   def __getitem__(self, index):
##   def __array_wrap__(self, arr, context=None):
#   
##     obj = np.ndarray.__getitem__(self, index)
#   
#     shape_desc = []
#     shape_slice = []
#   
#     # 1D assignment; only one slice is received
#     if isinstance(index, slice):
#       
#       # When step is None, the ':' was used.
#       if index.step == None:
#         slice_str = ''
#       else:
#         slice_str = '%d:%d:%d'%(index.start,index.step,index.stop)
#         
#       shape_desc = self.shape_desc
#       shape_slice = 1
#   
#     # Several dimensions
#     else:
#       shape_desc = []
#       shape_slice = []
#       for i in range(index.__len__()):
#         dim = index[i]
#         
#         if isinstance(dim, slice):
#            
#            # When step is None, the ':' was used.
#            shape_desc.append(self.shape_desc[i])
#            shape_slice.append(self.shape[i])
#            
#         elif isinstance(dim, int):
#            
#            if type(self.shape_desc[i]) == tuple:
#              shape_desc.append(self.shape_desc[i][dim])
#            else:
#              shape_desc.append('1')
#            shape_slice.append(1)
#              
#         else:
#            print('hmm')
#         
#       obj = obj.reshape(shape_slice)
#     
#     obj.shape_desc = shape_desc
#     
#     return obj
#   def T(self):
#      self.is_transposed = not self.is_transposed
#      return self.T

def arange(*args, **kwargs):
   return Ndarray(np.arange(*args, **kwargs))

def array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0):
   return Ndarray(np.array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin))

def abs(data):
#   from TypeExtensions import Ndarray
      
#   new_data = []
#   if isinstance(data, list):
#      for i in data:
#         if isinstance(i, Ndarray):
#            new_data.append(i.abs())
#         else:
#            new_data.append(np.abs(i))
#   elif isinstance(data, Ndarray):
#      new_data = np.abs(data) 
      
   if isNpArrayType(type(data)):
      return Ndarray(np.abs(data))
   else:
      return np.abs(data)

def conjugatetranspose(a):
   ''' Call transpose followed by conjugate '''
   return a.transpose().conjugate()

def db(val):
   from gfuncs import isArray
   lt = 10**(-99/20)
   
   if isArray(type(val)):
      val[val<lt] = lt
   elif val < lt:
      val = lt
      
   return 20*np.log10(val)

def dot(*args, **kwargs):
   """What the heck is wrong here?"""
   return Ndarray(np.dot(*args, **kwargs))

def euclidnorm(x):
   ''' Calc the euclidean norm of vector x '''
   return np.vdot(x,x).real**0.5

def exp(*args, **kwargs):
   return Ndarray(np.exp(*args, **kwargs))

def eye(N, M=None, k=0, dtype='float'):
   return Ndarray(np.eye(N, M, k, dtype))

class fft:
   @staticmethod
   def fft(*args, **kwargs):
      return Ndarray(np.fft.fft(*args, **kwargs))
   
   @staticmethod
   def fft2(*args, **kwargs):
      return Ndarray(np.fft.fft2(*args, **kwargs))
   
   @staticmethod
   def rfft(*args, **kwargs):
      return Ndarray(np.fft.rfft(*args, **kwargs))
   
   @staticmethod
   def ifft(*args, **kwargs):
      return Ndarray(np.fft.ifft(*args, **kwargs))
   
   @staticmethod
   def ifft2(*args, **kwargs):
      return Ndarray(np.fft.ifft2(*args, **kwargs))
   
   @staticmethod
   def irfft(*args, **kwargs):
      return Ndarray(np.fft.irfft(*args, **kwargs))

def inner(a,b):
   return Ndarray(np.inner(a,b))

def interpolate1D(x, y, x_new, kind='quadratic', fill_value=0):
   return interpolate.interp1d(x, y, kind=kind, fill_value=fill_value)(x_new)

def interpolate2D(x_idx, y_idx, data, x_idx_up, y_idx_up):  
   new_img = interpolate.RectBivariateSpline(x_idx, y_idx, img, kx=4, ky=4 )(x_up_idx, y_up_idx)
   
   return new_img
   

def linspace(start, stop, num=50, endpoint=True, retstep=False):
   return Ndarray(np.linspace(start, stop, num, endpoint, retstep))

def loadtxt(*args, **kwargs):
   return Ndarray(np.loadtxt(*args, **kwargs))

def log10(*args, **kwargs):
   #args[0][args[0]==0] = 1e-9
   return Ndarray(np.log10(*args, **kwargs))

def ones(shape, dtype=None, order='C'):
   return Ndarray(np.ones(shape, dtype, order))

def outer(a,b):
   return Ndarray(np.outer(a,b))

def mode_1d(data):
   unique_vals = np.unique(data)
   data_freq   = []
   mode = 0
   
   for i in unique_vals:
      data_copy = np.zeros(data.shape)
      data_copy[data==i] = 1
      data_freq.append(sum(data_copy))
   
   max = 0
   idx = 0
   for i in data_freq:
      if i > max:
         max = i
         mode = unique_vals[idx]
      idx += 1
       
   return mode

def mode_(data,dim=0):
    
   dims  = np.size(data.shape)
   
   if dims == 1:
      return mode_1d(data)
   elif dims == 2:
      if dim == 0:
         modes = np.zeros(data.shape[1])
         for i in np.arange(0,data.shape[1],1):
            modes[i] = mode_1d(data[:,i])
      else:
         modes = np.zeros(data.shape[0])
         for i in np.arange(0,data.shape[0],1):
            modes[i] = mode_1d(data[i,:])
   elif dims == 3:
      if dim == 0:
         modes = np.zeros((data.shape[1],data.shape[2]))
         for i in np.arange(0,data.shape[1],1):
            for j in np.arange(0,data.shape[2],1):
               modes[i,j] = mode_1d(data[:,i,j])
      elif dim == 1:
         modes = np.zeros((data.shape[0],data.shape[2]))
         for i in np.arange(0,data.shape[0],1):
            for j in np.arange(0,data.shape[2],1):
               modes[i,j] = mode_1d(data[i,:,j])
      else:   
         modes = np.zeros((data.shape[0],data.shape[1]))
         for i in np.arange(0,data.shape[0],1):
            for j in np.arange(0,data.shape[1],1):
               modes[i,j] = mode_1d(data[i,j,:])
                   
   return modes
   

class random:
   @staticmethod
   def normal(loc=0.0, scale=1.0, size=None):
      return Ndarray(np.random.normal(loc=loc, scale=scale, size=size))
   
   @staticmethod
   def uniform(*args, **kwargs):
      return Ndarray(np.random.uniform(*args, **kwargs))


def vstack(tup):
#   from TypeExtensions import Ndarray
   
   return Ndarray(np.vstack(tup))

def hstack(tup):
#   from TypeExtensions import Ndarray
   
   return Ndarray(np.hstack(tup))

def tukey(N, alpha=0.5):
   '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
   that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
   at \alpha = 0 it becomes a Hann window.
   
   We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
   output
   
   Reference
   ---------
   http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
   
   '''
   # Special cases
   if alpha <= 0:
      return np.ones(N) #rectangular window
   elif alpha >= 1:
      return np.hanning(N)
   
   # Normal case
   x = np.linspace(0, 1, N)
   w = np.ones(x.shape)
   
   # first condition 0 <= x < alpha/2
   first_condition = x<alpha/2
   w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))
   
   # second condition already taken care of
   
   # third condition 1 - alpha / 2 <= x <= 1
   third_condition = x>=(1 - alpha/2)
   w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2)))
   
   return w

def zeros(*args, **kwargs):
   return Ndarray(np.zeros(*args, **kwargs))


def proj1Dto2D(data, Nhx, Nhy, ylims):

#    Nhx,Nhy = Ntheta,int(Ntheta*9.0/16)
   plot_histogram = np.zeros((Nhx,Nhy))
#                x_idx    = np.linspace(0,1,Nhx)
#                y_idx    = np.linspace(0,1,Nhy)
#                x_idx_up = np.linspace(0,1,Nhx)
#                y_idx_up = np.linspace(0,1,Nhy)
   Ni,Nx = data.shape
   
   if Nx != Nhx:
      data = interpolate1D(x=np.linspace(0,1,Nx),
                           y=data,
                           x_new=np.linspace(0,1,Nhx),
                           kind='linear' )
      Ni,Nx = data.shape
      
# #    
#    data = data[0].reshape((1,Nx))
#    Ni = 1
#     
#    x_idx = np.arange(Nx)
#     
#    y_idx = (Nhy*data/ylims[1]).astype(np.int32)
#     
#    y_diff = np.diff(y_idx)
#     
#    plot_histogram[x_idx,y_idx] += 3
#     
#    X_idx,I_idx = np.meshgrid(x_idx,np.arange(Ni))
#     
#    X_diff_mask_ge1 = X_idx[:,1:][y_diff>1]
#    y_mask_ge1      = y_idx[:,1:][y_diff>1]
#    y_diff_mask_ge1 = y_diff[y_diff>1]
#     
#    X_diff_mask_le1 = X_idx[:,1:][y_diff<-1]
#    y_mask_le1      = y_idx[:,1:][y_diff<-1]
#    y_diff_mask_le1 = y_diff[y_diff<-1]
#     
#    x = X_diff_mask_ge1
#    y = y_mask_ge1
#    yd = y_diff_mask_ge1
#     
#    xm = X_diff_mask_le1
#    ym = y_mask_le1
#    ydm = y_diff_mask_le1
#        
#    Nx_mask = x.shape[0]
#    for xi in range(Nx_mask):
#       plot_histogram[x[xi], y[xi]-yd[xi]:y[xi]] += 1
# #       plot_histogram[x[xi], y[xi]] += 3
#        
#    Nxm_mask = xm.shape[0]
#    for xi in range(Nx_mask):
#       try:
#          plot_histogram[xm[xi-1], ym[xi]:ym[xi]-ydm[xi]] += 1
#       except:
#          print("auch")
# #       plot_histogram[xm[xi], ym[xi]] += 3
      

#    import pylab as pl
#    fn=pl.figure();ax=fn.add_subplot(111);ax.imshow(plot_histogram.T,origin='lower',aspect='auto',interpolation='nearest',cmap=pl.cm.gray_r);fn.savefig('test2.eps')
#    print("hello")
# #    
#    import pylab as pl
#    fn=pl.figure();ax=fn.add_subplot(111);ax.imshow(plot_histogram.T,origin='lower',aspect='auto',interpolation='nearest',cmap=pl.cm.gray_r);fn.savefig('test2.eps')
#    y_mask_diff_ge1 = X_idx[:,1:][y_diff>=1]
#    
#    
#    y_mask_l_min = data<ylims[0]
#    y_mask_ge_max = data>=ylims[1]
#    y_mask = y_mask_l_min & y_mask_ge_max
   
      
   
   
#    data_min = data.min()
#    data_max = data.max()

   x_indexes = np.arange(Nx)
   for i in range(Ni):
      y_idx = (Nhy*data[i]/ylims[1]).astype(np.int32)
      
      y = data[i]
      
      x_idx_valid = x_indexes[ (y>ylims[0]) & (y<ylims[1])]
      
      last_y = int(Nhy*(y[0]-ylims[0])/(ylims[1]-ylims[0]))
      for x_idx in x_idx_valid:

         new_y = int(Nhy*(y[x_idx]-ylims[0])/(ylims[1]-ylims[0]))
         
         try:
            if new_y > last_y:
               plot_histogram[x_idx, last_y+1:new_y+1] += 1
            elif new_y < last_y:
               plot_histogram[x_idx, new_y+1:last_y+1] += 1
            else:
               plot_histogram[x_idx, new_y] += 1
                 
            last_y = new_y
           
         except:
            print("auch")

#    import pylab as pl
#    fn=pl.figure();ax=fn.add_subplot(111);ax.imshow(plot_histogram.T,origin='lower',aspect='auto',interpolation='nearest',cmap=pl.cm.gray_r);fn.savefig('test2.eps')
#    print("hello")
   
   return plot_histogram
                  
   #    
   
def unwrap(data):
   
   if len(data.shape) == 1:
      Nx, = data.shape
      data = data.reshape((1,Nx))
      Ni = 1
   else:
      Ni,Nx = data.shape
   
   x = np.arange(Nx)
   y = np.zeros((Ni,Nx))
   for i in range(Ni):
      try:
         y[i,:] = data[i]
         
         # Two pass filtering
         for p in range(2):
            y_diff = np.diff(y)
            
            y_kernel = np.zeros(Nx-1)
            y_kernel[ y_diff[i]>np.pi/2 ]  = -np.pi
            y_kernel[ y_diff[i]<-np.pi/2 ] =  np.pi
            
            y_kernel_conv = np.convolve(y_kernel, np.ones(Nx))
            y[i,1:] += y_kernel_conv[:Nx-1]
         
      except:
         pass
   
   return y

   
#   old_axes = list(ax.name if ax.labels is None 
#               else (ax.name, tuple(ax.labels)) for ax in tup[0].axes)
#
#   
#   
#   if type(new_axis) != list:
#      new_axis = [new_axis]
#   
#   if axes == None:
#      new_data = Ndarray(data, axes=new_axis+old_axes, desc=desc)
#   else:
#      new_data = Ndarray(data, axes=axes, desc=desc)
   
#   return new_data

   

   
