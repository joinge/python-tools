from matplotlib.pylab import *
import matplotlib.ticker as ticker
#ion, show, \
#                  clim, cm, colorbar, \
#                  draw, \
#                  get_current_fig_manager, grid, \
#                  ioff, ion, \
#                  legend, \
#                  setp, subplot, subplots_adjust, \
#                  xlabel, xlim,  \
#                  ylabel, ylim

# If missing:
# >> pip install colormath
# >> pip install networkx
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

import mynumpy as np

import pylab as pl
# from gfuncs import swarning, isNpArrayType
from gfuncs import isNpArrayType

import os
GFX_PATH          = '.'
FIG_FORMAT        = 'pdf'
#FIG_FORMAT        = 'png'
INTERACTIVE_MODE  = False
AUTO_EXPORT       = False
FIG_LAYOUT        = True

#pl.rcParams['figure.figsize'] = (11,8)

def setTwoColumn():

   # two-column APS journal format

   # Legend
   pl.rcParams['legend.fontsize'] = 7
   pl.rcParams['legend.markerscale'] = 1
   pl.rcParams['legend.labelspacing'] = 0.5

   # Linewidth
   pl.rcParams['axes.linewidth']  = 0.5
   pl.rcParams['grid.linewidth']  = 0.25
   pl.rcParams['lines.linewidth']  = 0.5
   pl.rcParams['patch.linewidth']  = 0.5

   # Markers
   pl.rcParams['lines.markeredgewidth']  = 0.5
   pl.rcParams['lines.markersize']  = 3

   # Text
   pl.rcParams['axes.labelsize']  = 8
   pl.rcParams['axes.titlesize']  = 8
   pl.rcParams['font.size']       = 8
#    pl.rcParams['axes.labelpad'] = 0

   # Ticks
   pl.rcParams['xtick.labelsize'] = 7
   pl.rcParams['ytick.labelsize'] = 7
   pl.rcParams['xtick.major.pad'] = 2
   pl.rcParams['xtick.minor.pad'] = 2
   pl.rcParams['ytick.major.pad'] = 2
   pl.rcParams['ytick.minor.pad'] = 2
   pl.rcParams['xtick.major.width'] = 0.5
   pl.rcParams['xtick.minor.width'] = 0.25
   pl.rcParams['ytick.major.width'] = 0.5
   pl.rcParams['ytick.minor.width'] = 0.25
   pl.rcParams['xtick.major.size'] = 3
   pl.rcParams['xtick.minor.size'] = 2
   pl.rcParams['ytick.major.size'] = 3
   pl.rcParams['ytick.minor.size'] = 2
#    pl.ticker.

   pl.rcParams['text.usetex'] = True
   pl.rcParams['font.family'] = 'sansserif'
   pl.rcParams['figure.figsize'] = (7.04, 5.5)
#    pl.rcParams['axes'] = [0.125,0.2,0.95-0.125,0.95-0.2]
#    pl.axes().xaxis.LABELPAD = 5
#    pl.tight_layout()

def setSingleColumn():

   # documentclass 'article' with package 'fullpage'
   pl.rcParams['axes.labelsize']  = 10
   pl.rcParams['text.fontsize']   = 10
   pl.rcParams['legend.fontsize'] = 10
   pl.rcParams['xtick.labelsize'] = 8
   pl.rcParams['ytick.labelsize'] = 8
   pl.rcParams['xtick.major.pad'] = 4
   pl.rcParams['xtick.minor.pad'] = 4

#    pl.rcParams['axes.linewidth']  = 0.5
   pl.rcParams['text.usetex']     = True
   pl.rcParams['font.family']     = 'sansserif'
   pl.rcParams['figure.figsize']  = (4.774, 2.950) #(3.44, ?)
#    pl.rcParams['axes'] = [0.150,0.175,0.95-0.15,0.95-0.25]



fig_no = 0
def figure(mode='mosaic',increment_fig_no=True):
   global fig_no
   global INTERACTIVE_MODE

   screen_res = [1920, 1080]

   def make_disp_map(mode, screen_res, fig_width, fig_height, fig_no):

      if mode == 'diagonal':
         max_x = int(pl.floor((screen_res[0]-fig_width)/20))
         max_y = int(pl.floor((screen_res[1]-fig_height)/20))
      elif mode == 'mosaic':
         max_x = int(pl.floor(screen_res[0]/fig_width))
         max_y = int(pl.floor(screen_res[1]/fig_height))

      disp_map = [] #[(0, 60, fig_width, fig_height)]

      top_offset = 60

      for y in range(0,max_y):
         if y == max_y-1:
            mx = max_x-1
         else:
            mx = max_x
         for x in range(0,mx):
            if mode=='mosaic':
               disp_map.append((x*fig_width, top_offset+y*fig_height, fig_width, fig_height))
            elif mode=='diagonal':
               disp_map.append((20*x*y, top_offset+15*y*x, fig_width, fig_height))

      return disp_map[np.mod(fig_no, max_x*max_y-1)]


   fn = pl.figure()
   figman = pl.get_current_fig_manager()
#   fig_geometry = figman.window.geometry().getCoords()
   fig_width  = 200#fig_geometry[2] #0.5*screen_res[0]
   fig_height = 200#fig_geometry[3] #0.7*screen_res[1]
   dm = make_disp_map(mode, screen_res, fig_width, fig_height, fig_no)
   if increment_fig_no:
      fig_no += 1

   if INTERACTIVE_MODE:
#      print('%d %d %d %d' %(dm[0], dm[1], dm[2], dm[3]))
      figman.window.setGeometry(dm[0], dm[1], dm[2], dm[3]) # [0], 85, 800, 600
      return fn

   else:
#      from subprocess import call
#      call(["cp", "%s/waiting.png"%GFX_PATH, "%s/%d.png"%(GFX_PATH,fig_no)])
#      call(["gwenview", "%s/%d.png"%(GFX_PATH,fig_no)])
      return fn

import threading
class SaveMyFig( threading.Thread ):
   def __init__(self, *args, **kwargs):
      threading.Thread.__init__(self)
      self.args = args
      self.kwargs = kwargs

   def run ( self ):
      # When savefig() is called without parameters, figures are stored using successive numbers
      if self.args.__len__() == 0:
         filename = "%s/%d.%s"%(GFX_PATH,fig_no,FIG_FORMAT)
         pl.savefig(filename, **self.kwargs)

      else:
         pl.savefig(*self.args, **self.kwargs)


def savefig(*args, **kwargs):
#   import subprocess as sp
   savefig_thread = SaveMyFig(*args, **kwargs)
   savefig_thread.start()
#   savefig_thread.run( *args, **kwargs )



def close(*args):
   global fig_no
   if args.__len__() != 0 and type(args[0]) == str and args[0] == 'all':
      pl.close('all')
      fig_no = 0
   else:
      pl.close(*args)
      fig_no -= 1
      
def optimalLightness(npts, cmap_type, l_range=(0.0, 1.0)):
    """
    Helper function defining the optimality condition for a colormap.
    Depending on the colormap this might mean different things. The
    l_range argument can be used to restrict the lightness range so it
    does have to equal the full range.
    """
    if cmap_type == "sequential_rising":
        opt = np.linspace(l_range[0], l_range[1], npts)
    elif cmap_type == "sequential_falling": 
        opt = np.linspace(l_range[1], l_range[0], npts)
    elif cmap_type == "diverging_center_top":
        s = npts // 2
        opt = np.empty(npts)
        opt[:s] = np.linspace(l_range[0], l_range[1], s)
        opt[s:] = np.linspace(l_range[1], l_range[0], npts - s)
    elif cmap_type == "diverging_center_bottom":
        s = npts // 2
        opt = np.empty(npts)
        opt[:s] = np.linspace(l_range[1], l_range[0], s)
        opt[s:] = np.linspace(l_range[0], l_range[1], npts - s)
    elif cmap_type == "flat":
        opt = np.ones(npts) * l_range[0]
    return opt
      
def optimizeColormap(name, cmap_type=None, l_range=(0.0, 1.0)):
    cmap = plt.get_cmap(name)
    # Get values, discard alpha
    x = np.linspace(0, 1, 256)
    values = cmap(x)[:, :3]
    lab_colors = []
    for rgb in values:
        lab_colors.append(convert_color(sRGBColor(*rgb), target_cs=LabColor))
        
    if cmap_type == "flat":
        mean = np.mean([_i.lab_l for _i in lab_colors])
        target_lightness = optimalLightness(len(x), cmap_type=cmap_type, l_range=(mean, mean)) 
    else:
        target_lightness = optimalLightness(len(x), cmap_type=cmap_type, l_range=l_range) * 100.0
    
    for color, lightness in zip(lab_colors, target_lightness):
        color.lab_l = lightness
    # Go back to rbg.
    rgb_colors = [convert_color(_i, target_cs=sRGBColor) for _i in lab_colors]
    # Clamp values as colorspace of LAB is larger then sRGB.
    rgb_colors = [(_i.clamped_rgb_r, _i.clamped_rgb_g, _i.clamped_rgb_b) for _i in rgb_colors]
    
    cm = matplotlib.colors.LinearSegmentedColormap.from_list(name=name + "_optimized", colors=rgb_colors)
    return cm


def drawImageBox(ax, center, box_size, style=None, linewidth=1):
   
   uleft  = [center[0]-box_size[0]/2.0, center[1]+box_size[1]/2.0]
   uright = [center[0]+box_size[0]/2.0, center[1]+box_size[1]/2.0]
   bleft  = [center[0]-box_size[0]/2.0, center[1]-box_size[1]/2.0]
   bright = [center[0]+box_size[0]/2.0, center[1]-box_size[1]/2.0]
   
   if not style:
      style = 'r'
      
   xlim = ax.get_xlim()
   ylim = ax.get_ylim()

   ax2 = ax.plot([uleft[0],uright[0],bright[0],bleft[0],uleft[0]], [uleft[1],uright[1],bright[1],bleft[1],uleft[1]],
           style, linewidth=linewidth)
   
   ax.set_xlim(xlim)
   ax.set_ylim(ylim)
   
   return ax2
   
class ImageMosaic:
   def __init__(self, Nw, Nh, map = None,
                cbar_width = 0.05, cbar_margin = 0.04, lmargin = 0.05, rmargin = 0.1, bmargin = 0.1, tmargin = 0.1,
                xlabel='', ylabel='', xlabel_loc="", ylabel_loc="", xlab_adjust=2, ylab_adjust=2, size = None):
      
      if size:
         self.fig = pl.figure(figsize=size)
      else:
         self.fig = pl.figure()
         
      self.cbar_width = cbar_width
      self.cbar_margin = cbar_margin
      self.lmargin = lmargin
      self.rmargin = rmargin
      self.bmargin = bmargin
      self.tmargin = tmargin

      self.xlabel = xlabel
      self.ylabel = ylabel
      
      self.xlabel_loc = xlabel_loc
      self.ylabel_loc = ylabel_loc
      
      self.drawable_width  = 1.0 - lmargin - rmargin - cbar_width - cbar_margin
      self.drawable_height = 1.0 - bmargin - tmargin
      
      fig_dpi = pl.rcParams['figure.dpi']
      if size:
         fig_size = size
      else:
         fig_size = pl.rcParams['figure.figsize']
      font_size = pl.rcParams['font.size']
      
      self.font_width  = font_size/(fig_dpi*fig_size[0])
      self.font_height = font_size/(fig_dpi*fig_size[1])
      
      if xlabel_loc=="top":
         xlab = self.fig.text(self.lmargin + self.drawable_width/2.0,                                                    self.bmargin + self.drawable_height     - xlab_adjust*self.font_height, xlabel, ha='center', va='top', rotation='horizontal')
      else:
         xlab = self.fig.text(self.lmargin + self.drawable_width/2.0,                                                    self.bmargin                            - xlab_adjust*self.font_height, xlabel, ha='center', va='top', rotation='horizontal')
      if ylabel_loc=="right":
         ylab = self.fig.text(self.lmargin + self.drawable_width - (ylab_adjust*self.font_width), self.bmargin + self.drawable_height/2.0,                                ylabel, ha='center', va='center', rotation='vertical')
      else:
         ylab = self.fig.text(self.lmargin                       - (ylab_adjust*self.font_width), self.bmargin + self.drawable_height/2.0,                                ylabel, ha='right', va='center', rotation='vertical')
         
      self.mosaic_width    = self.drawable_width  / Nw
      self.mosaic_height   = self.drawable_height / Nh
      
      self.title = None
      self.Nw = Nw
      self.Nh = Nh
      self.map = map

      if map is None:

         self.axes = []
         self.images = []
         for w in range(Nw):
            haxes = []
            himages = []
            for h in range(Nh):
               ax = None
               ax = self.fig.add_axes([lmargin + w*self.mosaic_width,
                                       1.0 - tmargin - (h+1)*self.mosaic_height,
                                       self.mosaic_width,
                                       self.mosaic_height])
               haxes.append(ax)
               himages.append(None)
               
               if ylabel_loc == 'right':
                  if w<self.Nw-1:
                     pl.setp(ax.get_yticklabels(), visible=False)
                  else:
                     ax.yaxis.tick_right()
               else:
                  if w>0:
                     pl.setp(ax.get_yticklabels(), visible=False)
                  else:
                     ax.yaxis.tick_left()
                        
            self.axes.append(haxes)
            self.images.append(himages)
            
      else:
         self.axes = {}
         self.images = {}
         for w in range(map.shape[0]):

#             for h in range(map.shape[1]):
#             tile_x_start = tile_width
            
            ax = None
            ax = self.fig.add_axes([lmargin + map[w,0]*self.mosaic_width,
                                    1.0 - tmargin - (map[w,1] + map[w,3])*self.mosaic_height,
                                    self.mosaic_width  * map[w,2],
                                    self.mosaic_height * map[w,3]])
            
            self.axes["%d%d%d%d"%tuple(map[w])] = ax
            self.images["%d%d%d%d"%tuple(map[w])] = None
            

   def setTitle(self, title):
      
      pl.rcParams['text.usetex'] = True
      if self.title:
         self.title.set_text(title)
      
      else:
         self.title = self.fig.text(0.5, 1.0 - self.tmargin + 3*self.font_height, title, ha='center', va='center', rotation='horizontal')
         
         
   def showImage(self, x, y, data, Nx = None, Ny = None,
                 interpolation='nearest', aspect='auto', extent=None, cmap=pl.cm.jet, vbounds=None,
                 xbins=3, ybins=2, alpha=1, title= '', xlabel = '', ylabel = '', **kwargs):
      
      if not vbounds:
         vbounds = [data.min(), data.max()]
         
      self.plot_interpolation = interpolation
      self.plot_aspect        = aspect
      self.plot_colormap      = cmap
      self.plot_vbounds       = vbounds
      
      using_mapping = False
      if self.map is not None and Nx and Ny:
         axis  = self.axes["%d%d%d%d"%(x,y,Nx,Ny)]
         image = self.images["%d%d%d%d"%(x,y,Nx,Ny)]
         using_mapping = True
         
      elif x<self.Nw and y<self.Nh:
         axis  = self.axes[x][y]
         image = self.images[x][y]
      else:
         print("ERROR: ImageMosaic.showImage(x=%d,y=%d) out of bounds (Nw=%d, Nh=%d)"%(x,y,self.Nw, self.Nh))
         return

         
      if image == None:
         if extent:
            ax = axis.imshow( data
            , interpolation=interpolation
            , aspect=aspect
            , extent = extent
            , vmin = vbounds[0]
            , vmax = vbounds[1]
            , cmap = cmap
            , alpha = alpha
            , **kwargs
            )
         else:
            ax = axis.imshow( data
            , interpolation=interpolation
            , aspect=aspect
            , vmin = vbounds[0]
            , vmax = vbounds[1]
            , cmap = cmap
            , alpha = alpha
            , **kwargs
            )
         if using_mapping:
            self.images["%d%d%d%d"%(x,y,Nx,Ny)] = ax
         else:
            self.images[x][y] = ax

         axis.locator_params(axis='x', nbins=xbins)
         axis.locator_params(axis='y', nbins=ybins)
         
      else:
         image.set_data(data)
                           
      if title != '':
         axis.set_title(title)
         
      if y<self.Nh-1:
         pl.setp(axis.get_xticklabels(), visible=False)
      else:
         if self.xlabel_loc=="top":
            axis.xaxis.tick_top()
            axis.xaxis.set_label_position('top')
         else:
            axis.xaxis.tick_bottom()
            axis.xaxis.set_label_position('bottom')
      if y==(self.Nh-1)/2:
         axis.set_xlabel(xlabel)
         
      if self.ylabel_loc == "right":
         axis.yaxis.tick_right()
         if x==self.Nw-1:
            axis.yaxis.set_label_position('right')
            axis.set_ylabel(ylabel)
         else:
            pl.setp(axis.get_yticklabels(), visible=False)
      else: #left
         axis.yaxis.tick_left()
         if x==0:
            axis.yaxis.set_label_position('left')
            axis.set_ylabel(ylabel)
         else:
            pl.setp(axis.get_yticklabels(), visible=False)
         
      axis.set_title(title)
      
      return axis
                  
      
   def setColorBar(self, clims,
                   cbar_coord=None,
                   text='Dynamic Range [dBr]',
                   thresholded=[True,True],
                   custom_ticks=[],
                   custom_ticklabels=[]):
      
      cbar_coord = [1.0 - self.cbar_width - self.rmargin,
                    self.bmargin,
                    self.cbar_width,
                    self.drawable_height]
      
      self.cax = self.fig.add_axes(cbar_coord)
      self.cax.grid(False)
      self.cax.imshow(np.linspace(1, 0, 10e3)[:, None],
               aspect='auto',
               extent=(0, 1, clims[0], clims[1]),
               cmap=self.plot_colormap )
      self.cax.yaxis.set_ticks_position('right')
      self.cax.yaxis.set_label_position('right')
      self.cax.set_ylabel(text)
      pl.setp(self.cax.get_xticklabels(), visible=False)

      pl.rcParams['text.usetex'] = True
      
      if len(custom_ticks) > 0:
         self.cax.set_yticks(custom_ticks)
         if len(custom_ticklabels) > 0:
            self.cax.set_yticklabels(custom_ticklabels)
         
         return self.cax
      
      else:
         db_min = self.plot_vbounds[0]
         db_max = self.plot_vbounds[1]
   #       db_min = "$\le$ %d"%db_min
   #       db_max = "$\ge$ %d"%db_max
         
         db_diff = db_max - db_min
         db_step = 5*int(db_diff/50 + 1)
         db_start = int(np.floor((db_min+500)/db_step))*db_step-500
         
         db_ticks = [db_min]
         if thresholded[0]:
            db_labels = ["$\le$ %d"%db_min]
         else:
            db_labels = ["%d"%db_min]
         while True:
            db_start += db_step
            if db_start > db_max - float(db_step)/2+1:
               break
            if db_start - db_min > float(db_step)/2-1:
               db_ticks.append(db_start)
               db_labels.append("%d"%db_start)
         db_ticks.append(db_max)
         
         if thresholded[1]:
            db_labels.append("$\ge$ %d"%db_max)
         else:
            db_labels.append("%d"%db_max)
   
   #       db_labels = ["%s" % item for item in self.cax.get_yticks()]  # [item.get_text() for item in cax.get_yticklabels()]
   #       db_labels[0] = '$\le$ %d'%self.plot_vbounds[0]
         self.cax.set_yticks(db_ticks)
         self.cax.set_yticklabels(db_labels)
   #       self.cax.set_ylabel(self.ylabel)
   #       self.cax.yaxis.set_label_position('left')



   


def plot(*args, **kwargs):
#   from TypeExtensions import Ndarray
   global INTERACTIVE_MODE

   if args.__len__() > 1:
      xaxis = args[0]
      yaxis = args[1]
   else:
      xaxis = None
      yaxis = args[0]


#   def subplot(ax, x, y, **kwargs):
#      if x == None:
#         if isinstance(y, np.Ndarray):
#            try:
#               ax.plot(y.axis, y, **kwargs)
#            except:
#               ax.plot(y, **kwargs)
#            try:
#               ax.set_ylabel(y.axes.name(0))
#            except: pass
#            try:
#               ax.set_xlabel(y.axis.axes.name(0))
#            except: pass
#            try:
#               ax.set_title(y.desc)
#            except: pass
#         else:
#            ax.plot(y, **kwargs)
#      else:
#         ax.plot(x, y, **kwargs)

   try:
      fn = kwargs.pop('fn')
      ax = fn.add_axes([0.1,0.1,0.8,0.8])
   except:
      try:
         ax = kwargs.pop('ax')
      except:
         fn = figure()
         ax = fn.add_axes([0.1,0.1,0.8,0.8])


   # Recursively plot data in 'y'
   def subplot_y(ax, y, **kwargs):
      if isNpArrayType(y.__class__):
         if y.ndim > 1:
            for i in range(y.shape[0]):
               subplot_y(ax, y[i], **kwargs)
         else:
            ax.plot(y, **kwargs)
      elif isinstance(y, list):
         for i in range(y.__len__()):
            subplot_y(ax, y[i], **kwargs)
      else:
         ax.plot(y, **kwargs)

   # Exactly the same code as above, but with common x-axis
   def subplot_xy(ax, x, y, **kwargs):
      if isNpArrayType(y.__class__):
         if y.ndim > 1:
            for i in range(y.shape[0]):
               subplot_xy(ax, x, y[i], **kwargs)
         else:
            ax.plot(x, y, **kwargs)
      elif isinstance(y, list):
         for i in range(y.__len__()):
            subplot_xy(ax, x, y[i], **kwargs)
      else:
         ax.plot(x, y, **kwargs)


   if xaxis == None:
      subplot_y(ax, yaxis, **kwargs)

   elif isinstance(xaxis, list):
      if isinstance(yaxis, list):
         if xaxis.__len__() == yaxis.__len__():
            for x,y in zip(xaxis,yaxis):
               ax.plot(x, y, **kwargs)
         else:
            swarning('x and y-axis of different lengths. Only plotting y.')
            for y in yaxis:
               ax.plot(y, **kwargs)
      else:
         swarning('x-axis is a list, y-axis is not. Only plotting x[0].')
         ax.plot(xaxis[0], yaxis, **kwargs)
   else:
      subplot_xy(ax, xaxis, yaxis, **kwargs)


   if INTERACTIVE_MODE:
      savefig()
      
      
def getWindow(image,
              tx_coords,   # x,y
              tx_angle,    # radians
              img_coords,  # x_min, x_max, y_min, y_max
              width = 0.8, scale = 1):
   Nx,Ny = image.shape
   
   
   N = Nx
   
   img_width  = img_coords[1] - img_coords[0]
   img_length = img_coords[3] - img_coords[2]
   x_mean = np.mean(img_coords[0:2])
   y_mean = np.mean(img_coords[2:4])
   dx = float(img_width)/Nx
   dy = float(img_length)/Ny
   
   image_window = np.ones(image.shape)
   x_old = np.zeros(N)
   x_new = np.zeros(N)
   
#    tx_img_range = img_coords[2]
   r = img_coords[2] + dy/2
   has_overflowed = False
   for row in range(Ny):
      row_data = image_window[:,row]
      
      x_cut = r * np.sin(tx_angle/2)
      
#          x_cut_rel = x_cut / extent[0]
#          w_idx = np.linspace(0,1,N)
      
      x_old[:] = np.linspace(x_mean-(2-width)*x_cut, x_mean+(2-width)*x_cut, N) #*0.5/img_width + 0.5
      x_new[:] = np.linspace(img_coords[0], img_coords[1], Nx) #*0.5/img_width + 0.5
      
      up     = x_new[x_new > x_old[0]]
      updown = up[up <= x_old[-1]]
      
#       print(x_cut, updown.shape[0])
      
      Nlower  = Nx-up.shape[0]
      Nhigher = Nx-updown.shape[0]-Nlower 
      
      transition = (0.5 + 0.5*np.cos(np.linspace(0,np.pi,int(Nx*width*x_cut/(2*img_width)))))*scale + (1-scale)
            
#       fn = None
#       if row == 0:
#          fn = pl.figure()
#          ax = fn.add_subplot(121)
#          ax.plot(transition)
      
      
      Nt = transition.shape[0]
      if N <= 2*Nt:
         w = np.ones(N)
         has_overflowed = True
      else:
         w = np.hstack((2-scale-transition,np.ones(N-2*Nt),transition))
      
      w_new_center = np.interpolate1D(x_old, w, updown, kind='linear', fill_value=1-scale)
      
      w_new = np.hstack((np.ones(Nlower)*(1-scale), w_new_center, np.ones(Nhigher)*(1-scale)))
      
      image_window[:,row] = w_new
      
#       if row == 0:
#          ax = fn.add_subplot(122)
#          ax.plot(w_new)
#          fn.savefig('test2.eps')
      
      r = r + dy
      
   if has_overflowed:
      print("WARNING: Not making a window")
   
   return image_window


def setColorBar(fig,
                cbar_coord, # left, bottom, width, height
                vbounds=[-50,0],
                cmap=pl.cm.YlGnBu_r,
                text='Dynamic Range [dBr]',
                custom_ticks=[],
                custom_ticklabels=[]):
      
      cax = fig.add_axes(cbar_coord)
      cax.grid(False)
      cax.imshow(np.linspace(1, 0, 10e3)[:, None],
               aspect='auto',
               extent=(0, 1, vbounds[0], vbounds[1]),
               cmap=cmap )
      cax.yaxis.set_ticks_position('right')
      cax.yaxis.set_label_position('right')
      cax.set_ylabel(text)

      pl.rcParams['text.usetex'] = True
      
      if len(custom_ticks) > 0:
         cax.set_yticks(custom_ticks)
         if len(custom_ticklabels) > 0:
            cax.set_yticklabels(custom_ticklabels)
         
         return cax
      else:
         
         db_min = vbounds[0]
         db_max = vbounds[1]
         
         db_diff = db_max - db_min
         db_step = 5*int(db_diff/50 + 1)
         db_start = int(np.floor((db_min+500)/db_step))*db_step-500
         
         db_ticks = [db_min]
         db_labels = ["$\le$ %d"%db_min]
         while True:
            db_start += db_step
            if db_start > db_max - float(db_step)/2+1:
               break
            if db_start - db_min > float(db_step)/2-1:
               db_ticks.append(db_start)
               db_labels.append("%d"%db_start)
         db_ticks.append(db_max)
         db_labels.append("$\ge$ %d"%db_max)
   
         cax.set_yticks(db_ticks)
         cax.set_yticklabels(db_labels)
   
   
         pl.setp(cax.get_xticklabels(), visible=False)
         
         return cax



def imshow(X, cmap=None, norm=None, aspect='auto', interpolation='nearest', \
           alpha=None, vmin=None, vmax=None, origin=None,extent=None, \
           shape=None, filternorm=1, filterrad=4.0, imlim=None, \
           resample=None, url=None, hold=None, **kwargs):

   global INTERACTIVE_MODE

   pl.imshow(X, cmap, norm, aspect, interpolation, \
             alpha, vmin, vmax, origin, extent, \
             shape, filternorm, filterrad, imlim, \
             resample, url, hold)

   if INTERACTIVE_MODE:
      savefig()


def ion(*args, **kwargs):
   global INTERACTIVE_MODE

   INTERACTIVE_MODE = True
#   pl.ion(*args, **kwargs)

def computeConturs(path_image):
   import cv2, imutils
      
   # load the image, convert it to grayscale, and blur it slightly
   image = cv2.imread(path_image)
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   gray = cv2.GaussianBlur(gray, (7, 7), 0)

   # perform edge detection, then perform a dilation + erosion to
   # close gaps in between object edges
   edged = cv2.Canny(gray, 50, 100)
   edged = cv2.dilate(edged, None, iterations=1)
   edged = cv2.erode(edged, None, iterations=1)

   # find contours in the edge map
   cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_SIMPLE)
   cnts = cnts[0] if imutils.is_cv2() else cnts[1]
   return cnts
   

def drawImageObjectSizes(path_image,
                         pixel_density=(100,100) # (x,y) [px/m]
                         ):
   
   # import the necessary packages
   from scipy.spatial import distance as dist
   from imutils import perspective
   from imutils import contours
   import numpy as np
   import argparse
   import imutils
   import cv2
   import time
   
   def midpoint(ptA, ptB):
      return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#    # construct the argument parse and parse the arguments
#    ap = argparse.ArgumentParser()
#    ap.add_argument("-i", "--image", required=True,
#       help="path to the input image")
#    ap.add_argument("-w", "--width", type=float, required=True,
#       help="width of the left-most object in the image (in inches)")
#    args = vars(ap.parse_args())

   # load the image, convert it to grayscale, and blur it slightly
   image = cv2.imread(path_image)
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   gray = cv2.GaussianBlur(gray, (7, 7), 0)

   # perform edge detection, then perform a dilation + erosion to
   # close gaps in between object edges
   edged = cv2.Canny(gray, 50, 100)
   edged = cv2.dilate(edged, None, iterations=1)
   edged = cv2.erode(edged, None, iterations=1)

   # find contours in the edge map
   cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_SIMPLE)
   cnts = cnts[0] if imutils.is_cv2() else cnts[1]

   # sort the contours from left-to-right and initialize the
   # 'pixels per metric' calibration variable
   (cnts, _) = contours.sort_contours(cnts)
#    pixel_density = None

   # loop over the contours individually
   for c in cnts:
      # if the contour is not sufficiently large, ignore it
      if cv2.contourArea(c) < 100:
         continue

      # compute the rotated bounding box of the contour
      orig = image.copy()
      box = cv2.minAreaRect(c)
      box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
      box = np.array(box, dtype="int")

      # order the points in the contour such that they appear
      # in top-left, top-right, bottom-right, and bottom-left
      # order, then draw the outline of the rotated bounding
      # box
      box = perspective.order_points(box)
      cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

      # loop over the original points and draw them
      for (x, y) in box:
         cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

      # unpack the ordered bounding box, then compute the midpoint
      # between the top-left and top-right coordinates, followed by
      # the midpoint between bottom-left and bottom-right coordinates
      (tl, tr, br, bl) = box
      (tltrX, tltrY) = midpoint(tl, tr)
      (blbrX, blbrY) = midpoint(bl, br)

      # compute the midpoint between the top-left and top-right points,
      # followed by the midpoint between the top-righ and bottom-right
      (tlblX, tlblY) = midpoint(tl, bl)
      (trbrX, trbrY) = midpoint(tr, br)

      # draw the midpoints on the image
      cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
      cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
      cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
      cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

      # draw lines between the midpoints
      cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
         (255, 0, 255), 2)
      cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
         (255, 0, 255), 2)

      # compute the Euclidean distance between the midpoints
      dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
      dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

      # if the pixels per metric has not been initialized, then
      # compute it as the ratio of pixels to supplied metric
      # (in this case, inches)
#       if pixel_density is None:
#          pixel_density = dB / args["width"]

      # compute the size of the object
      dimA = dA / pixel_density[0]
      dimB = dB / pixel_density[1]

      # draw the object sizes on the image
      cv2.putText(orig, "{:.1f}in".format(dimA),
         (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
         0.65, (255, 255, 255), 2)
      cv2.putText(orig, "{:.1f}in".format(dimB),
         (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
         0.65, (255, 255, 255), 2)

      # show the output image
      fn = pl.figure()
      ax = fn.add_subplot(111)
      ax.imshow(orig)
      pl.show()
#       time.sleep(15)
#       cv2.imshow("Image", orig)
#       cv2.waitKey(0)