from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division


import numpy as np
#import m_general as M
import matplotlib.pyplot as plt
import general as M

from mpl_toolkits.basemap import Basemap,cm, shiftgrid


class NorthPacific_map(object):
    def __init__(self, data, lat, lon, clevs,view_scale=None, unit=None, cmap=None):
        view_scale=view_scale if view_scale is not None else 0.5
        unit=unit if unit is not None else 'no unit'

        gray1='grey'
        gray2='lightgrey'
        gray3='lightgrey'
        self.figure=M.figure_axis_xy(10,6)
        self.subplot=plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0)
        self.map= Basemap(width=13000000,height=8000000,
        			resolution='c',projection='aeqd',\
        			lat_1=-10,lat_2=70,lon_0=180,lat_0=30
        			)

        self.map.fillcontinents(color=gray1,lake_color=gray1)
        self.map.drawcoastlines(color=gray2)

        self.map.drawmeridians(np.arange(0,360,10),labels=[0,0,0,1],fontsize=12, color=gray3)
        self.map.drawparallels(np.arange(-90,90,15),labels=[1,0,0,0],fontsize=12, color=gray3)
        # make up some data on a regular lat/lon grid.\

        print(lon.shape, lat.shape)
        if lon.ndim == 2:
        		lon_X=lon
        		lat_X=lat
        else:
        		lon_X, lat_X=np.meshgrid(lon, lat)
        print(lon_X.shape, lat_X.shape)
        x,y= self.map(lon_X, lat_X)
        self.x=x
        self.y=y
        self.data=data
        self.clevs=clevs

        #cmap1 = plt.cm.gist_earth(np.linspace(0,1,clevs.size))
        #cmap2= LinearSegmentedColormap.from_list("my_colormap", ((0, 1, 0), (1, 0, 0),(1, 1, 1)), N=clevs.size, gamma=1.0)
        cmap=cmap if cmap is not None else plt.cm.ocean_r
        self.cs = self.map.contourf(x,y, data,clevs,cmap=cmap)

        #plt.clabel(self.cs, self.clevs[0:-1:2],inline=1,fontsize=9, fmt='%2.0f', colors='black', rotation=0)

        # add colorbar.
        self.cbar = self.map.colorbar(self.cs,location='right',pad="2%")
        self.cbar.ax.aspect=100
        self.cbar.outline.set_linewidth(0)
        self.cbar.set_label(unit)

        self.map.drawmapscale(-135, 17, -5, 17, 1000, fontsize = 12)

    def title(self, title_str):
		plt.title(title_str, loc='center', y=1.02, fontsize=14)
		#plt.title('Define the title2', loc='left', y=1, fontsize=12)

    def add_contourlines(self, clevs=None, color='white',zorder=12):
    	clevs=clevs if clevs is not None else self.clevs[4:-1:3]
    	self.cont = self.map.contour(self.x,self.y, self.data,clevs, colors=color,linestyles='-')
    	self.cbar.add_lines(self.cont)
        
    def save(self,name=None,path=None, verbose=True):
        import datetime
        import os
        savepath=path if path is not None else os.path.join(os.path.dirname(os.path.realpath('__file__')),'plot/')
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        name=name if name is not None else datetime.date.today().strftime("%Y%m%d_%I%M%p")
        extension='.png'
        full_name= (os.path.join(savepath,name)) + extension
        plt.savefig(full_name, bbox_inches='tight', format='png', dpi=180)
        if verbose:
            print('save with: ',name)


class Pacific_map(object):
    def __init__(self, data, lat, lon, clevs,view_scale=None, unit=None, cmap=None):
        view_scale=view_scale if view_scale is not None else 0.5
        unit=unit if unit is not None else 'no unit'

        gray1='grey'
        gray2='lightgrey'
        gray3='lightgrey'
        self.figure=M.figure_axis_xy(12,8)
        self.subplot=plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0)
        self.map= Basemap(projection='cyl',llcrnrlat=-80,urcrnrlat=60,\
                llcrnrlon=110,urcrnrlon=180+130,resolution='c')

        self.map.fillcontinents(color=gray1,lake_color=gray1)
        self.map.drawcoastlines(color=gray2)

        self.map.drawmeridians(np.arange(0,360,10),labels=[0,0,0,1],fontsize=12, color=gray3)
        self.map.drawparallels(np.arange(-90,90,15),labels=[1,0,0,0],fontsize=12, color=gray3)
        # make up some data on a regular lat/lon grid.\

        print(lon.shape, lat.shape)
        if lon.ndim == 2:
        		lon_X=lon
        		lat_X=lat
        else:
        		lon_X, lat_X=np.meshgrid(lon, lat)
        print(lon_X.shape, lat_X.shape)
        x,y= self.map(lon_X, lat_X)
        self.x=x
        self.y=y
        self.data=data
        self.clevs=clevs

        #cmap1 = plt.cm.gist_earth(np.linspace(0,1,clevs.size))
        #cmap2= LinearSegmentedColormap.from_list("my_colormap", ((0, 1, 0), (1, 0, 0),(1, 1, 1)), N=clevs.size, gamma=1.0)
        cmap=cmap if cmap is not None else plt.cm.ocean_r
        self.cs = self.map.contourf(x,y, data,clevs,cmap=cmap)

        #plt.clabel(self.cs, self.clevs[0:-1:2],inline=1,fontsize=9, fmt='%2.0f', colors='black', rotation=0)

        # add colorbar.
        self.cbar = self.map.colorbar(self.cs,location='right',pad="2%")
        self.cbar.ax.aspect=100
        self.cbar.outline.set_linewidth(0)
        self.cbar.set_label(unit)

        #self.map.drawmapscale(-135, 17, -5, 17, 1000, fontsize = 12)

    def title(self, title_str):
		plt.title(title_str, loc='left', y=1.02, fontsize=16)
		#plt.title('Define the title2', loc='left', y=1, fontsize=12)

    def add_contourlines(self, clevs=None, color='white',zorder=12):
    	clevs=clevs if clevs is not None else self.clevs[4:-1:3]
    	self.cont = self.map.contour(self.x,self.y, self.data,clevs, colors=color,linestyles='-')
    	self.cbar.add_lines(self.cont)
