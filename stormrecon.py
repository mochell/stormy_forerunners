import sys
<<<<<<< HEAD
sys.path.append('/Users/laure/Desktop/stage/travail/modules/stormy_forerunners/')
>>>>>>> 8ef2f2625ab3c1fbf467d06cf002aba67f6f8566

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

#import os
#from matplotlib.dates import DateFormatter, MinuteLocator
#from matplotlib import dates
import datetime as DT
import tools as MT
import spherical_geometry as M_geo
import general as M
import imp
import matplotlib.dates as dates
import os
import warnings
import copy


class ID_tracker(object):
    import datetime as datetime
    def __init__(self, s, date=None):
        self.string=s
        #return self.string

    def add_front(self, s, date=None):
        if date:
            now = self.datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
            self.string=s+'.'+self.string+'_'+now
        else:
            self.string=s+'.'+self.string
        self.string

    def add_back(self, s, date=None):
        if date:
            now = self.datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
            self.string=s+'.'+self.string+'_'+now
        else:
            self.string=self.string+'.'+s
        self.string

    def add(self, s, date=None):
        self.add_front(s,date=date)


class plot_time_chunks_python2(object):
    def __init__(self, time, f, data, pos, ax=None, fig=None, **kwargs):
        #pos=[ 30, 60, 90]

        self.f=f
        self.data=data

        self.time=time
        self.pos=pos
        #if isinstance(time[0], int):
        #    self.time_sec=time
        #elif isinstance(time[0], np.datetime64):
        #    self.time_sec=MT.datetime64_to_sec(time)
        #    #start_date=str(np.datetime64(t).astype('M8[s]'))
        #else:
        #    raise ValueError("unknown pos type")

        #if isinstance(pos[0], int):
        #    self.pos=pos
        #elif isinstance(pos[0], np.datetime64):
        #    print('print convert timeto sec')
        #    self.pos=MT.datetime64_to_sec(pos)
        #    #dates.date2num(time.astype(DT.datetime))
        #    #start_date=str(np.datetime64(t).astype('M8[s]'))
        #else:
        #    raise ValueError("unknown pos type")

        #print(self.time )
        #print(self.data.shape)
        #print('pos',self.pos)
        if type(self.pos[0]) is tuple:
            self.time_chu=data_chunks(self.time,self.pos, 0 )
            self.data_chu=data_chunks(self.data,self.pos, 0 )

        else:
            self.time_chu=data_chunks_split(self.time,self.pos, 0 )
            self.data_chu=data_chunks_split(self.data,self.pos, 0 )

        #print(len(self.time_chu))
        #print(len(self.data_chu))
        #print(len(self.pos), self.pos[0])

        self.Drawer=self.draw_next(**kwargs)
        #ax=self.Drawer.next()
        #contourfdata=plt.contourf(time_chu.next(),f,data_chu.next().T )
        if ax is None:
            self.ax=plt.gca()
        else:
            self.ax=ax

        if fig is None:
            self.fig=plt.gcf()
        else:
            self.fig=fig
        #plt.show()
        plt.ion()
        self.cbarflag=True
    def draw_next(self, **kwargs):
        for i in range(len(self.pos)):
            print(i)
            #plt.show()

            yield self.draw_fig(self.time_chu.next(), self.f, self.data_chu.next(), **kwargs)

        #plt.close()

    def draw_fig(self, time, f, data,clevs,ylim=None ,cmap=None,  **kwargs):
        import matplotlib.colors as colors
        self.ax.clear()
        time_local=time#time_chu.next()
        data_local=data#data_chu.next()
        print('time', time_local.shape)
        print('data', data_local.shape)


        #Figure=M.plot_periodogram(time_local,f[:],data_local[:,:], **kwargs)
        #fig=plt.gcf()
        #M.clevels(data_local[:,:], )

        #Figure.imshow(shading=True, downscale_fac=None, anomalie=False,ax=(self.ax,self.fig), cbar=self.cbarflag)
        #Figure.set_xaxis_to_days(int1=1, int2=2)
        #Figure.ax.set_yscale("linear", nonposy='clip')
        self.clevs=clevs
        cmap=plt.cm.PuBuGn if cmap is None else cmap

        shading='gouraud'
        norm = colors.BoundaryNorm(boundaries=self.clevs, ncolors=256)

        #self.cs=plt.contourf(time_local,f,data_local.T,self.clevs, **kwargs)
        self.cs=plt.pcolormesh(time_local,f,data_local.T,cmap=cmap , norm=norm, shading=shading)
        #self.ax.set_yscale("log", nonposy='clip')

        if self.cbarflag is True:
            self.cbar= plt.colorbar(self.cs,pad=0.01)#, Location='right')#
            self.cbar.ax.aspect=100
            self.cbar.outline.set_linewidth(0)
            #self.cbar.set_label('Power db(' + self.data_unit + '^2/f ')

        if ylim is not None:
            self.ax.set_ylim(ylim[0], ylim[1])
        #self.ax.set_xticklabels(time_local.astype('M8[D]')[trange][::6], minor=False)
        #drawnow(draw_fig)
        #draw_fig()#      The drawnow(makeFig) command can be replaced
        plt.draw()
        self.cbarflag=False
        #self.ax=Figure.ax
        return self.ax


class plot_time_chunks(object):
    def __init__(self, time, f, data, pos, ax=None, fig=None, **kwargs):
        #pos=[ 30, 60, 90]

        self.f=f
        self.data=data

        self.time=time
        self.pos=pos
        #if isinstance(time[0], int):
        #    self.time_sec=time
        #elif isinstance(time[0], np.datetime64):
        #    self.time_sec=MT.datetime64_to_sec(time)
        #    #start_date=str(np.datetime64(t).astype('M8[s]'))
        #else:
        #    raise ValueError("unknown pos type")

        #if isinstance(pos[0], int):
        #    self.pos=pos
        #elif isinstance(pos[0], np.datetime64):
        #    print('print convert timeto sec')
        #    self.pos=MT.datetime64_to_sec(pos)
        #    #dates.date2num(time.astype(DT.datetime))
        #    #start_date=str(np.datetime64(t).astype('M8[s]'))
        #else:
        #    raise ValueError("unknown pos type")

        #print(self.time )
        #print(self.data.shape)
        #print('pos',self.pos)
        if type(self.pos[0]) is tuple:
            self.time_chu=data_chunks(self.time,self.pos, 0 )
            self.data_chu=data_chunks(self.data,self.pos, 0 )

        else:
            self.time_chu=data_chunks_split(self.time,self.pos, 0 )
            self.data_chu=data_chunks_split(self.data,self.pos, 0 )

        #print(len(self.time_chu))
        #print(len(self.data_chu))
        #print(len(self.pos), self.pos[0])

        self.Drawer=self.draw_next(**kwargs)
        #ax=self.Drawer.next()
        #contourfdata=plt.contourf(time_chu.next(),f,data_chu.next().T )
        if ax is None:
            self.ax=plt.gca()
        else:
            self.ax=ax

        if fig is None:
            self.fig=plt.gcf()
        else:
            self.fig=fig
        #plt.show()
        plt.ion()
        self.cbarflag=True
    def draw_next(self, **kwargs):
        for i in range(len(self.pos)):
            print(i)
            #plt.show()

            yield self.draw_fig(self.time_chu.__next__(), self.f, self.data_chu.__next__(), **kwargs)

        #plt.close()

    def draw_fig(self, time, f, data,clevs,ylim=None ,cmap=None,  **kwargs):
        import matplotlib.colors as colors
        self.ax.clear()
        time_local=time#time_chu.next()
        data_local=data#data_chu.next()
        print('time', time_local.shape)
        print('data', data_local.shape)


        #Figure=M.plot_periodogram(time_local,f[:],data_local[:,:], **kwargs)
        #fig=plt.gcf()
        #M.clevels(data_local[:,:], )

        #Figure.imshow(shading=True, downscale_fac=None, anomalie=False,ax=(self.ax,self.fig), cbar=self.cbarflag)
        #Figure.set_xaxis_to_days(int1=1, int2=2)
        #Figure.ax.set_yscale("linear", nonposy='clip')
        self.clevs=clevs
        cmap=plt.cm.PuBuGn if cmap is None else cmap

        shading='gouraud'
        norm = colors.BoundaryNorm(boundaries=self.clevs, ncolors=256)

        #self.cs=plt.contourf(time_local,f,data_local.T,self.clevs, **kwargs)
        self.cs=plt.pcolormesh(time_local,f,data_local.T,cmap=cmap , norm=norm,
        shading=shading)

        #self.ax.set_yscale("log", nonposy='clip')

        if self.cbarflag is True:
            self.cbar= plt.colorbar(self.cs,pad=0.01)#, Location='right')#
            self.cbar.ax.aspect=100
            self.cbar.outline.set_linewidth(0)
            #self.cbar.set_label('Power db(' + self.data_unit + '^2/f ')

        if ylim is not None:
            self.ax.set_ylim(ylim[0], ylim[1])
        #self.ax.set_xticklabels(time_local.astype('M8[D]')[trange][::6], minor=False)
        #drawnow(draw_fig)
        #draw_fig()#      The drawnow(makeFig) command can be replaced
        plt.draw()
        self.cbarflag=False
        #self.ax=Figure.ax
        return self.ax

def data_chunks_split(data, pos, dim):
    if type(data) is np.ndarray:
        datalist=np.split(data,pos, 0)
        for D in datalist:
            yield D

    elif type(data) is list:
        raise ValueError("not porgrammed get")
        print('list')
        datalist=[]
        for L in data:
            print(L.shape)
            #np.split(L,pos, 0)
            datalist.append(np.split(L,pos, 0))

        for k in range(len(datalist[:][1])):
            print(k)
            yield datalist[k][:]

def data_chunks(data, pos, dim):
    if type(data) is np.ndarray:
        datalist=list()
        if dim == 0:
            for pp in pos:
                datalist.append(data[pp[0]:pp[1]])
        elif dim ==1:
            for pp in pos:
                datalist.append(data[:,pp[0]:pp[1]])

        for D in datalist:
            yield D

    elif type(data) is list:
        raise ValueError("not porgrammed get")
        print('list')
        datalist=[]
        for L in data:
            print(L.shape)
            #np.split(L,pos, 0)
            datalist.append(np.split(L,pos, 0))

        for k in range(len(datalist[:][1])):
            print(k)
            yield datalist[k][:]

class PointCollectorv3:
    def __init__(self, ax, Drawer=None):
        self.pointcount=0
        line, = ax.plot([np.nan], [np.nan], marker="o", markersize=4, color="red")
        lineD, = ax.plot([np.nan], [np.nan], marker="o", markersize=8, color="green")
        self.line = line
        self.lineD = lineD
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.D=[np.nan, np.nan]#line.get_xdata(), line.get_ydata())
        self.slopes=list()
        self.P1=[]
        self.P2=[]
        self.D=[]

        #self.ax=ax
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        #self.cid =line.figure.canvas.mpl_connect('key_press_event', self)
        self.Drawer=Drawer

    def __call__(self, event):
        print('click', event)
        if (event.inaxes!=self.line.axes) & (self.Drawer is not None):
            print('next chunk')
            #self.fig.canvas.mpl_disconnect(self.cid)
            newax=self.Drawer.next()
            #newax=self.line.axes
            #print(newax)

            line, = newax.plot([np.nan], [np.nan], marker="o", markersize=4, color="red")
            lineD, = newax.plot([np.nan], [np.nan], marker="o", markersize=8, color="green")

            self.line=line
            self.lineD = lineD
            self.cid =newax.figure.canvas.mpl_connect('button_press_event', self)
            self.pointcount=4
            #return
        if self.pointcount == 0:
            self.pointcount+=1
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)

            self.P1=(event.xdata, event.ydata)

            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()
        elif self.pointcount == 1:
            self.pointcount+=1
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)

            self.P2=(event.xdata, event.ydata)

            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()

        elif self.pointcount >= 2:
            print('its 3')
            self.pointcount=0
            self.D1=(event.xdata,event.ydata)
            self.slopes.append([self.P1, self.P2, self.D1])
            self.D.append(self.D1)
            self.D.append((np.nan,np.nan) )

            self.xs.append(np.nan)
            self.ys.append(np.nan)

            #P1=[]
            #P2=[]
            #D=[]

            self.lineD.set_data(event.xdata, event.ydata)
            self.lineD.figure.canvas.draw()

            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()

class PointCollectorv4:
    def __init__(self, ax, Drawer=None):
        self.pointcount=0
        line, = ax.plot([np.nan], [np.nan], marker="o", markersize=4, color="red")
        lineD, = ax.plot([np.nan], [np.nan], marker="o", markersize=8, color="green")
        self.line = line
        self.lineD = lineD
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.D=[np.nan, np.nan]#line.get_xdata(), line.get_ydata())
        self.slopes=list()
        self.P1=[]
        self.P2=[]
        self.D=[]

        #self.ax=ax
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        #self.cid =line.figure.canvas.mpl_connect('key_press_event', self)
        self.Drawer=Drawer

    def __call__(self, event):
        print('click', event)
        if (event.inaxes!=self.line.axes) & (self.Drawer is not None):
            print('next chunk')
            #self.fig.canvas.mpl_disconnect(self.cid)
            newax=self.Drawer.next()

            line, = newax.plot([np.nan], [np.nan], marker="o", markersize=4, color="red")
            lineD, = newax.plot([np.nan], [np.nan], marker="o", markersize=8, color="green")

            self.line=line
            self.lineD = lineD
            self.cid =newax.figure.canvas.mpl_connect('button_press_event', self)
            self.pointcount=4
            #newax.figure.canvas.draw()
            #return
        if self.pointcount == 0:
            self.pointcount+=1
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)

            self.P1=(event.xdata, event.ydata)

            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()
        elif self.pointcount == 1:
            self.pointcount+=1
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)

            self.P2=(event.xdata, event.ydata)

            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()

        elif self.pointcount >= 2:
            print('its 3')
            self.pointcount=0
            self.D1=(event.xdata,event.ydata)
            self.slopes.append([self.P1, self.P2, self.D1])
            self.D.append(self.D1)
            self.D.append((np.nan,np.nan) )

            self.xs.append(np.nan)
            self.ys.append(np.nan)


            #P1=[]
            #P2=[]
            #D=[]

            self.lineD.set_data(event.xdata, event.ydata)
            self.lineD.figure.canvas.draw()

            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()

def create_listofstorms(slopes, hist=None):

    storm=dict()
    list_of_storms=dict()
    list_of_storms['P1']=[]
    list_of_storms['P2']=[]
    list_of_storms['D']=[]

    hist='list of storms' if hist is None else MT.write_log(hist, 'list of storms')
    for s in slopes:
        #print(s, len(np.array(s)))
        if (sum([None in ss for ss in s]) != 0) or (np.isnan(np.array(s)).any()):
            print('None or NAN values, line Skipped:')
            print(s)
            hist=MT.write_log(hist, 'None or NAN values, line Skipped:'+str(s))
            warnings.warn("Some Point are Nan or None")
        else:

            if s[0][1] > s[1][1]: #pente descendante
                P1, P2 = s[1] , s[0]
            else:
                P2, P1 = s[1] , s[0] #P1:premiere arrivee P2: derniere arrivee
            D=s[2] #liste des points verts

            list_of_storms['P1'].append(P1)
            list_of_storms['P2'].append(P2)
            list_of_storms['D'].append(D)
    list_of_storms['hist']=hist
    return list_of_storms

#def ID_builder(Station, Pol, )

def convert_slope_intersect_to_MS1957(slope, intersect, realtime, verbose=False, as_Dataframe=True):
    """
    this function converts the nondimentional slope and intersect to
    a radial distance in meters and a inital time as datetime64
    """
    Tmin1=  realtime[-1]
    T0  =   realtime[0]
    Dt  =   (Tmin1-T0)

    if intersect.size > 1:
        import pandas as pd
        t0 = pd.to_datetime(realtime[0]) + pd.to_timedelta(intersect *Dt)
    else:
        t0 = Dt *  intersect + T0

    if verbose:
        print(T0)
        print(Dt)

    # intersect_adjusted=Storm.cal_intersect_adjust(params) ## add adjustedintersect here!! estiamted line goes now thourgh maximumo fthe model
    # t0_peak  =   Dt *  intersect_adjusted  +  T0

    Dt_sec=Dt.astype('m8[s]').astype(float)

    dfdt=    slope / Dt_sec

    g=9.8196
    r0= g /(4*np.pi*dfdt )
    if as_Dataframe:
        import pandas as pd
        return pd.DataFrame(data={'r0':r0 , 't0':t0 })
    else:
        return r0, t0

def convert_geo_time_to_dt64(geo):
    import copy
    S=copy.deepcopy(geo)
    for k,I in S.iteritems():
        if type(I) is list:
            S[k][0]=MT.sec_to_dt64(np.array(I[0]))
        elif isinstance(I, np.ndarray):
            S[k]=MT.sec_to_dt64(I)
    S['t0']=MT.sec_to_dt64(np.array(S['t0']))
    S['t0R']=MT.sec_to_dt64(np.array(S['t0R']))
    S['t0L']= MT.sec_to_dt64(np.array(S['t0L']))
    return S
def convert_geo_time_to_float_plot(geo):
    import copy
    S=copy.deepcopy(geo)
    converter=MT.sec_to_float_plot_single
    for k,I in S.iteritems():
        if type(I) is list:
            S[k][0]=converter(I[0])
        elif isinstance(I, np.ndarray):
            S[k]=MT.sec_to_float_plot(I)
    S['t0']=converter(np.array(S['t0']))
    S['t0R']=converter(np.array(S['t0R']))
    S['t0L']= converter(np.array(S['t0L']))
        #elif type(I) is float:
        #    S[k]=MT.sec_to_dt64(np.array(I))
    return S

class Storm(object):
    def __init__(self, ID):
        self.ID=ID
        self.hist='------ | '+ self.ID
        self.write_log('initialized')
        self.fit_dict=False
        self.SM_dict_pandas=None
        #if S is None:
        #date,

    def create_storm_geo(self, P1, P2, D, f, **karg):
        self.f=f
        self.geo=self.geometry(P1, P2, D, **karg)
        self.write_log('created geometry')

    def geometry(self, P1, P2, D, f_margins=0.001):
        #print(P1, P2, D)
        f=self.f
        mf=(P2[0]-P1[0])/(P2[1]-P1[1])
        t0=P1[0]- mf * P1[1]
        t0R=D[0]- mf * D[1]
        delta_t=abs(t0R-t0)
        if t0R > t0:
            t0L=t0-delta_t
        else:
            t0L=t0R
            t0R=t0+delta_t

        f_low=P1[1]-f_margins
        f_high=P2[1]+f_margins

        bound_r=mf*f + t0R
        bound_l=mf*f + t0L


        cline=mf*f+t0
        t0_75l=t0-delta_t*.5
        line75left=mf*f+t0-delta_t*.5
        return {'P1': P1, 'P2': P2, 'D': D,
                  'mf': mf, 't0': t0, 't0R':t0R,'t0L':t0L,'t0_75l':t0_75l,
                  'delta_t':delta_t, 'bound_r':bound_r, 'bound_l':bound_l,
                  'f_low':f_low, 'f_high':f_high,
                  'cline':cline, 'line75left':line75left, 'f_margins':f_margins}

    def plot_stormgeometry(self, time_flag='sec'):
        self.write_log('plotted geometry')
        f=self.f
        if time_flag == 'sec':
            S=self.geo
        elif time_flag == 'dt64':
            S=convert_geo_time_to_dt64(self.geo)
        elif time_flag == 'float_plot':
            S=convert_geo_time_to_float_plot(self.geo)
        else:
            raise ValueError("unknown time_flag")
        print(S['D'][0],S['D'][1])
        plt.plot(S['D'][0],S['D'][1],'.',color='g', markersize=20)
        plt.plot(S['P1'][0],S['P1'][1],'.', c='r', markersize=20)
        plt.plot(S['P2'][0],S['P2'][1],'.', c='r', markersize=20)

        plt.plot(S['t0'],0,'.', c='orange', markersize=20)
        plt.plot(S['t0R'],0,'.', c='orange', markersize=20)
        plt.plot(S['t0L'],0,'.', c='orange', markersize=20)

        plt.plot(S['cline'],f, c='k')
        plt.plot(S['bound_r'],f, c='grey')
        plt.plot(S['bound_l'],f, c='green')
        plt.plot(S['line75left'],f, c='red')
        if time_flag == 'sec':
            plt.plot(np.linspace(S['t0L'],S['bound_r'].max(), 10),np.ones(10)*S['f_low'], c='grey')
            plt.plot(np.linspace(S['t0L'],S['bound_r'].max(), 10),np.ones(10)*S['f_high'], c='grey')

        elif time_flag == 'dt64':
            tx=np.arange(S['t0L'],S['bound_r'].max(), np.timedelta64(1, 'D'))
            plt.plot(tx,np.ones(tx.size)*S['f_low'], c='grey')
            plt.plot(tx,np.ones(tx.size)*S['f_high'], c='grey')
        elif time_flag == 'float_plot':
            tx=np.arange(S['t0L'],S['bound_r'].max(),1)
            plt.plot(tx,np.ones(tx.size)*S['f_low'], c='grey')
            plt.plot(tx,np.ones(tx.size)*S['f_high'], c='grey')

    def plot_cutted_data(self, time_flag='float_plot', **karg ):
        self.write_log('plotted cutted data')
        from decimal import Decimal
        mmin=np.nanmin(self.masked_data)
        mmax=np.nanmax(self.masked_data)
        self.clevs=np.linspace(mmin, mmax, 31)
        #self.clevs=np.arange(0,1+.1,.1)*1e-5
        self.cbarstr=['%.1e' % Decimal(p) for p in self.clevs]
        print(self.masked_data)
        Figure=M.plot_spectrogram(self.time_dict[time_flag],self.f,self.masked_data,
                    #clevs=clevs,
                    sample_unit='1/'+self.dt_unit,
                    ylim=[self.geo['f_low'], self.geo['f_high']], cmap=plt.cm.PuBuGn, clevs=self.clevs, **karg)#(0, .1))

        Figure.imshow(shading=True, downscale_fac=None,anomalie=False, fig_size=[5, 2])


        Figure.set_xaxis_to_days()
        Figure.ax.set_yscale("linear", nonposy='clip')
        Figure.ax.set_title(self.ID)
        Figure.ax.set_ylim(-.001,max([.1, self.f.max()]))
        Figure.cbar.set_ticks(self.clevs)
        Figure.cbar.set_ticklabels(self.cbarstr)
        Figure.F.make_clear_weak()
        return Figure

    def create_mask(self, time):
        self.write_log('masked created')
        f, S = self.f, self.geo
        ll=np.vstack((S['bound_l'], S['bound_r']))
        maskarray=np.logical_and(np.zeros(time.size), True)
        #print(time)
        dt=int(np.diff(time).mean())
        for fi in range(f.size):
            mask=M.cut_nparray(time,ll[:,fi][0]-dt, ll[:,fi][1]+dt)
            maskarray=np.vstack((maskarray, mask))
        maskarray=np.delete(maskarray, 0,0)

        fmask=M.cut_nparray(f, S['f_low'], S['f_high'])
        _, fmaskmesh=np.meshgrid(time, fmask)

        self.mask_full=(fmaskmesh & maskarray).T

    def cut_full_data(self, data):
        self.data=np.copy(data)
        self.data[self.mask_full == False]=np.nan

        #return mdata

<<<<<<< HEAD
    def cut_data(self, time_in, f_data, data, direction, dt_unit, clevs, directional_filtering=False):
=======
    def cut_data(self, time_in, f_data, data, direction, dt_unit, clevs):
>>>>>>> 8ef2f2625ab3c1fbf467d06cf002aba67f6f8566
        import numpy.ma as ma
        self.dt_unit=dt_unit
        self.clevs=clevs
        if type(time_in) is dict:
            time=np.copy(time_in['sec'])
            self.dt_sec=np.diff(time).mean()
        else:
            time=np.copy(time_in)
            self.dt_sec=np.diff(time).mean()

        self.create_mask(time)

        fmask=M.cut_nparray(f_data, self.geo['f_low'], self.geo['f_high'])#np.logical_and(np.zeros(f_data.size)+1, True)
        print("self.geo['f_low']=",self.geo['f_low'])
        #adjsut geometry
        #print(len(fmask))
        self.f=self.f[fmask]
        self.geo['cline']=self.geo['cline'][fmask]
        self.geo['bound_r']=self.geo['bound_r'][fmask]
        self.geo['bound_l']=self.geo['bound_l'][fmask]
        self.geo['line75left']=self.geo['line75left'][fmask]


        # test data shape with time shape
        timemask=M.cut_nparray(time, self.geo['t0L'],self.geo['bound_r'].max())
        #self.xlim=(self.geo['t0L'],self.geo['bound_r'].max())
        #print(timemask)
        #return time[timemask],  S['masked_data'][:, timemask]
        if type(time_in) is dict:
            self.time=time[timemask]
            self.time_dict=dict()
            for k,I in time_in.iteritems():
                self.time_dict[k]=I[timemask]
        else:
            self.time=time[timemask]

        #print(fmask.shape)
        #print(data.shape)
        self.data=np.copy(data[timemask,:][:,fmask])
        #print(self.data.shape)
        self.mask=self.mask_full[timemask,:][:,fmask]

        #print('mask full', self.mask_full.shape)
        #print(self.mask.shape)
        self.masked_data=np.copy(self.data)
        #print(self.masked_data.shape, self.mask.shape)
        self.masked_data[self.mask ==False]=np.nan
        #print('self.mask=',self.mask)
        self.data_masked_array= ma.array(self.data, mask=self.mask)
<<<<<<< HEAD
 

        if directional_filtering==True:
       
            first_not_nan=np.where(np.isnan(self.masked_data[:,0])==False)[0][0]
            print(first_not_nan)
            #print(self.masked_data[:,0])
            print(time_in['dt64'][np.where(time_in['sec']==self.time[first_not_nan])])
        
            last_not_nan=np.where(np.isnan(self.masked_data[:,-1])==False)[0][-1]
            print(last_not_nan)
            #print(self.masked_data[:,-1])
            print(time_in['dt64'][np.where(time_in['sec']==self.time[last_not_nan])])
        
            time_length=last_not_nan-first_not_nan+1
            print('time_length=',time_length)
        
            max_index=np.where(self.masked_data == np.nanmax(self.masked_data))
            print(self.masked_data[max_index])
            print('max_index=', max_index)
            time_index_masked=max_index[0][0]
            freq_index_masked=max_index[1][0]
            time_index=np.where(time_in['sec']==self.time[time_index_masked])
            freq_index=np.where(f_data==self.f[freq_index_masked])
            print('time_index=',time_index, 'freq_index_=',freq_index)
        
        
            max_index=np.where(self.masked_data == np.nanmax(self.masked_data[first_not_nan+1/5*time_length:last_not_nan-1/5*time_length,len(self.f)/4:3*len(self.f)/4]))
            #print('max_index1=', max_index1)
           # print(self.masked_data[max_index])
            print('max_index=', max_index)
            time_index_masked=max_index[0][0]
            freq_index_masked=max_index[1][0]
            time_index=np.where(time_in['sec']==self.time[time_index_masked])
            freq_index=np.where(f_data==self.f[freq_index_masked])
            print('time_index=',time_index, 'freq_index_=',freq_index)
       
            peak_direction=direction[time_index,freq_index]
            print('peak_direction=', peak_direction)
        
            t_initial=self.time[0]
            print('t_initial=', t_initial)
            t_initial_index=np.where(time_in['sec']==t_initial)[0][0]
            print('t_initial=', time_in['dt64'][t_initial_index])
        
            f_initial=self.f[0]
            print('f_initial=', f_initial)
            f_initial_index=np.where(f_data==f_initial)[0][0]
        
            if peak_direction<80:
                for i in range(len(self.masked_data[:,1])):
                    for j in range(len(self.masked_data[1,:])):
                        if peak_direction+80<direction[t_initial_index+i,f_initial_index+j]<360+peak_direction-80:
                            self.masked_data[i,j]=np.nan
            if peak_direction>280:
                for i in range(len(self.masked_data[:,1])):
                    for j in range(len(self.masked_data[1,:])):
                        if peak_direction-80>direction[t_initial_index+i,f_initial_index+j]>(peak_direction+80)-360:
                            self.masked_data[i,j]=np.nan
            if 80<peak_direction<280:
                for i in range(len(self.masked_data[:,1])):
                    for j in range(len(self.masked_data[1,:])):
                        if peak_direction+80<direction[t_initial_index+i,f_initial_index+j] or peak_direction-80>direction[t_initial_index+i,j]:
                            self.masked_data[i,j]=np.nan                
        #print(self.masked_data)          
  
            self.write_log('cutted & assigned data of oroginal shape' + str(data.shape))
            self.write_log('data cutted')
        
        
=======
        #print('self.masked_data=',self.masked_data)
        #print('self.data_masked_array=',self.data_masked_array)
        
        
  ## Directional filtering:
       
        first_not_nan=np.where(np.isnan(self.masked_data[:,0])==False)[0][0]
        print(first_not_nan)
        #print(self.masked_data[:,0])
        print(time_in['dt64'][np.where(time_in['sec']==self.time[first_not_nan])])
        
        last_not_nan=np.where(np.isnan(self.masked_data[:,-1])==False)[0][-1]
        print(last_not_nan)
        #print(self.masked_data[:,-1])
        print(time_in['dt64'][np.where(time_in['sec']==self.time[last_not_nan])])
        
        time_length=last_not_nan-first_not_nan+1
        print('time_length=',time_length)
        
        max_index=np.where(self.masked_data == np.nanmax(self.masked_data))
        print(self.masked_data[max_index])
        print('max_index=', max_index)
        time_index_masked=max_index[0][0]
        freq_index_masked=max_index[1][0]
        time_index=np.where(time_in['sec']==self.time[time_index_masked])
        freq_index=np.where(f_data==self.f[freq_index_masked])
        print('time_index=',time_index, 'freq_index_=',freq_index)
        
        
        max_index=np.where(self.masked_data == np.nanmax(self.masked_data[first_not_nan+1/5*time_length:last_not_nan-1/5*time_length,len(self.f)/4:3*len(self.f)/4]))
        #print('max_index1=', max_index1)
       # print(self.masked_data[max_index])
        print('max_index=', max_index)
        time_index_masked=max_index[0][0]
        freq_index_masked=max_index[1][0]
        time_index=np.where(time_in['sec']==self.time[time_index_masked])
        freq_index=np.where(f_data==self.f[freq_index_masked])
        print('time_index=',time_index, 'freq_index_=',freq_index)
       
        peak_direction=direction[time_index,freq_index]
        print('peak_direction=', peak_direction)
        
        t_initial=self.time[0]
        print('t_initial=', t_initial)
        t_initial_index=np.where(time_in['sec']==t_initial)[0][0]
        print('t_initial=', time_in['dt64'][t_initial_index])
        
        f_initial=self.f[0]
        print('f_initial=', f_initial)
        f_initial_index=np.where(f_data==f_initial)[0][0]
        
        if peak_direction<80:
            for i in range(len(self.masked_data[:,1])):
                for j in range(len(self.masked_data[1,:])):
                    if peak_direction+80<direction[t_initial_index+i,f_initial_index+j]<360+peak_direction-80:
                        self.masked_data[i,j]=np.nan
        if peak_direction>280:
            for i in range(len(self.masked_data[:,1])):
                for j in range(len(self.masked_data[1,:])):
                    if peak_direction-80>direction[t_initial_index+i,f_initial_index+j]>(peak_direction+80)-360:
                        self.masked_data[i,j]=np.nan
        if 80<peak_direction<280:
            for i in range(len(self.masked_data[:,1])):
                for j in range(len(self.masked_data[1,:])):
                    if peak_direction+80<direction[t_initial_index+i,f_initial_index+j] or peak_direction-80>direction[t_initial_index+i,j]:
                        self.masked_data[i,j]=np.nan                
        #print(self.masked_data)          
        
        
  
        self.write_log('cutted & assigned data of oroginal shape' + str(data.shape))
        self.write_log('data cutted')
           
                 
        
        

>>>>>>> 8ef2f2625ab3c1fbf467d06cf002aba67f6f8566
    def load(self, path, verbose=False):
        #load data and attibutes
        D= MT.pickle_load(self.ID,path, verbose)
        for k, v in D.items():
            setattr(self, k, v)
        self.hist= MT.json_load(self.ID,path, verbose)[0]

        #if os.path.isfile(path+self.ID+'.h5'):
        #    with pd.HDFStore(path+self.ID+'.h5') as store2:
        #        #store2 = pd.HDFStore(path+self.ID+'x.h5')
        #        for k,I in store2.iteritems():
        #            setattr(self, k, I)
            #store2.close()
        #return A, B

    def save(self, save_path, verbose=False):
        import warnings
        from pandas import HDFStore
        from pandas.io.pytables import PerformanceWarning
        self.write_log('data saved')

        #save as an npy file with pickle flag False
        #+ Jason for meta data and par numbers.
        if not isinstance(self.SM_dict_pandas, type(None)):
            #SM_dic=self.SM_dict_pandas
            #SM_dic.to_hdf(save_path+self.ID+'.h5','w' )
            warnings.filterwarnings('ignore',category=PerformanceWarning)
            with HDFStore(save_path+self.ID+'.h5') as store:
                store['SM_dict']=self.SM_dict_pandas

            #store['fit_dict']=self.S.fit_dict
            #store['time_dict']=self.S.time_dict
            #del self.SM_dict_pandas

        #savedict=self.__dict__
        #print(savedict)
        savekeys=self.__dict__.keys()
        savekyes_less= list(set(savekeys) - set(['weight_data1d', 'weight','weight1d', 'data1d', 'mask_full', 'minmodel']))
        savedict=dict()
        for k in savekyes_less:
            savedict[k]=self.__dict__[k]

        #deletelist=['weight_data1d', 'weight','weight1d', 'data1d', 'mask_full', 'minmodel']
        #for key in deletelist:
        #    if key in savedict:
        #        del savedict[key]

        MT.pickle_save(self.ID,save_path, savedict, verbose=verbose)
        save_list=[self.hist]
        if self.fit_dict:
            from lmfit import Parameters
            params=Parameters()
            #print('fit dict!!')
            for k,I in self.fit_dict.iteritems():
                if type(I) is bool:
                    I=str(I)

            save_list.append(self.fit_dict)
            #MT.json_save(self.ID,save_path, [self.hist, self.fit_dict], verbose=verbose)
            self.fitter.params.dump(open(save_path+self.ID+'.fittedparrms.json', 'w'))


        MT.json_save(self.ID,save_path,save_list, verbose=verbose)
        MT.save_log_txt(self.ID,save_path, self.hist,  verbose=verbose)

    def normalize_time(self):
        time=np.copy(self.time_dict['sec'])
        dt=np.diff(self.time_dict['sec']).mean()#G.dt_periodogram
        time=(time-time[0])/dt#np.arange(0,time.size, 1)
        self.time_dict['normalized']=(time)/(time[-1])
        self.dt_sec=dt

    def normalize_time_unit(self, t):
        tp=np.copy(t)
        #dt=np.diff(self.time_dict['sec']).mean()#G.dt_periodogram
        return (tp-self.time_dict['sec'][0])/(self.time_dict['sec'][-1]-self.time_dict['sec'][0])
        #return tp/self.time_dict['sec'][-1]

    def denormalize_time_unit(self, t):
        TN=np.copy(t)
        #dt=np.diff(self.time_dict['sec']).mean()#G.dt_periodogram
        return TN * (self.time_dict['sec'][-1]-self.time_dict['sec'][0]) + self.time_dict['sec'][0]
        #return tp/self.time_dict['sec'][-1]

    def slope_to_dfdt_normalized(self):
        self.geo['slope_dfdt_norm']=self.dt_sec*self.time.size/self.geo['mf']
        return self.geo['slope_dfdt_norm']

    #def slope_dfdt_normalized_to_dt_normalize_df(self):
    #self.geo['slope_dfdt_norm']=self.dt_sec*self.time.size/self.geo['mf']
        #return self.geo['slope_dfdt_norm']

    def intersect_sec_to_dfdt_normalized(self, t0_in_sec):
        #return self.normalize_time_unit(t0_in_sec* self.geo['mf']/(self.geo['mf']-1) )
        return self.intersect_norm_to_dfdt_normalized(self.normalize_time_unit(t0_in_sec))

    def intersect_norm_to_dfdt_normalized(self, t0_in_norm):
        local_slope=self.slope_to_dfdt_normalized()
        return t0_in_norm* local_slope/(local_slope-1)

    #def dfdt_intersect_to_intersect_norm(self, intersect):
    #    local_slope=self.slope_to_dfdt_normalized()
    #    return t0_in_norm* local_slope/(local_slope-1)



    def substract_plain_simple(self, datasub=None, verbose=False):
        import brewer2mpl
        self.write_log('substract Plain use freq end points:')

        if datasub is None:
            datasub=self.data
            self.write_log('used data')
        else:
            datasub=datasub
            self.write_log('used prescribed data')

        time=self.time_dict['normalized']
        f=self.f

        yu=datasub.T.mean(1)[-4:-1].mean()
        yl=datasub.T.mean(1)[0:3].mean()
        #yu=datasub.T.mean(1)[-1].mean()
        #yl=datasub.T.mean(1)[0].mean()

        m=(yu-yl)/(f[-1]-f[0])
        yi =-m*f[0]+datasub.T.mean(1)[0]
        y= m*f+yi
        tt, yy=np.meshgrid(time, y)


        resid= (datasub.T - yy).T
        minpoint=0*abs(np.nanmin(resid))
        resid=resid+minpoint

        self.yy=yy+minpoint
        self.masked_data=np.copy(resid)
        #print(sum(np.isnan(self.masked_data)))
        self.write_log('saved line as self.subtracted_plain')

        self.masked_data[self.mask == False]=np.nan
        self.write_log('updated self.masked_data')

        if verbose:
            F=M.figure_axis_xy(6,5, view_scale=.6)

            plt.subplot(2,2, 1)
            mmax=max(datasub.max(),-datasub.min())
            cval=np.linspace(-mmax, mmax, 21)
            cmap = brewer2mpl.get_map('Paired', 'qualitative', 6, reverse=False).mpl_colormap

            plt.contourf(time, f, datasub.T, cval ,cmap=cmap)
            plt.colorbar()
            xlabelstr=('  ( time)')
            #plt.ylabel(('|X|^2/f (' + data_unit + '^2/' + sample_unit+ ')'))
            plt.xlabel(xlabelstr)
            #plt.ylim(0.04,0.08)
            plt.grid()

            plt.subplot(2,2, 2)
            plt.plot(f, datasub.T,c='r',  alpha=.2)
            plt.plot(f,y, c='k')
            plt.plot(f, datasub.T.mean(1), c='r')
            xlabelstr=('(Freq)')
            #plt.ylabel(('|X|^2/f (' + data_unit + '^2/' + sample_unit+ ')'))
            plt.xlabel(xlabelstr)
            #plt.ylim(0.04,0.08)
            plt.grid()


            plt.subplot(2,2, 3)
            #plt.contourf(time,f,, cval, cmap=cmap)

            #plt.contourf(time,f, datasub.T - model_result_corse.reshape(time.size, f.size).T, cmap=cmap)
            #plt.contourf(time,f, resid,cval, cmap=cmap)
            plt.contourf(time,f, resid.T,cval, cmap=cmap)

            plt.colorbar()
            #plt.plot(time, fitter.params['tslope'].value*time+fitter.params['amp'])
            xlabelstr=('(Time)')
            #plt.ylabel(('|X|^2/f (' + data_unit + '^2/' + sample_unit+ ')'))
            plt.xlabel(xlabelstr)
            #plt.yticks(None)
            #plt.ylim(0.04,0.08)
            plt.grid()


            plt.subplot(2,2, 4)
            #plt.plot(f,(datasub.T - model_result_corse.reshape(time.size, f.size).T).mean(1), c='k', alpha=0.5)
            #print(np.nanmean(resid.T, 1))
            plt.plot(f, resid.T, c='b', alpha=0.2)
            plt.plot(f, np.nanmean(resid.T, 1), c='b')
            plt.plot(f, f*0, c='k')
            xlabelstr=('(Freq)')
            #plt.ylabel(('|X|^2/f (' + data_unit + '^2/' + sample_unit+ ')'))
            plt.xlabel(xlabelstr)
            #plt.ylim(0.04,0.08)
            plt.grid()
            plt.show


        return resid

    def substract_plain(self, datasub=None,  wflag='ellipse', model='least_squares', fonly=False, verbose=False):
        self.write_log('substract Plain:')
        from lmfit import minimize, Parameters,Minimizer
        import model_plain2d as minmodel

        time=self.time_dict['normalized']
        imp.reload(minmodel)
        minmodel_flag='surface'
        if minmodel_flag is 'plain':
            model_residual_func=minmodel.residual_plain
        elif minmodel_flag is 'surface':
            model_residual_func=minmodel.residual_surface

        f=self.f
        params=Parameters()

        if fonly:
            vary=False
            self.write_log('time slope prohbited')
        else:
            vary=True
            self.write_log('time slope allowed')

        if minmodel_flag is 'plain':
            params.add('amp', value= 0, min=0, max=1)
            params.add('tslope', value= 0, vary=vary)#, min=0, max=.002)
            params.add('fslope', value= 0)#, min=0., max=1)
        elif minmodel_flag is 'surface':
            params.add('amp', value= 0, vary=False)
            params.add('fslope', value= 200., min=100, max=1000.)
            params.add('fp_par', value= 0., min=0., max=1000.)

        if datasub is None:
            datasub=self.data
            self.write_log('used data')
        else:
            datasub=datasub
            self.write_log('used prescribed data')

        # Init model
        model_init=model_residual_func(params, self.time_dict['normalized'], f, data=None, eps=None)

        # reshape variables
        data1d=datasub.reshape(datasub.shape[0]*datasub.shape[1])

        # tracking Nans in 2d array
        nan_track=np.isnan(data1d)

        if wflag == 'data':
            weight=M.runningmean(data1d, 10)
            weight[np.isnan(weight)]=0
            weight[weight <0]=0
            weight1d=weight/weight.std()
            weight=weight1d.reshape(datasub.shape[0],datasub.shape[1])

        elif wflag == 'ellipse':
            weight=self.bell_curve_2d()
            if weight.shape != datasub.shape:
                weight=weight.T
            weight1d=weight.reshape(datasub.shape[0]*datasub.shape[1])
            weight1d=weight1d/weight1d.std()
        self.write_log('used weight:' +wflag)

        lower_bound_error=1e-12#error_low.reshape(datasub.shape[0]*datasub.shape[1])
        weight_sum=((weight1d.max()-weight1d)+lower_bound_error)#*self.weight_data1d+lower_bound_error*1)
        mini=Minimizer(model_residual_func, params,fcn_args=(time, f,),
                            fcn_kws={'data':datasub, 'eps':None, 'weight':weight_sum}, nan_policy='omit')

        self.write_log('used minimizel model:' +model)
        if model == 'least_squares':
            fitter = mini.minimize(method=model,
                                jac='2-point', verbose=0, ftol=1e-8, xtol=1e-8, diff_step=10)#, reduce_fcn=reduce_fct)#, fcn_args=args, fcn_kws=kws,

        if model == 'leastsq':
            fitter = mini.minimize(method=model,ftol=1e-15)#, reduce_fcn=reduce_fct)#, fcn_args=args, fcn_kws=kws,
                               #iter_cb=iter_cb, scale_covar=scale_covar, **fit_kws)
        else:
            fitter = mini.minimize(method=model)
            #self.fitter = minimize(model_residual_func, params,method=model, args=(time, self.f,), kws={'data':datasub, 'eps':None, 'weight':self.weight_sum}, nan_policy='omit')#, reduce_fcn=reduce_fct)#, fcn_args=args, fcn_kws=kws,
                               #iter_cb=iter_cb, scale_covar=scale_covar, **fit_kws)

        # generate model function
        model_result_corse=model_residual_func(fitter.params, time, f , data=None, eps=None)
        residual_2d=reshape_residuals(fitter.residual,nan_track,datasub.shape)

        #plot data
        #fitter.params.pretty_print()
        # update masked_data, data and data_full (?)

        #masked_data=residual_2d*self.mask
        masked_data=(datasub- model_result_corse.reshape(datasub.shape[0], datasub.shape[1]))
        masked_data[np.nan_to_num(masked_data)<0]=0
        masked_data=masked_data*self.mask

        self.write_log('updated masked data')
        self.write_log('saved fitter as self.pain_fitter')

        #self.data_masked_array.data=residual_2d
        self.plain_fitter=fitter
        self.plain_fitter.model_timemean= model_result_corse.reshape(datasub.shape[0], datasub.shape[1]).mean(0)
        self.write_log(self.plain_fitter.params.pretty_repr())

        if verbose:
                #from seaborn import diverging_palette as colorplate
            self.write_log('plotted fitted plain')
            F=M.figure_axis_xy(5,12, view_scale=0.4)
            plt.subplot(5,1 ,1)
            plt.plot(weight_sum, c='g', label='weight')
            plt.title('weight')

            plt.subplot(5,1 ,2)
            plt.plot(model_result_corse.reshape(datasub.shape[0]*datasub.shape[1]), c='grey', label='model')
            plt.plot(datasub.reshape(datasub.shape[0]*datasub.shape[1]), c='b', label='data', alpha=0.5)
            #plt.plot(fitter.residual, c='r', label='residual', alpha=0.5)
            plt.plot(masked_data.reshape(datasub.shape[0]*datasub.shape[1]), c='r', label='adjusted data', alpha=0.5)

            plt.legend()

            plt.subplot(5,1, 3)
            mmax=max(-data1d.min(),data1d.max())
            cval=np.linspace(-mmax, mmax, 21)
            cmap = colorplate(220, 20, n=41, as_cmap=True)

            plt.contour(time,f,model_result_corse.reshape(time.size, f.size).T, colors='black', alpha=0.5)
            plt.contourf(time, f, datasub.T ,cmap=cmap)
            plt.colorbar()
            xlabelstr=('(Time)')
            plt.xlabel(xlabelstr)
            #plt.ylim(0.04,0.08)
            plt.grid()

            plt.subplot(5,1, 4)
            plt.colorbar()
            plt.contourf(time,f, datasub.T - model_result_corse.reshape(time.size, f.size).T, cmap=cmap)
            xlabelstr=('(Time)')
            plt.xlabel(xlabelstr)
            #plt.ylim(0.04,0.08)
            plt.grid()

            plt.subplot(5,1,5)
            plt.plot(f,self.plain_fitter.model_timemean )
            plt.plot(f,np.nanmean(datasub,0 ) )
        return masked_data


    def bell_curve_2d(self,verbose=False, freq_decay=.4):
        from scipy.signal import tukey
        time=self.time_dict['normalized']


        ff, _ =np.meshgrid(tukey(self.f.size,freq_decay), time)
        ll=np.vstack((self.geo['bound_l'], self.geo['bound_r']))
        tt_move=np.logical_and(np.zeros(time.size), True)
        tt_base=np.zeros(time.size)
        dt=int(np.diff(time).mean())
        for fi in range(self.f.size):
            ll_norm=self.normalize_time_unit([ll[:,fi][0]-dt, ll[:,fi][1]+dt])
            tpos=M.cut_nparray(time,ll_norm[0], ll_norm[1])
            hann=np.hanning(tpos.sum())**(1/2.0)#**2
            tt_local=np.zeros(time.size)
            tt_local[tpos]=hann

            tt_move=np.vstack((tt_move, tt_local))
        tt_move=np.delete(tt_move, 0,0)

        if verbose:
            plt.subplot(131)
            plt.contour(time, self.f, tt_move)
            plt.subplot(132)
            plt.contour(time, self.f, ff.T)
            plt.subplot(133)
            plt.contour(time, self.f, tt_move*ff.T, colors='green')
            plt.show()
        return tt_move*ff.T

    def create_weight(self, data, wflag='ellipse',freq_decay=0.2 , verbose= False):
        """
        wflag       'data', 'ellipse' or 'combined' .Flag that determines which weigthing method is used (default='ellipse')
        """
        if 'normalized' not in self.time_dict.keys():
            self.normalize_time()




        def dataweight(self, verbose=False):
            weight=M.runningmean(self.data1d, round(1*self.data1d.size*.001) )
            weight[np.isnan(weight)]=0
            weight[weight <0]=0
            self.weight1d=weight/weight.max() #/np.sqrt(np.nansum(weight**2))
            self.weight=self.weight1d.reshape(data.shape[0],data.shape[1])

            if verbose:
                plt.contourf(self.time, self.f, self.weight.T)
                plt.show()

        def ellipseweight(self, verbose=False):
            weight=self.bell_curve_2d(verbose=verbose, freq_decay=freq_decay)
            # if weight.shape == data.shape:
            #     self.weight=weight
            #     self.weight1d=weight.reshape(data.shape[0]*data.shape[1])
            # else:
            weight=weight.T
            self.weight=weight
            self.weight1d=weight.reshape(data.shape[0]*data.shape[1])


        if wflag == 'combined':
            #print('combined weight')
            dataweight(self, verbose=verbose)
            weight_data=self.weight1d
            weight_data2d=self.weight


            ellipseweight(self, verbose=verbose)
            weight_ellipse=self.weight1d
            weight_ellipse2d=self.weight

            #plt.contourf(weight_ellipse2d.T)


            self.write_log('       (ellipse + ellipse*data)/2')
            self.write_log('       running mean factor '+str(round(1*self.data1d.size*.001))+'data points ' )
            self.weight1d=(weight_ellipse + weight_data *weight_ellipse)/2
            self.weight=(weight_ellipse2d + weight_data2d *weight_ellipse2d)/2

            if verbose:
                plt.contourf(self.time, self.f, self.weight.T)

        elif wflag == 'data':
            #print('data weight')
            dataweight(self, verbose=verbose)
            self.weight=self.weight
            self.weight1d=self.weight1d

        elif wflag == 'ellipse':
            #print('ellipse weight')
            ellipseweight(self, verbose=verbose)
            self.weight=self.weight
            self.weight1d=self.weight1d


    def fit_model(self,params,ttype='JONSWAP_gamma', datasub=None , weight_opt=None, model='least_squares',
                  error_estimate=None, error_N=None,error_workers=None, error_nwalkers=None,  prior=None, set_initial=True,
                  error_opt=None):

        """
        This fits a model to the given data and returns residual, model and several measures of fit

        Inputs:
        ---------------------------
        params      parameters of the model of class Parameters (lmfit module)
        ttype       flag for the model that should be used for optomatzation. The the curent state only the default works
        datasub     the data that is used for fitting.It can be masked data (np.nan)
        weight_opt  determines weight model. If non set to some standard values
        model       argument that is passed to lmfit.Minimizer.
        error_estimate  None or list of parameter names for which an error estimate is performed. If None no error is estimated.
        error_N     number of iterations for monte-carlo-chain error estimate. (Default=500)
        prior       Dictionary of Priors (estiamted uncertainties of parameters). params within prior will added to the model costfunction (Jm)
        error_opt   Dictionary of additional options for mcmc workers

        Outputs:
        ---------------------------
        assigns a lot of properties the instance of the storm class.
        (a non complete list)
        self.params     Parameters that where used for initialized model

        self.weight     calcualted weight function without floor
        self.weight_sum 1d weight function with floor value

        self.nan_track  boolean of Nans in the data

        self.fitter                 Output from Minimizer(). Is an MinimizerResults() class
        self.fitter.params          fitted parameters
        self.fitter_error           output from error estimation if flag is set.

        self.time_syntetic          higher resulution nondimentional time axis for calculating model
        self.model_result           optimal resulting model 1D with high resolution time axis
        self.model_result_corse     optimal (fitted) model 1D with same timeaxis as the data.

        self.fitter.residual        residual including the weight function in 1d. Has the length of valid datapoint in data_sub (N-#of np.nans)
                                    Does not include datapoint from regularization
        self.residual_2d            residual including the weight function in 2d. I.e. the last run of the fitting function (set by ttype)
                                    has np.nan at the same place as datasub, infered from self.nan_track
                                    Does not include datapoint from regularization

        self.fit_dict               Output of create_fit_dict() method

        """
        from lmfit import minimize, Parameters,Minimizer
        f=self.f
        self.params=params
        if datasub is None:
            datasub=self.masked_data
            self.write_log('used masked_data')
        else:
            datasub=datasub
            self.write_log('used prescibed data')


        self.write_log('used residual model: '+ttype)
        if ttype == 'JONSWAP_gamma':
            import models.JONSWAP_gamma as minmodel

            time=self.time_dict['normalized']
            if prior is not None:
                self.prior = prior
            else:
                self.prior = None
            minimizer_fcn_kws = {'data':datasub, 'eps':None, 'prior':self.prior}
            model_residual_func=minmodel.cost
            self.model_residual_func= model_residual_func
            self.minmodel           = minmodel

        elif ttype == 'JONSWAP_gamma_regularization':
            import model_gamma_JONSWAP as minmodel
            time=self.time_dict['normalized']
            if prior is not None:
                self.prior=prior
                minimizer_fcn_kws={'data':datasub, 'eps':None, 'prior':self.prior}
                model_residual_func=minmodel.residual_JANSWAP_gamma_regularization
            else:
                raise Warning('JONSWAP_gamma_regularization Model requires a prior dictionary (set option as prior=prior_dict)' )
            self.model_residual_func=model_residual_func
            self.minmodel           = minmodel

        elif ttype == 'JONSWAP_gamma_regularization_acc':
            import model_gamma_JONSWAP as minmodel
            time=self.time_dict['normalized']
            if prior is not None:
                self.prior=prior
                minimizer_fcn_kws={'data':datasub, 'eps':None, 'prior':self.prior}
                model_residual_func=minmodel.residual_JANSWAP_gamma_regularization_acc
            else:
                raise Warning('JONSWAP_gamma_regularization Model requires a prior dictionary (set option as prior=prior_dict)' )
            self.model_residual_func=model_residual_func
            self.minmodel           = minmodel

        elif ttype == 'gaussian_gamma':
            import models.gaussian_gamma as minmodel
            time=self.time_dict['normalized']
            if prior is not None:
                self.prior = prior
            else:
                self.prior = None
            minimizer_fcn_kws = {'data':datasub, 'eps':None, 'prior':self.prior}
            model_residual_func=minmodel.cost
            self.model_residual_func= model_residual_func
            self.minmodel           = minmodel
        else:
            raise Exception("model type is not correct defined")

        # initialize model
        self.model_init=model_residual_func(params, self.time_dict['normalized'],f , data=None, eps=None)

        #print()
        # reshape variables
        self.data1d=datasub.reshape(datasub.shape[0]*datasub.shape[1])
        # tracking Nans in 2d array
        self.nan_track=np.isnan(self.data1d)



        if weight_opt is not None:
            self.write_log('used weight model opt: \n' + str(weight_opt))

            if type(weight_opt) is not dict:
                raise ValueError('weight_opt is not a dict')
            elif 'type' not in weight_opt:
                weight_opt['type']='combined'
                self.write_log('set type to standard: ' + str(weight_opt['type']))

            elif 'freq_decay' not in weight_opt:
                weight_opt['freq_decay']=0.1
                self.write_log('set freq_decay to standard: ' + str(weight_opt['freq_decay']))

            elif 'lower_bound_error' not in weight_opt:
                weight_opt['lower_bound_error']=1e-6
                self.write_log('set lower_bound_error to standard: ' + str(weight_opt['lower_bound_error']))

            self.write_log('used residual model: '+ttype)
        else:
            weight_opt['type']='combined'
            weight_opt['freq_decay']=0.1
            weight_opt['lower_bound_error']=1e-6


        self.create_weight( datasub, wflag=weight_opt['type'] , freq_decay=weight_opt['freq_decay'], verbose=False)# creates self.weight and self.weight1d

        self.weight_sum=(1*self.weight1d+weight_opt['lower_bound_error'])#*self.weight_data1d+lower_bound_error*1)
        self.weight_sum2d=(1*self.weight+weight_opt['lower_bound_error'])#*self.weight_data1d+lower_bound_error*1)

        #minimizer_fcn_kws['weight']=self.weight_sum
        minimizer_fcn_kws['weight']=self.weight_sum2d

        #self.weight_sum=self.weight_sum/self.weight_sum.sum()
        #print(self.weight_sum.shape)
        #print(self.weight_data1d)
        #plt.plot(self.weight_data1d)

        # Fit model
        #print(datasub.shape, Rinv.shape, time.shape, self.f.shape) #least_squares lbfgsb tnc  'propagate''  differential_evolution
        # print(model_residual_func )
        # params.pretty_print()
        # print(time.shape, self.f.shape)
        # minimizer_fcn_kws['weight']=None
        # for k,I in minimizer_fcn_kws.iteritems():
        #     if I is np.array:
        #         print(k,I.shape)
        #     else:
        #         print(k,I, type(I))
        #
        # print('---')

        mini=Minimizer(model_residual_func, params,fcn_args=(time, self.f,),
                            fcn_kws=minimizer_fcn_kws, nan_policy='omit')

        self.write_log('used minimizer model: '+model)
        if model == 'least_squares':
            self.fitter = mini.minimize(method=model,
                                jac='3-point', verbose=1, ftol=1e-15, xtol=1e-14)# , diff_step=1)#, reduce_fcn=reduce_fct)#, fcn_args=args, fcn_kws=kws,
            #mini.params['slope']
                               #iter_cb=iter_cb, scale_covar=scale_covar, **fit_kws)
            #self.fitter = minimize(model_residual_func, params,method=model, args=(time, self.f,),
            #                    kws={'data':datasub, 'eps':None, 'weight':self.weight_sum}, nan_policy='omit',
            #                    jac='3-point', verbose=1, ftol=1e-15, xtol=1e-12, diff_step=10)#, reduce_fcn=reduce_fct)#, fcn_args=args, fcn_kws=kws,
            #                   #iter_cb=iter_cb, scale_covar=scale_covar, **fit_kws)
        if model == 'leastsq':
            self.fitter = mini.minimize(method=model,ftol=1e-15)#, reduce_fcn=reduce_fct)#, fcn_args=args, fcn_kws=kws,
                               #iter_cb=iter_cb, scale_covar=scale_covar, **fit_kws)
            #self.fitter = minimize(model_residual_func, params,method=model, args=(time, self.f,),
            #                    ftol=1e-15)#, reduce_fcn=reduce_fct)#, fcn_args=args, fcn_kws=kws,
            #                   #iter_cb=iter_cb, scale_covar=scale_covar, **fit_kws)

        else:
            self.fitter = mini.minimize(method=model)
            #self.fitter = minimize(model_residual_func, params,method=model, args=(time, self.f,), kws={'data':datasub, 'eps':None, 'weight':self.weight_sum}, nan_policy='omit')#, reduce_fcn=reduce_fct)#, fcn_args=args, fcn_kws=kws,
                               #iter_cb=iter_cb, scale_covar=scale_covar, **fit_kws)

        if error_estimate:
            error_N=500 if error_N is None else error_N
            error_workers=1 if error_workers is None else error_workers
            if error_estimate == 'all':
                error_estimate=self.params.valuesdict().keys()
            self.fitter_error=self.error_estimator(mini, error_estimate, workers=error_workers,  steps=error_N, nwalkers=error_nwalkers,
                                                    set_initial=set_initial, others=error_opt)

            # generate model function
            self.time_syntetic=np.arange(0,1, .01)
            self.model_result=model_residual_func(self.fitter_error.params, self.time_syntetic, self.f , data=None, eps=None)
            self.model_result_corse=model_residual_func(self.fitter_error.params, time, self.f , data=None, eps=None)

            self.residual_2d=reshape_residuals(self.fitter.residual,self.nan_track,datasub.shape)

        else:
            #self.mini=mini # just for testing
            self.fitter_error=False
            # generate model function
            self.time_syntetic=np.arange(0,1, .01)
            self.model_result=model_residual_func(self.fitter.params, self.time_syntetic, self.f , data=None, eps=None)
            self.model_result_corse=model_residual_func(self.fitter.params, time, self.f , data=None, eps=None)

            self.residual_2d=reshape_residuals(self.fitter.residual,self.nan_track,datasub.shape)
        #plot data
        #F.save_light(path=plotpath, name='gamma_JANSWA_result_all')

        #self.fitter.params.pretty_print()
        self.create_fit_dict()

    def error_estimator(self, mini, error_estimate, workers=1, steps=500 , nwalkers=None, set_initial=True, others=None):
        from time import clock
        start = clock()

        nwalkers=len(error_estimate)*2 +2 if nwalkers is None else nwalkers
        nwalkers=max(len(error_estimate)*2 +2, nwalkers)
        self.write_log('--- error_estimates:')
        self.write_log('N=' +str(steps)+ ' vars:'+ str(error_estimate) )
        self.write_log(' workers= ' + str(workers) + ', parallel version is in beta '  )
        self.write_log(' nwalkers=' + str(nwalkers) )

        if (set_initial is True) | (set_initial == 'Gauss' ):
            if (others is not None) & (others['ntemps'] > 1 ):
                init_pos=np.zeros( (nwalkers, others['ntemps']) ).T
                twod_flag=True
            else:
                init_pos=np.zeros( (nwalkers ) )
                twod_flag=False

            print('init_pos shape' + str(init_pos.shape) )

            #for k,p in self.prior.iteritems():
            for k,p in self.fitter.params.iteritems():

                #print(np.random.normal(0.0, p, error_nwalkers).shape )
                best_guess_mean=p.value

                if k in self.prior:
                    prior_std=self.prior[k]['m_err']
                else:
                    prior_std= (p.value-p.value*0.8)

                if type(prior_std) is list:
                    prior_std=(prior_std[1]-prior_std[0])/2.0

                #print(p['m0'], std_guess)
                if (set_initial == 'Gauss'):
                    # Use normal distribution
                    if (others is not None) & (others['ntemps'] > 1 ):
                        randnumber= np.random.normal(best_guess_mean, prior_std, nwalkers*others['ntemps']).reshape(nwalkers, others['ntemps'])
                    else:
                        randnumber= np.random.normal(best_guess_mean, prior_std, nwalkers)

                else:
                    # use uniform distribution
                    if (others is not None) & (others['ntemps'] > 1 ):
                        randnumber= np.random.uniform(  best_guess_mean-prior_std*1,
                                                       best_guess_mean+prior_std*1, nwalkers*others['ntemps']).reshape(nwalkers, others['ntemps'])
                    else:
                        randnumber= np.random.uniform(  best_guess_mean-prior_std*1,
                                                       best_guess_mean+prior_std*1, nwalkers)

                #print(init_pos.shape)
                if twod_flag:
                    init_pos=np.dstack( (init_pos, randnumber.T) )
                else:
                    init_pos=np.vstack( (init_pos, randnumber.T) )


            if (others is not None) & (others['ntemps'] > 1 ):
                init_pos=init_pos[:,:, 1:]
            else:
                init_pos=init_pos[1:,:].T
            print(init_pos.shape)
            seed=None

        elif callable(set_initial):
            seed=set_initial
            init_pos=None

        else:
            seed=None
            init_pos=None

        print('seed is '+ str(seed) )
        params=copy.deepcopy(self.fitter.params)
        for key in params.iterkeys():
            if key in error_estimate:
                params[key].set(vary=True)
            else:
                params[key].set(vary=False)
        burn=min(100, int(steps/4.0))
        #m= mini.emcee(params=params, steps=steps, burn=burn, nwalkers=nwalkers, workers=workers)#, is_weighted=True)

        if others is None:
            others={'ntemps':1}
        print('others is  '+ str(others) )
        #print('init_pos shape' + str(init_pos.shape) )
        #print(init_pos )
        m= mini.emcee(params=params, steps=steps, burn=burn, nwalkers=nwalkers, workers=workers, pos=init_pos,seed=seed,  **others)#, is_weighted=True)

        self.write_log('error_estimate time: ' + str(round(clock()-start,2) ) +'seconds')
        self.write_log(' eval/sec=' + str(round( nwalkers*(steps-burn)/ (clock()-start),2) ) )
        self.write_log('---')
        return m

    #def create_rt0_propabilities(self, params_dict):


    def simple_fitstats(self, verbose=False):
        """
        Create fitting statistics
        """
        Jm_regulizer = self.minmodel.Jm_regulizer

        a                   = abs(M.nannormalize(self.fitter.residual))
        max3                = list(a[a.argsort()[::-1][0:3]])

        J_D_sqr             = np.sum( self.fitter.residual**2 ) # does only include data-model
        if hasattr(self, 'prior') and (self.prior is not None):
            J_M_sqr           = sum( [i**2 for i in Jm_regulizer(self.fitter.params.valuesdict(), self.prior)] ) # does only include data-model
            error_mean_model  = J_M_sqr / float(len(self.prior))
        else:
            J_M_sqr           = 0.0
            error_mean_model  = 0.0

        chisqr              = J_D_sqr + J_M_sqr
        normalized_chisqr   = chisqr   /   ( np.nanstd( self.fitter.residual ) **2  *  self.fitter.ndata )

        error_frac          = np.sum( self.fitter.residual**2 ) /  np.sum( ( ( self.data1d * self.weight_sum )[~self.nan_track]  )**2  )
        error_frac_data     = J_D_sqr                           /  np.sum( ( ( self.data1d * self.weight_sum )[~self.nan_track]  )**2  )


        if verbose:
            F=M.figure_axis_xy(5, 4)
            residual_hist=plt.hist(M.nannormalize(self.fitter.residual), np.arange(-5,5,.1), normed=True)
            plt.ylim(0, 1)
            print('maximum residual: ' +str(max3)+ '; normalized chi^2: '+str(normalized_chisqr))

        return max3, normalized_chisqr, error_frac , error_frac_data , error_mean_model , chisqr

    def create_fit_dict(self):
        fit_dict=dict()
        from lmfit import Parameters



        for key,item in self.fitter.__dict__.iteritems(): ## should be self.fitter.params.iteritem. this may grap the initial value!
            #print( '--')
            if (type(item) is list) or (type(item) is dict) or (type(item) is np.ndarray):
                #print('Not')
                pass
            elif type(item) is Parameters:
                #print('dont save parameter type')
                pass
            else:
                #print(key , type(item))#,  item )
                fit_dict[key]=item

        parameter_list=list()
        if self.fitter_error:
            for key,item in self.fitter_error.params.iteritems():
                #print( '--')
                parameter_list.append(key)
                if (type(item) is list) or (type(item) is dict) or (type(item) is np.ndarray):
                    #print('Not')
                    pass
                else:
                    #print(key , type(item))#,  item )
                    fit_dict[key]={'value':item.value, 'stderr':item.stderr, 'correl':item.correl}
        fit_dict['ID']=self.ID
        fit_dict['params_list']=parameter_list

        #same data std "scaler" in fit_dict
        fit_dict['factor']=self.factor
        fit_dict['max3'], fit_dict['normalized_chisqr'], fit_dict['error_frac'], fit_dict['error_frac_data'], fit_dict['error_mean_model'], fit_dict['chisqr_man'],  =self.simple_fitstats()
        if 'errorbars' in fit_dict:
            fit_dict['errorbars']= bool(fit_dict['errorbars'])
        self.fit_dict=fit_dict

    def plot_fitted_model(self, flim=(0.04,0.08), datasub=None, data_unit=None):
        import brewer2mpl
        import string

        if hasattr(self, 'fitter_error') and (self.fitter_error is not False):
            fitter=self.fitter_error
        else:
            fitter=self.fitter

        fn=iter([i+')' for i in list(string.ascii_lowercase)])
        #flim=(0.04,0.08)

        if datasub is None:
            datasub=self.masked_data
            datasub_less_noise=self.masked_data_less_noise
        else:
            datasub=datasub
            datasub_less_noise=datasub

        #if ~hasattr(self, 'model_init'):
        #    self.model_init=self.model_residual_func(self.params, self.time_dict['normalized'],self.f , data=None, eps=None)

        time=self.time_dict['normalized']
        f=self.f

        mmax=5e-6#datasub.max()
        cval=np.copy(self.clevs[:])#np.linspace(-mmax, mmax, 41)

        sample_unit='s'
        data_unit='1'
        datalabel='normalized Energy (' + data_unit + '^2/' + sample_unit+ ')'
        xlabelstr=('(Time)')
        cmap = brewer2mpl.get_map('Spectral', 'diverging', 6, reverse=True).mpl_colormap
        cmap.set_bad('white')

        fig=M.figure_axis_xy(8, 12, fig_scale=1, container=True, view_scale=.5)
        plt.suptitle(self.ID +' | '+str(self.time_dict['dt64'][int(self.time_dict['dt64'].size/2)].astype('M8[h]')) +
                        '\n redchi=' +str(round(self.fit_dict['redchi'],3))
                        +', normalized chisqrt=' +str(round(self.fit_dict['normalized_chisqr'],3))
                        +', maxres='+str(round(self.fit_dict['max3'][0],3)) , y=1.02)

        # Initalized model and data 2D
        S1 = plt.subplot2grid((5,2), (0, 0),rowspan=2,facecolor='w', colspan=1 )
        #S1=M.subplot_routines(S1)

        plt.contourf(time,self.f,datasub.T, cval, cmap=cmap)
        #plt.colorbar()
        plt.contour(time, f, self.model_init.reshape(time.size, f.size).T, colors='black')
        self.plot_line_params(time,  self.params['slope'].value, self.params['intersect'].value, c='r', alpha=0.7)
        #plt.plot(time,self.params['slope'].value*time+self.params['intersect'], c='r', alpha=0.7)
        plt.contour(time,f,self.weight.T, np.linspace(0, 1,5), colors='orange', alpha=1, linewidths=2)

        plt.ylabel(('f (Hz)'))
        #plt.xlabel(xlabelstr)
        plt.ylim(flim)
        #plt.grid()
        plt.title(fn.next()+' '+'Initalized Model and data 2D', loc='left', y=1.02)


        # Model and Data in 2D
        S2 = plt.subplot2grid((5,2), (0, 1),rowspan=2, colspan=1 )
        cmap = brewer2mpl.get_map('Spectral', 'diverging', 6, reverse=True).mpl_colormap
        plt.contourf(time,f,datasub_less_noise.T, cval, cmap=cmap)
        cb=plt.colorbar(label=datalabel)
        cb.outline.set_visible(False)
        #cb.label()

        self.plot_line_params(time, self.fitter.params['slope'].value, self.fitter.params['intersect'], c='white', alpha=0.5)

        plt.contour(self.time_syntetic,f,self.model_result.reshape(self.time_syntetic.size, f.size).T, colors='black', alpha=0.5)
        self.plot_line_params(time, fitter.params['slope'].value, fitter.params['intersect'], c='r')
        self.plot_line_params(time, fitter.params['slope'].value , self.cal_intersect_adjust(fitter.params), c='r', alpha=0.7)
        #

        #plt.ylabel(('f (Hz)'))
        #plt.xlabel(xlabelstr)
        plt.ylim(flim)
        #plt.grid()
        plt.title(fn.next()+' '+'Model and Data in 2D', loc='left', y=1.02)


        # Init data in 1D
        S3 = plt.subplot2grid((5,2), (2, 0),rowspan=1,facecolor='w', colspan=1 )
        plt.plot(self.model_init, c='k',alpha=0.4, label='model')
        plt.plot(datasub.reshape(datasub.shape[0]*datasub.shape[1]), c='b',  alpha=0.5, label='data')
        plt.title(fn.next()+' '+'Initial Model and Data in 1D', loc='left', y=1.06)

        plt.legend()


        # Model and Data in 1D
        S4 = plt.subplot2grid((5,2), (2, 1),rowspan=1,facecolor='w', colspan=1 )
        plt.plot(self.model_result_corse, c='k', label='model',  alpha=0.4,)
        plt.plot(datasub.reshape(datasub.shape[0]*datasub.shape[1]), c='b', alpha=0.5, label='data')
        #plt.plot(abs(model_result_corse/datasub.reshape(datasub.shape[0]*datasub.shape[1])), c='b', alpha=0.5, label='model/data')
        plt.legend()
        #plt.ylim(0, 100)
        plt.title(fn.next()+' '+'Fitted Model and Data in 1D', loc='left', y=1.06)

        # weight
        S5 = plt.subplot2grid((5,2), (3, 0),rowspan=1,facecolor='w', colspan=1 )


        plt.plot(self.weight_sum, alpha=0.8, c='orange', label='weight')
        plt.plot(datasub.reshape(datasub.shape[0]*datasub.shape[1]), c='b', alpha=0.5, label='data')
        #plt.plot(self.weight_data1d, c='r', label='data weight')
        plt.legend()
        plt.ylim(0, np.max(self.weight_sum)*2)
        plt.title(fn.next()+' '+'Weight in 1D', loc='left', y=1.02)
        #plt.grid()


        # 2d residual

        S7 = plt.subplot2grid((5,2), (3, 1),rowspan=2,facecolor='w', colspan=1 )
        cmap = brewer2mpl.get_map('RdBu', 'diverging', 10, reverse=True).mpl_colormap
        cval=np.linspace(-self.clevs[-1], self.clevs[-1], 20)
        #plt.contourf(time,f,self.residual_2d.T, cval, cmap=cmap)
        plt.contourf(time,f,(datasub-self.model_result_corse.reshape(datasub.shape[0],datasub.shape[1])).T, cval, cmap=cmap)
        cbar=plt.colorbar()
        cbar.outline.set_visible(False)
        #plt.contour(time,f,, colors='w')
        plt.plot(time, fitter.params['slope'].value*time+ fitter.params['intersect'], c='r')
        xlabelstr=('Normalized Time')
        plt.xlabel(xlabelstr)
        plt.ylim(flim)

        #plt.grid()
        plt.title(fn.next()+' '+'Residual in 2D without weigthing', loc='left', y=1.02)



        # time mean residual
        S6 = plt.subplot2grid((5,2), (4, 0),rowspan=1,facecolor='w', colspan=1 )

        model_result_tmean=np.copy(self.model_result_corse)
        model_result_tmean[self.nan_track]=np.nan
        model_result_tmean=np.nanmean(model_result_tmean.reshape(self.time.size, f.size).T, 1)
        #model_tmean=np.nanmean(self.model_result.reshape(self.time_syntetic.size, f.size).T,1)

        #resid2=(datasub-self.model_result_corse.reshape(datasub.shape[0],datasub.shape[1])).T#self.weight_sum.reshape(time.size, f.size).T
        #resid2_tmean=np.nanmean(resid2, 1)
        #weight_tmean=np.nanmean(self.weight_sum.reshape(time.size, f.size).T, 1)

        #model_tmean=model_tmean/weight_tmean
        residual_tmean=np.nanmean(self.residual_2d.T,1)

        data_tmean=np.nanmean(datasub.T, 1)
        #plt.plot(f,  weight_tmean, label='diff')

        plt.plot(f,model_result_tmean , label='model', c='k')

        #dd_mask=self.model_result_corse < np.percentile(self.model_result_corse,75)
        #dd_mask_2d=dd_mask.reshape(self.time.size, f.size).T
        #dd_2d=self.model_result_corse.reshape(self.time.size, f.size).T
        #dd_2d[dd_mask_2d]=np.nan
        #plt.plot(f,np.nanmean( dd_2d, 1) , label='model', c='k')

        #plt.plot(f,model_tmean_corse , label='model corse')
        #plt.plot(f,residual_tmean, label='residual func')
        plt.plot(f,data_tmean, label='data' , c='b', alpha=0.5)


        _, _, _, masked_data2=self.get_max_data()
        plt.plot(f,    np.nanmean(  (masked_data2* self.factor).T , 1), label='data under Curve' , c='b')

        plt.plot(f,-(data_tmean-model_result_tmean), label='residual')

        plt.plot(f,residual_tmean, label='weighted residual' )
        if hasattr(self, 'plain_fitter'):
            if hasattr(self.plain_fitter, 'model_timemean'):
                plt.plot(f,self.plain_fitter.model_timemean, label='subtracted surface' )
        #plt.plot(f,-resid2_tmean, label='residual2' )
        plt.legend()
        plt.xlim(flim)
        pcut=M.cut_nparray(f, flim[0], flim[1])
        lylim=np.max([model_result_tmean[pcut].min() ,model_result_tmean[pcut].max()])*1.3
        plt.ylim(-lylim ,lylim)

        plt.title(fn.next()+' '+'Time mean Residual', loc='left', y=1.06)
        plt.xlabel(('f (Hz)'))
        plt.plot(f,data_tmean*0, c='grey' )

        plt.subplots_adjust(left=None, bottom=None, right=None, top=.9,
                        wspace= 0.25, hspace=.5)

        return fig

    def plot_fitsimple(self, flim=None, datasub=None, k=False):
        import brewer2mpl
        #flim=(0.04,0.08)

        if hasattr(self, 'fitter_error'):
            fitter=self.fitter_error
        else:
            fitter=self.fitter


        if datasub is None:
            datasub=self.masked_data
        else:
            datasub=datasub

        time=self.time_dict['dt64']
        f=self.f
        flim= (self.geo['f_low'], self.geo['f_high']) if flim is None else flim

        mmax=datasub.max()
        cval=self.clevs#np.linspace(0, mmax, 31)
        sample_unit='s'
        data_unit='m'
        #datalabel='Energy (' + data_unit + '^2/' + sample_unit+ ')'
        datalabel='Energy (normalized)'

        xlabelstr=('(Time)')
        cmap = brewer2mpl.get_map('Paired', 'qualitative', 6, reverse=False).mpl_colormap
        fig=M.figure_axis_xy(7, 4, fig_scale=1, container=True, view_scale=.5)
        plt.suptitle(self.ID +' | '+str(self.time_dict['dt64'][int(self.time_dict['dt64'].size/2)].astype('M8[h]')) +
                        '\n redchi=' +str(round(self.fit_dict['redchi'],3))
                        +', normalized chisqrt=' +str(round(self.fit_dict['normalized_chisqr'],3))
                        +', maxres='+str(round(self.fit_dict['max3'][0],3)), y=1.06)

        # Model and Data in 2D
        S2 = plt.subplot2grid((2,3), (0, 0),rowspan=2,facecolor='white', colspan=2 )
        plt.contourf(time,f,datasub.T, cval, cmap=cmap)
        cb=plt.colorbar(label=datalabel)
        #cb.label()
        plt.contour(self.time_dict['dt64'],f,self.model_result_corse.reshape(self.time.size, f.size).T, colors='black', alpha=0.5)

        self.plot_line_params_realtime(c='r')
        #plt.plot(time,(self.time_dict['normalized']-self.SM_dict_pandas['t0']['normalized'])/self.SM_dict_pandas['dTdf']['normalized'], c='r')

        #plt.plot(time, self.fitter.params['slope'].value*time+self.fitter.params['intersect'], c='r')

        ax=plt.gca()
        ax.xaxis_date()
        Month = dates.MonthLocator()
        Day = dates.DayLocator(interval=5)#bymonthday=range(1,32)
        Hour = dates.HourLocator(interval=12)#bymonthday=range(1,32)

        dfmt = dates.DateFormatter('%y-%b-%dT%H:%M')


        ax.xaxis.set_major_locator(Day)
        ax.xaxis.set_major_formatter(dfmt)
        ax.xaxis.set_minor_locator(Hour)

        # Set both ticks to be outside
        ax.tick_params(which = 'both', direction = 'out')
        ax.tick_params('both', length=6, width=1, which='major')
        ax.tick_params('both', length=3, width=1, which='minor')


        plt.ylabel(('f (Hz)'))
        #plt.xlabel(xlabelstr)
        plt.ylim(flim)
        plt.grid()
        plt.title('Model and Data in 2D', loc='left', y=1)

        # time mean residual
        S6 = plt.subplot2grid((2,3), (0, 2),rowspan=2,facecolor='w', colspan=1 )


        model_result_tmean=np.copy(self.model_result_corse)
        model_result_tmean[self.nan_track]=np.nan
        model_result_tmean=np.nanmean(model_result_tmean.reshape(self.time.size, f.size).T, 1)
        #model_tmean=np.nanmean(self.model_result.reshape(self.time_syntetic.size, f.size).T,1)

        #resid2=(datasub-self.model_result_corse.reshape(datasub.shape[0],datasub.shape[1])).T#self.weight_sum.reshape(time.size, f.size).T
        #resid2_tmean=np.nanmean(resid2, 1)
        #weight_tmean=np.nanmean(self.weight_sum.reshape(time.size, f.size).T, 1)

        #model_tmean=model_tmean/weight_tmean
        residual_tmean=np.nanmean(self.residual_2d.T,1)

        data_tmean=np.nanmean(datasub.T, 1)
        #plt.plot(f,  weight_tmean, label='diff')

        plt.plot(model_result_tmean,f , label='model', c='k')
        #plt.plot(f,model_tmean_corse , label='model corse')
        #plt.plot(f,residual_tmean, label='residual func')
        plt.plot(data_tmean,f, label='data' , c='b')
        plt.plot(-(data_tmean-model_result_tmean),f ,label='residual')

        plt.plot(residual_tmean,f, label='(model) weighted residual' )
        #plt.plot(f,-resid2_tmean, label='residual2' )
        plt.legend()
        plt.xlim(flim)
        pcut=M.cut_nparray(f, flim[0], flim[1])
        lylim=np.max([np.nanmin(data_tmean[pcut]) ,np.nanmax(data_tmean[pcut])])
        lymin=-lylim*.2
        plt.xlim(lymin ,lylim)

        plt.title('Time mean Residual', loc='left', y=1)
        S6.set_yticks([])
        plt.xlabel(('Amplitude'))
        #plt.grid()
        plt.plot(data_tmean*0,f, c='grey' )

        plt.subplots_adjust(left=None, bottom=None, right=None, top=.9,
                        wspace= 0.25, hspace=.5)

        return fig

    def plot_fit_hist(self, hist_dict_org,flim=None, datasub=None, k=False):

        hist_dict=hist_dict_org.copy() # make sure that T.hist_dict is not tought
        # try to delete t_scale from dict
        try:
            del hist_dict['tscale']
            print('ignore tscale ')
        except:
            pass

        if hasattr(self, 'fitter_error'):
            fitter=self.fitter_error
        else:
            fitter=self.fitter

        totalrows=8
        totalcollums=5
        subrow=totalrows-3
        nhists=np.size(hist_dict.keys())
        nhists_half=int(np.ceil((nhists)/2.))
        import brewer2mpl
        #flim=(0.04,0.08)

        if datasub is None:
            datasub=self.masked_data
        else:
            datasub=datasub

        time=self.time_dict['dt64']
        f=self.f
        flim= (self.geo['f_low'], self.geo['f_high']) if flim is None else flim

        mmax=datasub.max()
        cval=self.clevs#np.linspace(0, mmax, 31)
        sample_unit='s'
        data_unit='m'
        datalabel='Energy (' + data_unit + '^2/' + sample_unit+ ')'
        xlabelstr=('(Time)')
        fig=M.figure_axis_xy(9, 8, fig_scale=1, container=True, view_scale=.5)

        if hasattr(self, 'fitter'):
            plt.suptitle(self.ID +' | '+str(self.time_dict['dt64'][int(self.time_dict['dt64'].size/2)].astype('M8[h]')) + '\n'
                            +'Normalized chisqrt=' +str(round(self.fit_dict['normalized_chisqr'],3))
                            +', Fractional Error=' +str(round(self.fit_dict['error_frac'],3))
                            +', Fractional Data Model ='+str(round(self.fit_dict['error_mean_model'],3)), y=1.06)
        else:
            plt.suptitle(self.ID +' | '+str(self.time_dict['dt64'][int(self.time_dict['dt64'].size/2)].astype('M8[h]')) +
                            '\n redchi=' +'undefined'
                            +', normalized chisqrt=' +str(round(self.fit_dict['normalized_chisqr'],3))
                            +', maxres='+str(round(self.fit_dict['max3'][0],3)), y=1.06)


        # Model and Data in 2D
        S2 = plt.subplot2grid((totalrows,totalcollums), (0, 0),rowspan=subrow,facecolor='white', colspan=2)
        cmap = brewer2mpl.get_map('Paired', 'qualitative', 6, reverse=False).mpl_colormap


        ax2=plt.contourf(time,f,datasub.T, cval, cmap=cmap)


        cb=plt.colorbar(label=datalabel, fraction=0.046, pad=0.09, orientation="horizontal")
        cb.outline.set_visible(False)

        ticklabels,ticks_rounded=MT.tick_formatter(cval, interval=2, rounder=0, expt_flag=True)
        cb.ax.set_yticklabels(ticks_rounded)

        #cb.label()
        plt.contour(self.time_dict['dt64'],f,self.model_result_corse.reshape(self.time.size, f.size).T, colors='black', alpha=0.5)
        self.plot_line_params_realtime(c='r')

        ax=plt.gca()
        ax.xaxis_date()
        Month = dates.MonthLocator()
        Day14 = dates.DayLocator(interval=5)
        Day = dates.DayLocator(interval=1)#bymonthday=range(1,32)
        Hour = dates.HourLocator(interval=6)#bymonthday=range(1,32)

        dfmt = dates.DateFormatter('%y-%b-%dT%H:%M')


        ax.xaxis.set_major_locator(Day14)
        ax.xaxis.set_major_formatter(dfmt)
        ax.xaxis.set_minor_locator(Day)


        plt.ylabel(('f (Hz)'))
        #plt.xlabel(xlabelstr)
        plt.ylim(flim)
        #plt.grid()
        plt.title('Model and Data in 2D', loc='left', y=1)

        # time mean residual
        S6 = plt.subplot2grid((totalrows,totalcollums), (0, 2),rowspan=subrow,facecolor='w', colspan=2 )


        model_result_tmean=np.copy(self.model_result_corse)
        model_result_tmean[self.nan_track]=np.nan
        model_result_tmean=np.nanmean(model_result_tmean.reshape(self.time.size, f.size).T, 1)


        residual_tmean=np.nanmean(self.residual_2d.T,1)

        data_tmean=np.nanmean(datasub.T, 1)
        #plt.plot(f,  weight_tmean, label='diff')

        plt.plot(model_result_tmean,f , label='model', c='k')
        #plt.plot(f,model_tmean_corse , label='model corse')
        #plt.plot(f,residual_tmean, label='residual func')
        plt.plot(data_tmean,f, label='data' , c='b', alpha=0.7)
        plt.plot(-(data_tmean-model_result_tmean),f ,c='g', alpha=.4, label='residual')

        plt.plot(residual_tmean,f, c='g', label='(model) weighted residual' )

        if hasattr(self, 'plain_fitter'):
            if hasattr(self.plain_fitter, 'model_timemean'):
                plt.plot(self.plain_fitter.model_timemean,f, label='substracted surface', c='orange', alpha=.6 )
                plt.plot(data_tmean - self.plain_fitter.model_timemean,f, label='adjusted data', c='b')

        plt.legend(frameon=False, loc=1, #bbox_to_anchor=(0., .8, 1., .8)
                   ncol=1, mode="expand", borderaxespad=0.)

        plt.xlim(flim)
        pcut=M.cut_nparray(f, flim[0], flim[1])
        lylim=np.max([data_tmean[pcut].min() ,data_tmean[pcut].max()])*1.2
        lymin=-lylim*.4
        plt.xlim(lymin ,lylim)

        plt.title('Time mean Residual', loc='left', y=1)
        #S6.set_yticks([])
        plt.xlabel(('Amplitude'))
        #plt.grid()
        plt.plot(data_tmean*0,f, c='grey' )

        plt.subplots_adjust(left=None, bottom=None, right=None, top=.9,
                        wspace= 0.25, hspace=.5)



        Nk=5
        hist_pos_right=0

        keys_right=['error_frac', 'chisqr_man', 'normalized_chisqr', 'error_frac_data', 'error_mean_model']

        for k in list(set(hist_dict.keys()) & set(keys_right)):
            S7 = plt.subplot2grid((totalrows,totalcollums), (hist_pos_right, Nk-1 ),rowspan=1,facecolor='w', colspan=1)

            AX=plt.plot(hist_dict[k][1], np.append(hist_dict[k][0],0), drawstyle='steps-post')
            #S7.update( right=0.05)
            if hasattr(self, 'fitter') & fitter.params.has_key(k):
                plt.plot([fitter.params[k].value, fitter.params[k].value], [0 , hist_dict[k][0].max()], 'r-')
                plt.title(k+ ' :'+"%.2f" % fitter.params[k].value,  y=.9, size=10)
            elif hasattr(self, 'fit_dict'):
                plt.plot([self.fit_dict[k], self.fit_dict[k]], [0 , hist_dict[k][0].max()], 'r-')
                if k == 'max3':
                    plt.title(k+ ' :'+"%.2f" % self.fit_dict[k][0],  y=.9, size=10)
                else:
                    plt.title(k+ ' :'+"%.2f" % self.fit_dict[k],  y=.9, size=10)
            #plt.plot([self.fitter.init_values[k], self.fitter.init_values[k]], [0 , hist_dict[k][0].max()], 'k-')


            #xlabelstr=('  ( time)')
            #plt.ylabel(('|X|^2/f (' + data_unit + '^2/' + sample_unit+ ')'))
            #plt.xlabel(xlabelstr)
            #plt.ylim(0.04,0.08)
            plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')

        #    plt.tick_params(
        #    axis='y',          # changes apply to the x-axis
        #    which='both',      # both major and minor ticks are affected
        #    left='off',      # ticks along the bottom edge are off
        #    right='off',         # ticks along the top edge are off
        #    labelleft='off')

        #    plt.axis('off')


            Nk+=1
            plt.subplots_adjust(left=None, bottom=-.4, right=None, top=.6,
                            wspace= 0, hspace=.001)

            if Nk > 5:
                Nk = 5
                hist_pos_right+=1

        #subplotspec = plt.GridSpec(totalrows,5).new_subplotspec((0, 3),
        #                                               rowspan=subrow,
        #                                               colspan=2)

        Nk=1
        hist_pos=subrow

        for k in list(set(hist_dict.keys()) ^ set(keys_right)):

            S7 = plt.subplot2grid((totalrows,totalcollums), (hist_pos, Nk-1 ),rowspan=1,facecolor='w', colspan=1)

            AX=plt.plot(hist_dict[k][1], np.append(hist_dict[k][0],0), drawstyle='steps-post')
            #S7.update( right=0.05)
            if hasattr(self, 'fitter') & fitter.params.has_key(k):
                plt.plot([fitter.params[k].value, fitter.params[k].value], [0 , hist_dict[k][0].max()], 'r-')
                plt.title(k+ ' :'+"%.2f" % fitter.params[k].value,  y=.9, size=10)
            elif hasattr(self, 'fit_dict'):
                plt.plot([self.fit_dict[k], self.fit_dict[k]], [0 , hist_dict[k][0].max()], 'r-')
                if k == 'max3':
                    plt.title(k+ ' :'+"%.2f" % self.fit_dict[k][0],  y=.9, size=10)
                else:
                    plt.title(k+ ' :'+"%.2f" % self.fit_dict[k],  y=.9, size=10)
            #plt.plot([self.fitter.init_values[k], self.fitter.init_values[k]], [0 , hist_dict[k][0].max()], 'k-')


            #xlabelstr=('  ( time)')
            #plt.ylabel(('|X|^2/f (' + data_unit + '^2/' + sample_unit+ ')'))
            #plt.xlabel(xlabelstr)
            #plt.ylim(0.04,0.08)
            plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')

        #    plt.tick_params(
        #    axis='y',          # changes apply to the x-axis
        #    which='both',      # both major and minor ticks are affected
        #    left='off',      # ticks along the bottom edge are off
        #    right='off',         # ticks along the top edge are off
        #    labelleft='off')

        #    plt.axis('off')


            Nk+=1
            plt.subplots_adjust(left=None, bottom=-.4, right=None, top=.6,
                            wspace= 0, hspace=.001)

            if Nk > 4:
                Nk=1
                hist_pos+=1

        return fig

    def plot_fit_polar_hist(self, hist_dict_org,flim=None, datasub=None, k=False):

        hist_dict=hist_dict_org.copy() # make sure that T.hist_dict is not tought
        # try to delete t_scale from dict
        try:
            del hist_dict['tscale']
            print('ignore tscale ')
        except:
            pass


        if hasattr(self, 'fitter_error'):
            fitter=self.fitter_error
        elif hasattr(self, 'fitter'):
            fitter=self.fitter
        else:
            fitter=None

        totalrows=8
        subrow=totalrows-3
        nhists=np.size(hist_dict.keys())
        nhists_half=int(np.ceil((nhists)/2.))
        import brewer2mpl
        #flim=(0.04,0.08)

        if datasub is None:
            datasub=self.masked_data
        else:
            datasub=datasub

        time=self.time_dict['dt64']
        f=self.f
        flim= (self.geo['f_low'], self.geo['f_high']) if flim is None else flim

        mmax=datasub.max()
        cval=self.clevs#np.linspace(0, mmax, 31)
        sample_unit='s'
        data_unit='m'
        datalabel='Energy (' + data_unit + '^2/' + sample_unit+ ')'
        xlabelstr=('(Time)')
        fig=M.figure_axis_xy(10, 7.5, fig_scale=1, container=True, view_scale=.5)

        if hasattr(self, 'fitter'):
            plt.suptitle(self.ID +' | '+str(self.time_dict['dt64'][int(self.time_dict['dt64'].size/2)].astype('M8[h]')) +
                            '\n redchi=' +str(round(self.fitter.redchi,3))
                            +', normalized chisqrt=' +str(round(self.fit_dict['normalized_chisqr'],3))
                            +', maxres='+str(round(self.fit_dict['max3'][0],3)), y=1.06)
        else:
            plt.suptitle(self.ID +' | '+str(self.time_dict['dt64'][int(self.time_dict['dt64'].size/2)].astype('M8[h]')) +
                            '\n redchi=' +'undefined'
                            +', normalized chisqrt=' +str(round(self.fit_dict['normalized_chisqr'],3))
                            +', maxres='+str(round(self.fit_dict['max3'][0],3)), y=1.06)


        # Model and Data in 2D
        S2 = plt.subplot2grid((totalrows,5), (0, 0),rowspan=subrow,facecolor='white', colspan=2)
        cmap = brewer2mpl.get_map('Paired', 'qualitative', 6, reverse=False).mpl_colormap


        ax2=plt.contourf(time,f,datasub.T, cval, cmap=cmap)


        cb=plt.colorbar(label=datalabel, fraction=0.046, pad=0.09, orientation="horizontal")
        cb.outline.set_visible(False)

        ticklabels,ticks_rounded=MT.tick_formatter(cval, interval=2, rounder=0, expt_flag=True)
        cb.ax.set_yticklabels(ticks_rounded)

        #cb.label()
        plt.contour(self.time_dict['dt64'],f,self.model_result_corse.reshape(self.time.size, f.size).T, colors='black', alpha=0.5)

        #plt.plot(time, self.fitter.params['slope'].value*time+self.fitter.params['intersect'], c='r')

        ax=plt.gca()
        ax.xaxis_date()
        Month = dates.MonthLocator()
        Day14 = dates.DayLocator(interval=5)
        Day = dates.DayLocator(interval=1)#bymonthday=range(1,32)
        Hour = dates.HourLocator(interval=6)#bymonthday=range(1,32)

        dfmt = dates.DateFormatter('%y-%b-%dT%H:%M')


        ax.xaxis.set_major_locator(Day14)
        ax.xaxis.set_major_formatter(dfmt)
        ax.xaxis.set_minor_locator(Day)


        plt.ylabel(('f (Hz)'))
        #plt.xlabel(xlabelstr)
        plt.ylim(flim)
        #plt.grid()
        plt.title('Model and Data in 2D', loc='left', y=1)

        # time mean residual
        S6 = plt.subplot2grid((totalrows,5), (0, 2),rowspan=subrow,facecolor='w', colspan=1 )


        model_result_tmean=np.copy(self.model_result_corse)
        model_result_tmean[self.nan_track]=np.nan
        model_result_tmean=np.nanmean(model_result_tmean.reshape(self.time.size, f.size).T, 1)


        residual_tmean=np.nanmean(self.residual_2d.T,1)

        data_tmean=np.nanmean(datasub.T, 1)
        #plt.plot(f,  weight_tmean, label='diff')

        plt.plot(model_result_tmean,f , label='model', c='k')
        #plt.plot(f,model_tmean_corse , label='model corse')
        #plt.plot(f,residual_tmean, label='residual func')
        plt.plot(data_tmean,f, label='data' , c='b', alpha=0.7)
        plt.plot(-(data_tmean-model_result_tmean),f ,c='g', alpha=.4, label='residual')

        plt.plot(residual_tmean,f, c='g', label='(model) weighted residual' )

        if hasattr(self, 'plain_fitter'):
            if hasattr(self.plain_fitter, 'model_timemean'):
                plt.plot(self.plain_fitter.model_timemean,f, label='substracted surface', c='orange', alpha=.6 )
                plt.plot(data_tmean - self.plain_fitter.model_timemean,f, label='adjusted data', c='b')

        plt.legend(frameon=False, loc=1, #bbox_to_anchor=(0., .8, 1., .8)
                   ncol=1, mode="expand", borderaxespad=0.)

        plt.xlim(flim)
        pcut=M.cut_nparray(f, flim[0], flim[1])
        lylim=np.max([data_tmean[pcut].min() ,data_tmean[pcut].max()])
        lymin=-lylim*.4
        plt.xlim(lymin ,lylim)

        plt.title('Time mean Residual', loc='left', y=1)
        #S6.set_yticks([])
        plt.xlabel(('Amplitude'))
        #plt.grid()
        plt.plot(data_tmean*0,f, c='grey' )

        plt.subplots_adjust(left=None, bottom=None, right=None, top=.9,
                        wspace= 0.25, hspace=.5)



        subplotspec = plt.GridSpec(totalrows,5).new_subplotspec((0, 3),
                                                       rowspan=subrow,
                                                       colspan=2)
        figl=plt.gcf()
        Sr=figl.add_subplot(subplotspec, projection="polar")

        PS=M.plot_polarspectra(self.MEM['freq'],self.MEM['theta']*np.pi/180,self.MEM['D'],
                       unit=self.MEM['unit_dir'],data_type='fraction',lims=(5,40), verbose=False)
        PS.ylabels=np.arange(10, 50, 10)
        PS.linear(ax=Sr,circles=self.MEM['circ_lim'] )

        plt.title('Directional Spectrum', loc='left', y=1)



        Nk=1
        hist_pos=subrow
        for k in hist_dict.keys():

            S7 = plt.subplot2grid((totalrows,5), (hist_pos, Nk-1 ),rowspan=1,facecolor='w', colspan=1)

            AX=plt.plot(hist_dict[k][1], np.append(hist_dict[k][0],0), drawstyle='steps-post')
            #S7.update( right=0.05)
            if fitter is not None & fitter.params.has_key(k):
                plt.plot([self.fitter.params[k].value, self.fitter.params[k].value], [0 , hist_dict[k][0].max()], 'r-')
            elif hasattr(self, 'fit_dict'):
                plt.plot([self.fit_dict[k], self.fit_dict[k]], [0 , hist_dict[k][0].max()], 'r-')
            #plt.plot([self.fitter.init_values[k], self.fitter.init_values[k]], [0 , hist_dict[k][0].max()], 'k-')


            #xlabelstr=('  ( time)')
            #plt.ylabel(('|X|^2/f (' + data_unit + '^2/' + sample_unit+ ')'))
            #plt.xlabel(xlabelstr)
            #plt.ylim(0.04,0.08)
            plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')

        #    plt.tick_params(
        #    axis='y',          # changes apply to the x-axis
        #    which='both',      # both major and minor ticks are affected
        #    left='off',      # ticks along the bottom edge are off
        #    right='off',         # ticks along the top edge are off
        #    labelleft='off')

        #    plt.axis('off')

            if fitter is not None & fitter.params.has_key(k):
                plt.title(k+ ' :'+"%.2f" % self.fitter.params[k].value,  y=.9, size=10)
            elif hasattr(self, 'fit_dict'):
                if k == 'max3':
                    plt.title(k+ ' :'+"%.2f" % self.fit_dict[k][0],  y=.9, size=10)
                else:
                    plt.title(k+ ' :'+"%.2f" % self.fit_dict[k],  y=.9, size=10)
            Nk+=1
            plt.subplots_adjust(left=None, bottom=-.4, right=None, top=.6,
                            wspace= 0, hspace=.001)

            if Nk > 5:
                Nk=1
                hist_pos+=1

        return fig

    def cal_intersect_adjust(self, params):

        """
        Shifts intersect line to cross maximumof model
        returns the new intersect with the time axis
        """
        pp=np.where( self.model_result.reshape(self.time_syntetic.size,self.f.size).T  == self.model_result.max() )

        f_max=self.f[pp[0][0]]
        t_max=self.time_syntetic[pp[1][0]]

        pp1=t_max - f_max / params['slope'].value
        #deltat=(pp1-t_max)
        return pp1

    def convert_normalized_intersect_slope(self, params=None, stderr=True, ndata=None, return_type='Pandas'):
        #intersect is in freq.
        import pandas as pd
        if self.fitter_error:
            params=self.fitter_error.params if params is None else params
        else:
            #raise Warning("fitter_error does not exist. exit without conversion.")
            #print("!!fitter_error does not exist. exit without conversion.")
            self.write_log('varable not set: fitter_error does not exist. exit without conversion.')
            stderr=False

        # create headers for Table
        ndata=self.fitter.ndata if ndata is None else ndata
        SM_dict=dict()
        if stderr:
            s=[ "slope (Hz/[])", "Initial Time ([])", "Initial Time Peak([])", "Storms Time delta", "Data Cut t0", "Slope Std (Hz/[])","Time Std ([])"]
            SM_dict_pandas = pd.DataFrame({'unit':s} , index=['dfdt', 't0', 't0_peak' , 'DT', 'frame_t0','dtdf_std', 't0_std']).T #, columns= ['units']
            SM_dict_pandas.loc['unit'] = SM_dict_pandas.loc['unit'].astype('category')
            #df2.loc['units']=units1
        else:
            s=[ "slope (time_unit/sec)", "Initial Time (time_unit)",  "Initial Time Peak([])", "Storms time delta", "Stroms geo t0"]
            SM_dict_pandas = pd.DataFrame({'unit':s} , index=[ 'dfdt', 't0', 't0_peak', 'DT', 'frame_t0']).T #, columns= ['units']
            SM_dict_pandas.loc['unit'] = SM_dict_pandas.loc['unit'].astype('category')

        for key in self.time_dict.keys():
            if key ==  'datetime':
                pass
            else:
                unit =  key
                Tmin1=  self.time_dict[unit][-1]
                T0  =   self.time_dict[unit][0]
                Dt  =   (Tmin1-T0)

                t0      =    Dt *  params['intersect'] + T0

                intersect_adjusted=self.cal_intersect_adjust(params) ## add adjustedintersect here!! estiamted line goes now thourgh maximumo fthe model
                t0_peak  =   Dt *  intersect_adjusted  +  T0

                #print(Dt , type(Dt), key)
                if key in ['dt64','time']:
                    dfdt=    -999
                else:
                    dfdt=    params['slope'].value / Dt

                SM_dict[key]={'t0':t0, 't0_peak':t0_peak, 'dfdt':dfdt}

                if stderr:
                    if ndata is None:
                        raise Exception("ndata=None, set number of data points ndata from S.fittern.ndata")
                    else:
                        dels=params['slope'].stderr#*np.sqrt(ndata) #emece provides= std (q.84-q.16)/2, not standard error!
                        #delf0=params['intersect'].stderr#*np.sqrt(ndat
                        delt0=params['intersect'].stderr# units of normalized time

                        #t0_std=Dt * np.sqrt( dels**2 *intersect_adjusted**2 /params['slope'].value**4 +  delf0**2 /params['slope'].value**2  )
                        t0_std=Dt * delt0


                        if key in ['dt64','time']:
                            dtdf_std=-999
                        else:
                            #dTdf_std= abs(dels /(params['slope'].value**2 ))
                            dtdf_std= abs(Dt * dels /params['slope'].value**2) # units of Hz/sec
                            #dfdt_std= params['slope'].stderr / Dt # units of Hz/sec

                        #dfdt_std= abs(Dt * dels * params['slope'].value**2)

                        SM_dict[key]['t0_std']=t0_std
                        SM_dict[key]['dtdf_std']=dtdf_std

                        SM_dict_pandas.loc[key]= [dfdt,t0, t0_peak, Dt, T0, dtdf_std, t0_std]

                else:
                    SM_dict_pandas.loc[key]=[dTdf,t0, Dt, T0]
        g=9.8196

        SM_dict_pandas['r0'] = g /(4*np.pi* SM_dict_pandas['dfdt'].iloc[1:])
        SM_dict_pandas.loc['unit']['r0']='Radial Distance (m)'

        SM_dict_pandas['r0_std'] = SM_dict_pandas['dtdf_std'].iloc[1:] * g /(4* np.pi)
        #SM_dict_pandas['r0_std'] = g * SM_dict_pandas['dfdt_std'].iloc[1:] /  ( 4 * np.pi * SM_dict_pandas['dfdt'].iloc[1:]**2)
        SM_dict_pandas.loc['unit']['r0_std']='Radial Distance Std (m)'
        # --
        SM_dict_pandas['r0_deg'] = SM_dict_pandas['r0'].iloc[1:]/(1000.0*110.0)
        SM_dict_pandas.loc['unit']['r0_deg']='Radial Distance (deg)'

        SM_dict_pandas['r0_deg_std'] = SM_dict_pandas['r0_std'].iloc[1:]/(1000.0*110.0)
        SM_dict_pandas.loc['unit']['r0_deg_std']='Radial Distance Std (deg)'

        self.SM_dict_pandas=SM_dict_pandas

        #, 'radial distance'
        SM_dict['r0m']=g /(4*np.pi*SM_dict['sec']['dfdt'])
        SM_dict['r0km']=SM_dict['r0m']/1000

        self.SM_dict=SM_dict
        self.write_log('created self.SM_dict and self.SM_dict_pandas')

        if return_type == 'Pandas':
            return self.SM_dict_pandas
        else:
            return self.SM_dict

    # --- directional Spectra methods
    def find_event_freqrange(self, quantiles):
        """
        find the quantiles of time mean frequency model
        """
        model_shape=self.model_result_corse.reshape(self.time.size, self.f.size)
        cdf=np.cumsum(model_shape.mean(0))
        cdf=cdf/cdf[-1]
        pos=list()
        for p in quantiles:
            pos.append(self.f[np.where(cdf >=p )[0][0]])
        return pos

    def determine_peak_angle(self, quantile=[.1, .5, .9]):
        from m_spectrum import peak_angle
        """
        This function finds the peak angle for the freqency range around the fitted model peak

        Paramters:
        ----------------
        S    Storm class. must have fitted model (S.model_result_corse, S.time, S.f) as well as \
                directional spectrumm dictionary S.MEM
        quantile    list of quantile range for definening quantile range around frequency peak\

        Returns:
        ----------------
        S.SM_dict['angle1']    Peak angle in degree (-180,180)
        S.SM_dict['angle2']    Secondary Peak in degree (-180, 180) if not found set to None

        S.MEM['angle1']   =  S.SM_dict['angle1']
        S.MEM['angle2']   =  S.SM_dict['angle2']
        """

        if not isinstance(self.MEM, dict):
            raise Warning('MEM dictionary does not exist. Return without doing stuff')
            return

        fpos=self.find_event_freqrange( quantile)
        #print(fpos)
        #save quantile in fit dict
        qdict=dict()
        for k in range(len(quantile)):
            qdict[quantile[k]]=fpos[k]

        self.SM_dict['freq_fit_quantile']=qdict

        self.MEM['circ_lim']=(1/fpos[0],1/fpos[-1])

        max_jump=5
        smooth_l=5
        angles=peak_angle(self.MEM['D'],self.MEM['freq'],fpos, max_jump, smooth_l )
        #print(angles)

        self.MEM['angle1']=self.MEM['theta'][angles[0]]
        self.SM_dict['angle1']=self.MEM['theta'][angles[0]]

        anlge1list=np.zeros(self.SM_dict_pandas.shape[0]-1)+self.MEM['theta'][angles[0]]
        pandasadd=list()
        pandasadd.append('Degree East from North (Deg)')
        for i in anlge1list:
            pandasadd.append(i)

        self.SM_dict_pandas['angle1']=pandasadd


        if angles.size > 1:
            self.MEM['angle2']=self.MEM['theta'][angles[1]]
            self.SM_dict['angle2']=self.MEM['theta'][angles[1]]

            anlge2list=np.zeros(self.SM_dict_pandas.shape[0]-1)+self.MEM['theta'][angles[1]]
            pandasadd=list()
            pandasadd.append('Degree East from North (Deg)')
            for i in anlge2list:
                pandasadd.append(i)

            self.SM_dict_pandas['angle2']=pandasadd

        else:
            self.MEM['angle2']=None
            self.SM_dict['angle2']=None
            self.SM_dict_pandas['angle2']=None

    def get_max_data(self, data=None, tresh_perc=75, noise_perc=50 , plot_flag=False):
        """
        This function returns the maximum Spectrum of the data under the fitted model function, as well as the data under the model

        Inputs:
        S           Storm class
        data        (default) S.masked_data_less_noise The data where the maximu is picked from
        tresh_perc  used to define the contour which seperates the data area from the Noise
                    tresh_perc is a percentile of the model data points that should be included by this contour
        noise_perc  Not used atm
        plot_flag   If True the resulting cutted data is plotted, the plot is in normalized units

        Returns:
        Smax        peak amplitude
        fmax        frequency of peak amplitude
        tmax        dictonary of peak time

        data_best       non normalized Spectragram of the data under the fitted model > tresh_perc
        (masked_data2)  Nan where fitted model is below the threshhold contour
                        Units: m^2/Hz

        """
        
        if hasattr(self, 'fitter_error'):
            fitter=self.fitter_error
        else:
            fitter=self.fitter


        data=self.masked_data_less_noise if data is None else data
        tresh=np.percentile(self.model_result_corse,tresh_perc)
        model_mask=self.model_result_corse < tresh
        model_mask_2d=model_mask.reshape(self.time.size, self.f.size)

        #tresh_noise=np.nanpercentile(data,noise_perc)
        #noise_level=tresh_noise

        masked_data2=np.copy(data/self.fit_dict['factor'])# denormalize data
        masked_data2[model_mask_2d]=np.nan
        signal_level=np.nanmean(np.nanmean(masked_data2))

        if plot_flag:
            self.plot_fitsimple(datasub=masked_data2* self.factor)

        dd=np.nanmean(masked_data2,0)
        if np.isnan(dd).sum() == dd.size:
            self.write_log('no max data found because data under model is nan')
            tmax=dict()
            for k,T in self.time_dict.iteritems():
                tmax[k]=np.nan

            return np.nan, np.nan, tmax, masked_data2

        ppp=M.find_max_ts(dd, smooth=True,spreed=2, verbose=False)[0]
        pp=ppp[np.where(dd[ppp]==dd[ppp].max())[0][0]]

        fmax=self.f[pp]

        red_line=(  self.time_dict['normalized']-  self.fitter.params['slope'].value  ) * self.fitter.params['intersect'].value
        tmaxpos=np.where(np.min(abs(red_line-fmax))==abs(red_line-fmax))[0][0]

        tmax=dict()
        for k,T in self.time_dict.iteritems():
            tmax[k]=T[tmaxpos]

        return dd[pp], fmax, tmax, masked_data2

    def event_time(self):
        return self.get_max_data()[2]['dt64'].astype('M8[h]')

    def plot_line_params(self,time, slope, intersect, *args, **kwargs):
        """
        This method is a wrapper for plotting the sloped from the parameters slope and intersect
        inputs:
        time             time vector. For params, use normalized time [0, 1] np.array
        slope            slope of the "dispersed peak frequencies" df/dt [Hz/ normalized time]
        intersect        intersect with the time axis in units of normalized time
        **kargs          are passed to plt.plot routine

        returns:
        None

        """
        intersectF = - intersect * slope
        pfreq      =  time * slope + intersectF
        plt.plot(time, pfreq,  *args, **kwargs)

    def plot_line_params_realtime(self,time=None, *args, **kwargs):
        """
        This method is a wrapper for plotting the sloped from the parameters slope and intersect
        inputs:
        time             not normalized time vector
        **kargs          are passed to plt.plot routine

        returns:
        None

        """

        if hasattr(self, 'fitter_error'):
            fitter=self.fitter_error
        else:
            fitter=self.fitter

        time=self.time_dict['dt64'] if time is None else time
        #intersectf=- self.SM_dict['normalized']['t0'].value * 1/self.SM_dict['normalized']['dTdf']
        #plt.plot(time, self.time_dict['normalized']  * 1/self.SM_dict['normalized']['dTdf'] +intersectf,*args, **kwargs)

        intersectf=- self.SM_dict['normalized']['t0'] * self.SM_dict['normalized']['dfdt']
        plt.plot(time, self.time_dict['normalized']  * self.SM_dict['normalized']['dfdt'] +intersectf,*args, **kwargs)


        #plt.plot(time, (self.time_dict['normalized']-self.SM_dict['normalized']['t0'])/self.SM_dict['normalized']['dTdf'],*args, **kwargs)

    def write_log(self, s, verbose=False):
        self.hist=MT.write_log(self.hist, s, verbose=verbose, short=False)
    def log(self):
        #print('.hist variable')
        print(self.hist)

def reshape_residuals(resid,nan_track,  datashape):
    resid_shape=np.ones(nan_track.size)*np.nan
    it = np.nditer(resid)
    for k in np.arange(0, resid_shape.size):
        if ~nan_track[k]:
            resid_shape[k]=it[0]
            #print(k is False)
            #print(k, it[0])
            it.iternext()
        else:
            resid_shape[k]=np.nan

    return resid_shape.reshape(datashape)

class station_stats(object):
    def __init__(self, ID, save_path=None, plot_path=None):

        self.ID=ID
        self.hist='------ | '+ self.ID
        self.write_log('initialized')
        self.save_path=None if save_path is None else save_path
        self.plot_path=None if plot_path is None else plot_path

        self.plot_flag=True
        self.save_flag=True

        #self.L=None

    def create_estimates_table(self, stormlist):
        import pandas as pd
        #Create Table
        self.write_log('Create table with Storm Estimames')

        L1= pd.DataFrame(index=[ 't0',   't0_std' , 't0_std_day' , 'r0', 'r0_std' , 'r0_deg', 'r0_deg_std', 'angle1','angle2']) #, columns= ['units']
        for storm in stormlist:
            try:
                S=MT.h5_load(storm, self.save_path)
                if 'angle1' in S.T.index:
                    L1[storm]=[S['t0']['dt64'],S['t0_std']['dt64'], S['t0_std']['day'],    S['r0']['sec'],  S['r0_std']['sec'],     S['r0_deg']['sec'] , S['r0_deg_std']['sec'], S['angle1']['sec'],  S['angle2']['sec']  ]
                else:
                    L1[storm]=[S['t0']['dt64'],S['t0_std']['dt64'], S['t0_std']['day'],    S['r0']['sec'],  S['r0_std']['sec'],     S['r0_deg']['sec'] , S['r0_deg_std']['sec'], None, None ]
            except:
                print('fail to load', storm)
                self.write_log('fail to load '+ storm)
                pass

        # Convert Errors estimates in Boundaries
        stdfac=1 # Thisis a adjustment factor, since the Error is just too large!
        self.write_log('use adjusting factor for uncertainties, they are off!')

        L1.loc['timelB']= [-L1.loc['t0_std'][k]*stdfac+L1.loc['t0'][k] for k in range(L1.shape[1])]
        L1.loc['timeuB']= [+L1.loc['t0_std'][k]*stdfac+L1.loc['t0'][k] for k in range(L1.shape[1])]

        #(L1.loc['timeuB']-L1.loc['timelB'])#/(60*60*24)

        stdfac=1 # Thisis a adjustment factor, since the Error is just too large!
        L1.loc['radiallB']= [-L1.loc['r0_deg_std'][k]*stdfac+L1.loc['r0_deg'][k] for k in range(L1.shape[1])]
        L1.loc['radialuB']= [+L1.loc['r0_deg_std'][k]*stdfac+L1.loc['r0_deg'][k] for k in range(L1.shape[1])]

        self.single_storm=MT.h5_load(stormlist[0], self.save_path)
        hist, self.single_storm_meta=MT.json_load(stormlist[0], self.save_path)
        return L1

    def create_max_data_table(self, stormlist, return_data=False, data=None):
        import pandas as pd

        self.write_log('Create table for maximal datapoint')
        Ldata_max= pd.DataFrame(index=[ 'datamax',  'fmax' , 'tmax_dt64' , 'tmax_sec' ]) #, columns= ['units']

        masked_data_dict=dict()
        for s in stormlist:
            S=Storm(s)
            S.load(self.save_path, verbose=False)

            #S.plot_fitsimple()
            if data == 'less_noise':
                data=S.masked_data_less_noise
                #print('used masked less noise data')

            elif data == 'masked':
                data=S.masked_data
                #print('used masked data')
            else:
                #print('datetype not set')
                data=None

            Smax, fmax, tmax, masked_data2 =S.get_max_data( data=data )
            if sum(np.isnan([Smax, fmax])) == 2:
                self.write_log(S.ID +' model above none - nop max data')
            Ldata_max[s]=[Smax , fmax, tmax['dt64'], tmax['sec'] ]

            if return_data:
                masked_data_dict[s]={'masked_data':masked_data2, 'f':S.f, 'time':S.time_dict}

        if return_data:
            return Ldata_max , masked_data_dict
        else:
            return Ldata_max

    def create_fitting_stats_table(self, stormlist):
        import pandas as pd
        #Create Table
        self.write_log('Create table with fitting statistics')

        if self.single_storm_meta is not None:
            S=self.single_storm_meta
        else:
            raise Exception("Please assign an Storm (Storm.Storm) to self.single_storm before executing this function")

        self.write_log('calculate max3')
        keys=list()
        for k,I in S.items():
            #print(type(S[k ]))
            if k is 'max3':
                keys.append(k)
            elif isinstance(I , dict):# or isinstance(I , list) :
                #print(k)
                pass
            else:
                #print(type(I))
                keys.append(k)

        self.stat_keys=keys
        L2= pd.DataFrame(index=[ k for k in keys]) #, columns= ['units']
        for storm in stormlist:
            try:
                hist,S2=MT.json_load(storm, self.save_path)
                L2[storm]=[S2[k] for k in keys]
            except:
                #print('fail to load', storm)
                pass
            #L[storm]=[S['t0']['sec'],S['t0_std']['sec'],    S['r0']['sec'],  S['r0_std']['sec'],     S['r0_deg']['sec'] , S['r0_deg_std']['sec'] ]

        return L2

    #def create_angle_table(self, stormlist):
    #    import pandas as pd


        #Create Table
    #    self.write_log('Create table with fitting statistics')

    #    if self.single_storm_meta is not None:
    #        S=self.single_storm_meta
    #    else:
    #        raise Exception("Please assign an Storm (Storm.Storm) to self.single_storm before executing this function")

    #    print(S.keys())
    #    if isinstance(S.MEM, dict):
    #        keys=['anlge1', 'angle2']

    #    self.angle_keys=keys
    #    Langle= pd.DataFrame(index=[ keys ]) #, columns= ['units']
    #    for storm in stormlist:
    #        try:
    #            hist,S2=MT.json_load(storm, self.save_path)
    #            Langle[storm]=[S2.MEM[k] for k in keys]
    #        except:
    #            #print('fail to load', storm)
    #            pass
    #    else:
    #        self.write_log('Angle table obmitted, since MEM did not exsist')
    #
    #    return Langle

    def create_params_table(self, stormlist):
        import pandas as pd
        self.write_log('Create table for fitting parameters and plot if plot flag is True')

        from lmfit import Parameters

        params=Parameters()
        Sp_default=params.load(open(self.save_path+stormlist[0]+'.fittedparrms.json', 'r' )  )
        #print(Sp_default.keys())
        Lparams= pd.DataFrame(index=[i for i in Sp_default.keys()] ) #, columns= ['units']
        #print(Lparams)
        for storm in stormlist:
            try:
                S2=params.load(open(self.save_path+storm+'.fittedparrms.json', 'r' )  )
                Lparams[storm]=[S2[k].value for k in Sp_default.keys()]
            except:
                print('failed '+ storm)

        # create dictionary of histograms and plot
        self.write_log('create dictionary of histograms')
        if not hasattr(self, 'hist_dict'):
            self.hist_dict=dict()

        if self.plot_flag is False:
            print('set self.plot_flag=True for plotting')

        for k in Sp_default.keys():
            if self.plot_flag:
                F=M.figure_axis_xy(5, 4, fig_scale=1, container=False, view_scale=.5)

            H=plt.hist(Lparams.loc[k].values.tolist(), 40,  color='blue', histtype='bar')
            self.hist_dict[k]=H
            if self.plot_flag:
                plt.title(self.ID+' '+ k, loc='left')
                plt.grid()
                plt.xlabel(k)
                #F.ax.set_yticks([])
                F.make_clear_weak()
                F.save(name=self.ID+'_fit_dict_params_'+k, path=self.plot_path)

        return Lparams

    def create_tables(self, stormlist):
        from pandas import concat
        self.write_log('create all Tables')
        L1=self.create_estimates_table(stormlist)
        L2=self.create_fitting_stats_table(stormlist)
        Lparams=self.create_params_table(stormlist)
        Ldata_max=self.create_max_data_table(stormlist)

        self.stormlist=stormlist
        L = concat([L1, L2, Lparams, Ldata_max], keys=['estimates', 'params', 'storms', 'data_peak'])
        #L = concat([L1, L2, Lparams], keys=['estimates', 'params', 'storms'])
        self.L=L.transpose()

    def plot_first_overview(self, L1=None, sation_lat=-78.18,lat_lim=(-80,90) ):
        from matplotlib import colors
        self.write_log('Statio latitude is at '+ str(sation_lat))

        if L1 is not None:
            L1=L1['estimates'].transpose()
            print('use L1 input')
        elif hasattr(self, 'L'):
            print('use L input')
            L1=self.L['estimates'].transpose()
        else:
            raise Exception("Please assign a pandas dataframe to L1 or make shure that L is propper build")

        if hasattr(self, 'single_storm'):
            S=self.single_storm
        else:
            raise Exception("Please assign a single pandas dict (Storm.Storm.SM_dict_pandas) to self.single_storm before executing this function")

        F=M.figure_axis_xy(12, 4, fig_scale=1, container=True, view_scale=.5)
        plt.suptitle(self.ID+ ' Overview | total='+str(L1.shape[1]) )#, ha='left', x=.05)

        # Initalized model and data 2D
        S1 = plt.subplot2grid((1,7), (0, 0),rowspan=1,facecolor='w', colspan=4 )
        S1=M.subplot_routines(S1)
        for k in range(L1.shape[1]):
            plt.plot([L1.loc['timelB'][k],L1.loc['timeuB'][k] ], [L1.loc['r0_deg'][k]+sation_lat, L1.loc['r0_deg'][k]+sation_lat ], c='k' )
            plt.plot([ L1.loc['t0'][k], L1.loc['t0'][k] ]  ,  [L1.loc['radiallB'][k]+sation_lat,L1.loc['radialuB'][k]+sation_lat ], c='r' )

        plt.plot( L1.loc['t0'],L1.loc['r0_deg']+sation_lat,'.', c='k', MarkerSize=3 )

        plt.xlabel(S['t0']['unit'])
        plt.ylabel(S['r0_deg']['unit'])
        plt.title('\nRadial Distance and Time', loc='left')
        plt.grid()
        plt.ylim(lat_lim)
        S1.make_clear_weak()

        S2 = plt.subplot2grid((1,7), (0, 4),rowspan=1,facecolor='w', colspan=1 )
        S2=M.subplot_routines(S2)
        plt.title('# of Storms', loc='left')

        bins=np.arange(-80, lat_lim[1], 2)
        H=plt.hist(L1.loc['r0_deg']+sation_lat, bins, orientation="horizontal")
        plt.grid()
        plt.ylim(lat_lim)
        plt.xlabel('# Storms /L1atitude')
        S2.ax.set_yticks([])
        #S2.ax.yaxis.set_majorticklabels=None
        S2.make_clear_weak()

        S3 = plt.subplot2grid((1,7), (0, 5),rowspan=1,facecolor='w', colspan=1 )
        S3=M.subplot_routines(S3)
        plt.title('Radial Distance Error', loc='left')
        H=plt.hist2d(list(L1.loc['r0_std']/1000.0),list(L1.loc['r0_deg']+sation_lat), bins=(20, bins), norm=colors.LogNorm())
        plt.grid()
        plt.ylim(lat_lim)
        plt.xlabel('Std (km)')
        S3.ax.set_yticks([])
        S3.make_clear_weak()

        S4 = plt.subplot2grid((1,7), (0, 6),rowspan=1,facecolor='w', colspan=1 )
        S4=M.subplot_routines(S4)
        plt.title('Inital Time Error ', loc='left')
        H=plt.hist2d(list(L1.loc['t0_std']/ np.timedelta64(1, 'h')),list(L1.loc['r0_deg']+sation_lat), bins=(20, bins), norm=colors.LogNorm())
        plt.grid()
        plt.xlabel('Std (hours)')

        plt.ylim(lat_lim)
        S4.ax.set_yticks([])
        S4.make_clear_weak()
        if self.save_flag:
            F.save(name=self.ID+'dist_time_stats', path=self.plot_path)

        return F

    def plot_fitting_stats_histograms(self, L2=None):

        if L2 is not None:
            L2=L2
        elif self.L is not None:
            L2=self.L['params'].transpose()
        else:
            warnings.warn("Please assign a pandas dataframe to L2 or make shure that L is propper build")

        if not hasattr(self, 'hist_dict'):
            self.hist_dict=dict()
        save_keys=['redchi', 'chisqr', 'bic', 'normalized_chisqr', 'aic', 'error_frac', 'error_frac_data', 'error_mean_model', 'chisqr_man']

        for k in save_keys:
            #try:
            F=M.figure_axis_xy(5, 4, fig_scale=1, container=False, view_scale=.5)

            plt.title(self.ID+' '+ k, loc='left')

            H=plt.hist(list(L2.loc[k].values), 30,  color='black')
            plt.grid()
            plt.xlabel(k)
            #F.ax.set_yticks([])
            F.make_clear_weak()
            F.save(name=self.ID+'_fit_dict_fitstats_'+k, path=self.plot_path)
            if any(k in s for s in save_keys):
                self.hist_dict[k]=H
            # except:
            #     print('failed : ', k)

        #F=M.figure_axis_xy(5, 4, fig_scale=1, container=False, view_scale=.5)
        #k='max3'
        #plt.title(self.ID+' '+ k, loc='left')
        #H=plt.hist([i[0] for i in L2.loc['max3']], 30,  color='black')
        #plt.grid()
        #plt.xlabel(k)
        #F.ax.set_yticks([])
        #F.make_clear_weak()
        #F.save(name=self.ID+'_fit_dict_fitstats_'+k, path=self.plot_path)

        #self.hist_dict[k]=H


    def plot_radial_distance_hist(self, stormlist=None, name='_r0_deg'):

        stormlist=self.stormlist if stormlist is None else stormlist
        # Plot histogram of radial disrance

        F=M.figure_axis_xy(5, 4, fig_scale=1, container=False, view_scale=.5)

        plt.title(self.ID+' '+name+ ' Histogram', loc='left')
        ll=[i for i in list(self.L.loc[stormlist]['estimates']['r0_deg']) if i is not None]
        hist_list=[i for i in ll if i is not np.nan]
        hist_list2=[i for i in hist_list if ~np.isnan(i)]
        #hist_list2=[i for i in hist_list if i is != 'nan']
        #print(hist_list2)
        #hist_list=[i for i in list(self.L.loc[stormlist]['estimates']['r0_deg']) if i is not np.nan]
        H=plt.hist(hist_list2, 40,  color='green', histtype='bar')
        plt.grid()
        plt.xlabel('r0_deg')
        #F.ax.set_yticks([])
        F.make_clear_weak()
        F.save(name=self.ID+name+'_hist', path=self.plot_path)
        self.write_log('radial_distance plotted and saved'+self.save_path+self.ID+name+'_hist')

    def plot_angle_hist(self, stormlist=None, name='_Angle'):

        stormlist=self.stormlist if stormlist is None else stormlist
        # Plot histogram of radial disrance

        F=M.figure_axis_xy(5, 4, fig_scale=1, container=False, view_scale=.5)

        plt.title(self.ID+' '+name+ '1 Histogram', loc='left')
        ll=[i for i in list(self.L.loc[stormlist]['estimates']['angle1']) if i is not None]
        hist_list=[i for i in ll if i is not np.nan]

        H=plt.hist(hist_list, 40,  color='green', histtype='bar')
        plt.grid()
        plt.xlabel('angle (deg)')
        #F.ax.set_yticks([])
        F.make_clear_weak()
        F.save(name=self.ID+name+'1_hist', path=self.plot_path)
        self.write_log('angle1 plotted and saved'+self.save_path+self.ID+name+'_hist')


        F=M.figure_axis_xy(5, 4, fig_scale=1, container=False, view_scale=.5)

        plt.title(self.ID+' '+name+ '2 Histogram', loc='left')
        ll=[i for i in list(self.L.loc[stormlist]['estimates']['angle2']) if i is not None]
        hist_list=[i for i in ll if i is not np.nan]

        H=plt.hist(hist_list, 40,  color='green', histtype='bar')
        plt.grid()
        plt.xlabel('angle (deg)')
        #F.ax.set_yticks([])
        F.make_clear_weak()
        F.save(name=self.ID+name+'2_hist', path=self.plot_path)
        self.write_log('angle2 plotted and saved'+self.save_path+self.ID+name+'_hist')

    def plot_all_storms(self, name='all_storms', stormlist=None):
        stormlist=self.stormlist if stormlist is None else stormlist

        stormlist_plotter(name, stormlist, self.save_path,  hist_dict=self.hist_dict, save=True, plot_path=self.plot_path)
        print('saved at '+self.save_path+name)
        self.write_log('all storms plotted and save at saved at '+self.save_path+name)

    def plot_all_storms_alias(self, name=None, stormlist=None):
        import glob
        path=self.plot_path+'/'+name
        MT.mkdirs_r(path)
        print(str(os.getcwd()))
        print('linking to plots in path: '+'../../B02_stats/all_storms/')

        if os.listdir(path) == []:
            for s in stormlist:
                os.symlink('../../B02_stats/all_storms/'+s+'.ov.png', path+'/'+s+'.ov.png')
        else:
            #print(path+name+'/*')
            files=glob.glob(path+'/*')
            #print(files)
            for f in files:
                os.remove(f)
            for s in stormlist:
                #print('../all_storms/'+s+'.ov.png')
                os.symlink('../../B02_stats/all_storms/'+s+'.ov.png', path+'/'+s+'.ov.png')



        #stormlist_plotter(name, stormlist, self.save_path,  hist_dict=self.hist_dict, save=True, plot_path=self.plot_path)
        #print('saved at '+self.save_path+name)
        #self.write_log('all storms plotted and save at saved at '+self.save_path+name)


    def save(self):
        import warnings
        import pandas as pd
        warnings.filterwarnings("ignore")

        save_path_local=self.save_path+'stats/'
        MT.mkdirs_r(save_path_local)
        name=self.ID+'_tables_master'

        store = pd.HDFStore(save_path_local+name+'.h5')
        store['L'] = self.L.transpose()
        store.close()

        pickle_savelist=dict()
        psavelistvount=0

        if hasattr(self, 'select1'):
            pickle_savelist['select1']=self.select1
            psavelistvount+=1

        if hasattr(self, 'classA'):
            pickle_savelist['classA']=self.classA
            psavelistvount+=1
        if hasattr(self, 'classB'):
            pickle_savelist['classB']=self.classB
            psavelistvount+=1
        if psavelistvount != 0:
            MT.pickle_save(save_path_local+name+'_selectlists', save_path_local, pickle_savelist)

        if hasattr(self, 'hist_dict'):
            hist_dict2=dict()
            for k, I in self.hist_dict.iteritems():
                print(k + 'to hist_dict2')
                hist_dict2[k]=[I[0].tolist(),I[1].tolist() ]
        else:
            hist_dict2=0
            self.stat_keys=0

        #print(hist_dict2.keys())
        MT.json_save(name, save_path_local, [self.stormlist,hist_dict2, self.stat_keys], verbose=True)
        #store['hist'] = self.hist
        # save log as txt
        MT.save_log_txt(self.ID,save_path_local, self.hist,  verbose=False)

    def load(self):
        import pandas as pd
        save_path_local=self.save_path+'stats/'
        name=self.ID+'_tables_master'

        store = pd.HDFStore(save_path_local+name+'.h5')
        #print(store)
        self.L= store['L']
        self.L=self.L.transpose()
        store.close()

        self.stormlist,self.hist_dict2,self.stat_keys =MT.json_load(save_path_local+name, save_path_local)
        self.hist=MT.load_log_txt(self.ID+'.hist.txt', save_path_local)

        print(self.hist_dict2)

        if hasattr(self, 'hist_dict2') and self.hist_dict2 != 0:
            print('self has attribute hist_dict2')
            self.hist_dict=dict()
            for k, I in self.hist_dict2.iteritems():
                self.hist_dict[k]=(np.array(I[0]),np.array(I[1]) )
        else:
            print('self.hist_dict2  does not exist')

        if os.path.isfile(save_path_local+name+'_selectlists.npy'):
            D=MT.pickle_load(save_path_local+name+'_selectlists', save_path_local)
            for k, v in D.items():
                setattr(self, k, v)

    def write_log(self, s, verbose=False):
        self.hist=MT.write_log(self.hist, s, verbose=verbose, short=False)
    def log(self):
        #print('.hist variable')
        print(self.hist)

def stormlist_plotter_with_hists(listname, stormlist, save_path, hist_dict,  save=False, plot_path=None):
    for SID in stormlist:
        S=Storm(SID)
        S.load(save_path, verbose=False)
        F=S.plot_fit_polar_hist(hist_dict)
        if save:
            if plot_path is None:
                raise ValueError("define plot_path")
            F.save_light(name=SID+'.ov', path=plot_path+'/'+listname)

def stormlist_plotter(listname, stormlist, save_path , hist_dict=None, save=False, plot_path=None):
    for SID in stormlist:
        S=Storm(SID)
        S.load(save_path, verbose=True)
        #try:
        if hist_dict is None:
            F=S.plot_fitsimple()
        elif type(hist_dict) is dict:

            F=S.plot_fit_hist(hist_dict)

        else:
            raise ValueError("hist_dict must be ether not defined or a dictionary of histograms")

        if save:
            if plot_path is None:
                raise ValueError("define plot_path")
            F.save_light(name=SID+'.ov', path=plot_path+'/'+listname)
        # except:
        #     print(S.ID+ ' obmitted')
    return F

def find_systems_by_time(DD, tmin, tmax):
        """
        input:
        DD      dict of cyclone tracks (CycloneModule_10_3.systemtrack)
        tmin    startime (np.datetime64)
        tmax    endtime (np.datetime64)
        if tmin=tmax find nearest

        returns
        D       dict that containsp cyclonetracks in that hace at least 1 timestep in the tmin-tmax window
        """
        D=dict()
        for k, I in DD.iteritems():
            ttime=np.array(I.data['timestamp'])
            if tmin == tmax:
                if (  tmin > ttime[0]  )  & (  tmin < ttime[-1]  ):
                    #print('add')
                    D[k]=I
            else:

                if M.cut_nparray(np.array(I.data['timestamp']), tmin, tmax).any():
                    D[k]=I
        print('found '+str(len(D)) + ' Tracks')
        return D

class Storm_match(Storm):

    def __new__(cls, Storm, tracks, tracks_name, t_best_guess=None):
        Storm.__class__ = cls
        return Storm

    def __init__(self, Storm, tracks, tracks_name, t_best_guess=None):
        """
        picks stormtracks that have at least one coman date with t_best_guess
        returns:
        SM.Tracks  dictionary of Systemtrack objects SM.Tracks.data is a Pandas Table that contains the track data

        """
        #self.plot_fitsimple()
        #Storm.Storm.__init__(self, SID)
        #self.load(save_path, verbose=True)
        self.add_tracks(tracks, tracks_name, t_best_guess)

    def add_tracks(self, tracks, name, t_best_guess= None):
        """
        This method add another set of track to the Tracks dict. according to t_best_guess
        """
        self.SM_dict_pandas['t0']['dt64']
        t_best_guess=self.SM_dict_pandas['t0']['dt64'] if t_best_guess is None else t_best_guess
        if not hasattr(self, 'Tracks_all' ):
            self.Tracks_all=dict()

        self.Tracks_all[name]= find_systems_by_time( tracks, np.datetime64(t_best_guess), np.datetime64(t_best_guess) )

    def find_systems_by_time(self, DD, tmin, tmax):
            """
            input:
            DD      dict of cyclone tracks (CycloneModule_10_3.systemtrack)
            tmin    startime (np.datetime64)
            tmax    endtime (np.datetime64)
            if tmin=tmax find nearest

            returns
            D       dict that containsp cyclonetracks in that hace at least 1 timestep in the tmin-tmax window
            """
            D=dict()
            for k, I in DD.iteritems():
                ttime=np.array(I.data['timestamp'])
                if tmin == tmax:
                    if (  tmin > ttime[0]  )  & (  tmin < ttime[-1]  ):
                        #print('add')
                        D[k]=I
                else:

                    if M.cut_nparray(np.array(I.data['timestamp']), tmin, tmax).any():
                        D[k]=I
            print('found '+str(len(D)) + ' Tracks')
            return D

    def define_erros(self, SM_dict_pandas=None , std_fact=2.0):
        """
        This Method defines the error ranges in space and time
        for cyclone matching
        inputs:
        dict    dict that as the erros in units of meters and seconds. Must be a S.SM_dict_pandas dict
        std_fact factor that multiplies the std estimate, default 2. Mean the erros are 2 standard deviations around the mean

        outputs:
        self.errorlimts     dict with 3 entry list (min, best estimate, max) for the keys r0 (meters) and t0 (datetime64 timestamp)
        """

        SM_dict_pandas=self.SM_dict_pandas if SM_dict_pandas is None else SM_dict_pandas

        radius=list()
        radius.append( self.SM_dict_pandas['r0']['sec'] -  self.SM_dict_pandas['r0_std']['sec'] *std_fact )
        radius.append( self.SM_dict_pandas['r0']['sec']  )
        radius.append( self.SM_dict_pandas['r0']['sec'] +  self.SM_dict_pandas['r0_std']['sec'] *std_fact )

        t0=list()
        t0.append(   self.SM_dict_pandas['t0']['dt64'] -  self.SM_dict_pandas['t0_std']['dt64']*std_fact )
        t0.append(   self.SM_dict_pandas['t0']['dt64']  )
        t0.append(   self.SM_dict_pandas['t0']['dt64'] +  self.SM_dict_pandas['t0_std']['dt64']*std_fact )

        self.errorlimts={ 'r0':radius , 't0':t0 }

    def match_tracks(self, Station_pos, tracks=None, errorlimts=None):
        """
        This method finds tracks that are within the defined error
        inputs:
        Station_pos   Tuple about Station Position
        tracks        a Systemtrack object that contains tracks
                      if None it takes self.Tracks dictonary
        error_limits  a dict that contains error limits (output of method define_errors)
                      if None it takes self.error_limits dictonary

        outputs:
        self.Tracks   Systemtrack dict with track that match at in space and time.
                      added coloumns to Track.data table:
                      'match_space' True if track position is within errors
                      'match_3d'    True if track postion and time within errors
                      'match_time'  True if track time within errors
        self.Tracks_stats
                      a dict with statistics about the matched storm Tracks_stats

         """

        self.Tracks_stats=dict()
        self.Tracks_matched=dict()
        self.station_pos=Station_pos

        tracks=self.Tracks_all if tracks is None else tracks
        errorlimts=self.errorlimts if errorlimts is None else errorlimts

        for TK_name, TK in tracks.iteritems():
            track_stats=dict()
            track_matched=dict()
            for TID, Track_system in TK.iteritems():
                Track=Track_system.data
                # calculate radial distance to Station
                Track['Radial_to_S']    =Track.apply(lambda row:   M_geo.haversine(   Station_pos[0],   Station_pos[1], row['long'], row['lat'] ) *1000.0    , axis=1) # Haversine formula returns km
                Track['Angle_from_S']   =Track.apply(lambda row:   M_geo.bearing(   Station_pos[0],   Station_pos[1], row['long'], row['lat'] )    , axis=1) # Bering in degree deviation from north


                # define radial Distance error in meters
                radial_min = errorlimts['r0'][0]
                radial_max = errorlimts['r0'][2]
                tmin = errorlimts['t0'][0]
                tmax = errorlimts['t0'][2]


                #ask if track is within these boundaries any time
                Track['match_space']   =  Track['Radial_to_S'].ge(radial_min) & Track['Radial_to_S'].le(radial_max)
                Track['match_time']    =  Track['timestamp'].ge(tmin) & Track['timestamp'].le(tmax)

                #ask if track is within these boundaries and witin the time limit
                Track['match_3d']      =  Track['Radial_to_S'].ge(radial_min) & Track['Radial_to_S'].le(radial_max) &  Track['timestamp'].ge(tmin) & Track['timestamp'].le(tmax)

                #print(sum( Track['match_space'])  )
                #print(sum( Track['match_time'] )  )

                #print( (sum(Track['match_space']) > 0) & (sum(Track['match_time']) > 0) )

                Track['best_guess']=Track['lat'] > 100

                keylist=['lat', 'long360', 'Radial_to_S',  'Angle_from_S', 'radius', 'timestamp']
                if (  sum(Track['match_space'])  > 0  ) &  ( sum(Track['match_time']) > 0 ):
                    track_matched[TID]=Track_system
                    #print( TK_name , TID, ' machted')
                    track_stats[TID]=dict()

                    match_key='match_space'
                    stats_space=dict()

                    tloc=self.return_best_match_index(Track, match_key)
                    Track['best_guess'].loc[tloc]='space'
                    for kk in keylist:
                        stats_space[kk]=Track.loc[tloc][kk]

                    stats_space['col']= tloc
                    track_stats[TID]['space']=stats_space


                    match_key='match_time'
                    stats_time=dict()
                    tloc=self.return_best_match_index(Track, match_key)
                    Track['best_guess'].loc[tloc]='time'
                    for kk in keylist:
                        stats_time[kk]=Track.loc[tloc][kk]

                    stats_time['col']= tloc
                    track_stats[TID]['time']=stats_time

                match_key='match_3d'
                if sum(Track[match_key]) > 0:
                    stats_both=dict()

                    stats_both['3d_match']=True
                    tloc=self.return_best_match_index(Track, match_key)
                    Track['best_guess'].loc[tloc]='3d'
                    for kk in keylist:
                        stats_both[kk]=Track.loc[tloc][kk]

                    stats_both['col']= tloc
                    track_stats[TID]['3d']=stats_both

            #print(TK_name, TID)
            self.Tracks_stats[TK_name]           =track_stats
            self.Tracks_matched[TK_name]         =track_matched

    #def tracks_stats(Tracks):
    def find_max_DpDr(self, tracktable):
        'returns tavble row with max intensity of cyclone'
        return tracktable.iloc[ [tracktable['DpDr'].argmax()]]

    def return_best_match_index(self, Tracktable,match_key):
        """ identifies row in Track Table that is the best match for the given match_key.
            It uses the mean ond the timestamps that are matched and finds the nearest row in the table

        inputs:
        Tracktable   a table of a Systemtrack object
        match_key    an identifier of the table collom. That collom should only have True/False

        output:
        tloc          a row location in track table that is the best match
        """

        matched_times=Tracktable['timestamp'][Tracktable[match_key]]
        return int(np.mean(matched_times.index))

    def plot_tracks_n_circles(self, tracks=None , errorlimts=None, plot_matches=True, subplot=False, col= None):# **kwargs
        """
        tracks        a Systemtrack object that contains tracks

        """
        from AA_plot_base import rosssea_map_plotter
        from m_earth_geometry import create_great_cirle_on_map
        inensity_scaler=45.0

        if col is None:
            try:
                col=M_color.color(path=config['paths']['local_script'], name='mhell_colortheme17')

                col_tild_line       =col.colors['red']
                col_matched_time    =col.colors['green']
                col_matched_space   =col.colors['orange']
                col_matched_both    =col.colors['darkblue']

                grey                =col.grey
                black               =col.black

                line_r              =col.aug1
                line_angle          =col.cascade3

            except:

                col_tild_line       ='red'
                col_matched_time    ='blue'
                col_matched_space   ='lightblue'
                col_matched_both    ='blue'

                grey                =[.5, .5, .5]
                black               ='black'

                line_r              ='orange'
                line_angle          ='lightblue'


        else:

            col_tild_line       =col.colors['red']
            col_matched_time    =col.colors['green']
            col_matched_space   =col.colors['orange']
            col_matched_both    =col.colors['darkblue']

            grey                =col.grey
            black               =col.black

            line_r              =col.aug1
            line_angle          =col.cascade3



        tracks_all          =self.Tracks_all.itervalues().next() if tracks is None else tracks
        tracks_matched  =self.Tracks_matched.itervalues().next() if tracks is None else tracks

        errorlimts=self.errorlimts if errorlimts is None else errorlimts

        P=rosssea_map_plotter(col=None, view_scale=.70, subplot=subplot)

        # all Storm track
        for Csystem in tracks_all.itervalues():
            tx, ty=P.map(list(Csystem.data['long']),list(Csystem.data['lat']) )
            plt.plot(tx, ty,linewidth=.8,  zorder=3,  c=grey)

        #machted storm tracks
        for Csystem in tracks_matched.itervalues():
            tx, ty=P.map(list(Csystem.data['long']),list(Csystem.data['lat']) )
            plt.plot(tx, ty,linewidth=2,  zorder=4,  c=black, alpha=.5)
            plt.scatter(tx, ty, np.exp(Csystem.data['DpDr']/inensity_scaler)/2,c=black, alpha=1, zorder=5)
            try:
                DpDrmax=self.find_max_DpDr(Csystem.data)
                txmax, tymax=P.map(list(DpDrmax['long']),list(DpDrmax['lat']) )
                plt.plot(txmax, tymax, 'o', c='black',zorder=6, markerfacecolor='grey' , markersize=10, alpha=1)
            except:
                pass

            if plot_matches:

                match_key='match_time'
                lon=Csystem.data['long'][Csystem.data[match_key]]
                lat=Csystem.data['lat'][Csystem.data[match_key]]
                #print(list(lon),list(lat) )
                tx, ty=P.map(list(lon),list(lat) )
                #Ptime, =plt.plot(tx, ty, 'o', linewidth=1, markersize=5,  zorder=5,  c=col_matched_time, label='Time Match')
                Ptime= plt.scatter(tx, ty, np.exp(Csystem.data['DpDr'][Csystem.data[match_key]]/inensity_scaler)/2, zorder=5, c=col_matched_time, label='Time Match')
                #match_key='match_space'
                #lon=Csystem.data['long'][Csystem.data[match_key]]
                #lat=Csystem.data['lat'][Csystem.data[match_key]]
                #print(list(lon),list(lat) )
                #tx, ty=P.map(list(lon),list(lat) )
                #Pspace, =plt.plot(tx, ty, 'o', linewidth=.5, markersize=3,  zorder=6,  c=col_matched_space ,  label='Space Match')


        # #In[]

        if errorlimts is not None:
            circs_mappped=list()
            for radius in errorlimts['r0']:
                circ= create_great_cirle_on_map(P.map, self.station_pos[0], self.station_pos[1], radius/1000.0, angle_range=(-120,140) )
                circs_mappped.append(circ )#, lw=2., Color=col.colors['red'], zorder=7 )

            plt.plot(circs_mappped[0][0],circs_mappped[0][1], lw=1., Color=col_tild_line, alpha=0.5,  zorder=3 )
            plt.plot(circs_mappped[1][0],circs_mappped[1][1], lw=2., Color=col_tild_line, alpha=0.8,  zorder=3 )
            plt.plot(circs_mappped[2][0],circs_mappped[2][1], lw=1., Color=col_tild_line, alpha=0.5,  zorder=3 )

        try:
            plt.legend(handles=[Pspace, Ptime],  loc=1 )#(Ptime, Pspace)
        except:
            pass
        #circ=M_geo.create_great_cirle_on_map(P.map, D_pos['DR01'][0], D_pos['DR01'][1], radius[0]/1000.0, angle_range=(-120,140) )#, lw=1., Color=col.colors['red'], alpha=0.5,  zorder=8 )
        #circ=M_geo.create_great_cirle_on_map(P.map, D_pos['DR01'][0], D_pos['DR01'][1], radius[2]/1000.0, angle_range=(-120,140) )#, lw=1., Color=col.colors['red'], alpha=0.5, zorder=8 )
        return P

        #plt.plot(circs_mappped)

    def plot_synoptic_situation(self, tracks=None , errorlimts=None, plot_matches=True, col=None, view_scale=1):
        import string
        import brewer2mpl
        from matplotlib import colors
        from m_general import runningmean

        if col is None:
            try:
                col=M_color.color(path=config['paths']['local_script'], name='mhell_colortheme17')

                col_tild_line       =col.colors['red']
                col_matched_time    =col.colors['green']
                col_matched_space   =col.colors['orange']
                col_matched_both    =col.colors['darkblue']

                grey                =col.grey
                black               =col.black

                line_r              =col.cascade2
                line_angle          =col.cascade3

            except:

                col_tild_line       ='red'
                col_matched_time    ='blue'
                col_matched_space   ='lightblue'
                col_matched_both    ='blue'

                grey                =[.5, .5, .5]
                black               ='black'

                line_r              ='orange'
                line_angle          ='lightblue'


        else:

            col_tild_line       =col.colors['red']
            col_matched_time    =col.colors['green']
            col_matched_space   =col.colors['orange']
            col_matched_both    =col.colors['darkblue']

            grey                =col.grey
            black               =col.black

            line_r              =col.cascade2
            line_angle          =col.cascade3

        tracks_matched2=self.Tracks_matched.itervalues().next() if tracks is None else tracks
        fn2=iter([i+')' for i in list(string.ascii_lowercase)])


        datasub=self.data * self.factor
        time=self.time_dict['dt64']
        f=self.f
        flim= (  (self.geo['f_low']), self.geo['f_high'])

        mmax=datasub.max()
        cval=self.clevs#np.linspace(0, mmax, 31)
        sample_unit='s'
        data_unit='m'
        datalabel='Power Anomalie  10*log10(' + data_unit + '^2/' + sample_unit+ ')'
        xlabelstr=('(Time)')
        cmap = brewer2mpl.get_map('Paired', 'qualitative', 6, reverse=False).mpl_colormap


        fig=M.figure_axis_xy(8.5, 7, fig_scale=1, container=True, view_scale=view_scale)
        #plt.suptitle('Spectrogram of Wave Event around ' + str(S.event_time().astype('M8[D]'))) #, loc='left')#, y=1)
        #plt.suptitle(S.ID +' | '+str(S.time_dict['dt64'][int(S.time_dict['dt64'].size/2)].astype('M8[h]')) , y=1.0)

        # Model and Data in 2D
        S2 = plt.subplot2grid((5,8), (3, 0),facecolor='white', colspan=3 , rowspan=2)
        plt.contourf(time,f,datasub.T, cval, cmap=cmap)
        cb=plt.colorbar(label=datalabel, orientation="horizontal")
        #cb.label()
        plt.contour(self.time_dict['dt64'],f,self.model_result_corse.reshape(self.time.size, f.size).T, colors='black', alpha=0.5)


        self.plot_line_params_realtime(c=col_tild_line, zorder=9)

        ax=plt.gca()
        ax.xaxis_date()
        Month = dates.MonthLocator()
        Day = dates.DayLocator(interval=2)#bymonthday=range(1,32)
        Hour = dates.HourLocator(range(0,24, 6) ) #interval=24)#bymonthday=range(1,32)

        #dfmt = dates.DateFormatter('%y-%b-%dT%H:%M')
        dfmt = dates.DateFormatter('%y-%b-%dT%H')

        ax.xaxis.set_major_locator(Day)
        ax.xaxis.set_major_formatter(dfmt)
        ax.xaxis.set_minor_locator(Hour)

        # Set both ticks to be outside
        ax.tick_params(which = 'both', direction = 'out')
        ax.tick_params('both', length=6, width=1, which='major')
        ax.tick_params('both', length=3, width=1, which='minor')

        plt.title(fn2.next() + ' Spectrogram of Wave Event \n   ' + str(self.event_time().astype('M8[D]')) + ' | ' + self.ID ,loc='left') # 'Spectrogram of Wave Event around \n' + str(S.event_time().astype('M8[D]'))
        plt.ylabel(('f (Hz 10^-2)'))
        #plt.xlabel(xlabelstr)
        plt.ylim(flim)
        plt.xlim(self.time_dict['dt64'][0],self.time_dict['dt64'][-1] )
        plt.grid()

        S8 = plt.subplot2grid((5,8), (0, 0),facecolor='w', colspan=8, rowspan=3 )
        fig.Smap=S8

        P=self.plot_tracks_n_circles(tracks=None, errorlimts=errorlimts, plot_matches=plot_matches,subplot=True ,  col=col )
        fig.Smap.P=P


        #SIE=AA_grid.grap_AMSR_SIE(convert_pd_timestamp_to_12d_str(S.SM_dict_pandas['t0']['dt64']),
        #                          convert_pd_timestamp_to_12d_str(S.SM_dict_pandas['t0']['dt64']) )
        #dx=3
        #AA.draw_SIE(F.P, SIE.data[0,::dx,::dx], SIE.lon[::dx, ::dx], SIE.lat[::dx, ::dx], smooth=True, colors='black')


        plt.title(fn2.next() + ' Stormtracks and great Circles \n   ' ,loc='left')

        S8 = plt.subplot2grid((5,8), (3, 3),facecolor='white', colspan=5, rowspan=2)

        if plot_matches:
            ylimmax_list=list()
            ylimmin_list=list()
            for Track in tracks_matched2.itervalues():
                track_dpdt=M.nannormalize(  np.array(Track.data['DpDr'].apply(lambda x: np.array(x)))/100.0 )
                pl1, =plt.plot(Track.data['timestamp'], runningmean(track_dpdt, 3, tailcopy=True)*2, linewidth=1.5,  zorder=7,  c=black, alpha=.9, label='DpDr normalized')# label='DpDr (2 *hPa/10e2 km)')
                track_radius=np.array( Track.data['p_cent'].apply(lambda x: np.array(x)))/(100.0*2)
                track_radius=track_radius-np.nanmean(track_radius)
                pl2, =plt.plot(Track.data['timestamp'], runningmean(track_radius, 3, tailcopy=True),  linewidth=1.5,  zorder=7,  c=line_r, alpha=1,  label='Central P anomaly hPa/2') # hPa
                #pl3, =plt.plot(Track.data['timestamp'], Track.data['Angle_from_S'],  linewidth=1.5,  zorder=7,  c=line_angle, alpha=1,  label='Angle to Station (Deg)')


                ymin=np.min([np.nanmin(track_radius), np.nanmin(track_dpdt)])
                ymax=np.max([ np.nanmax(track_radius), np.nanmax(track_dpdt) ])+2

                if Track.data['best_guess'].str.contains('3d').any():
                    lin=Track.data[Track.data['best_guess'] == '3d']
                    plp1, =plt.plot([list(lin['timestamp']) , list(lin['timestamp'])  ] , [0, 0], 'o', markersize=6, linewidth=1.5,  zorder=10,  c=col_tild_line, alpha=.9,  label='3d_match')

                if Track.data['best_guess'].str.contains('time').any():
                    lin=Track.data[Track.data['best_guess'] == 'time']
                    plp1, =plt.plot([list(lin['timestamp']) , list(lin['timestamp'])  ] , [2, 2], 'o', markersize=6, linewidth=1.5,  zorder=9,  c=col_matched_time, alpha=.9,  label='3d_match')

                if Track.data['best_guess'].str.contains('space').any():
                    lin=Track.data[Track.data['best_guess'] == 'space']
                    plp1, =plt.plot([list(lin['timestamp']) , list(lin['timestamp'])  ] , [1, 1], 'o', markersize=6,  linewidth=1.5,  zorder=9,  c=col_matched_space, alpha=.9,  label='3d_match')

                match_line_time=[2 if i == True else np.nan for i in Track.data['match_time']]
                match_line_space=[1 if i == True else np.nan for i in Track.data['match_space']]

                pl4, =plt.plot(Track.data['timestamp'], match_line_time ,  linewidth=3 ,zorder=8, c=col_matched_time, alpha=.6 , label='Time Match')
                pl5, =plt.plot(Track.data['timestamp'], match_line_space,  linewidth=3, zorder=7, c=col_matched_space, alpha=.6 , label='Space Match')
                #plt.plot(Track.data['timestamp'], match_line_both,  c=col.colors['lightblue'] , label='Both Match')
                try:
                    DpDrmax=self.find_max_DpDr(Track.data)
                    pl6, =plt.plot([list(DpDrmax['timestamp']) , list(DpDrmax['timestamp'])  ] , [0, 0], 'o' , c='black',zorder=9, markerfacecolor='grey' , markersize=8, alpha=1 , label='Max. intensity')
                    plt.plot([list(DpDrmax['timestamp']) , list(DpDrmax['timestamp'])  ] , [ymin, ymax], '-' , c='grey',zorder=2, markerfacecolor='grey', linewidth=0.5)
                except:
                    pass
                plt.grid()
                plt.legend(handles=[pl1, pl2,pl4,pl5,pl6 ], loc=1)


                ax=plt.gca()
                ax.xaxis_date()
                Month = dates.MonthLocator()
                Day = dates.DayLocator(interval=2)#bymonthday=range(1,32)
                Hour = dates.HourLocator(range(0,24, 6) ) #interval=24)#bymonthday=range(1,32)

                #dfmt = dates.DateFormatter('%y-%b-%dT%H:%M')
                dfmt = dates.DateFormatter('%m-%dT%H')

                ax.xaxis.set_major_locator(Day)
                ax.xaxis.set_major_formatter(dfmt)
                ax.xaxis.set_minor_locator(Hour)

                # Set both ticks to be outside
                ax.tick_params(which = 'both', direction = 'out')
                ax.tick_params('both', length=6, width=1, which='major')
                ax.tick_params('both', length=3, width=1, which='minor')
                ylimmax_list.append(ymax)
                ylimmin_list.append(ymin)

            if ylimmin_list:
                plt.ylim(min(ylimmin_list), max(ylimmax_list))

        plt.title(fn2.next() + ' Matched Stormtrack quantities \n   ' ,loc='left')
        return fig

class Fetch_Propability(object):
    def __init__(self,SID):
        self.ID=SID

    def create(self, fitter_error, params_dict, timeaxis):

        self.fitter_error     = fitter_error
        self.converted_chain  = self.convert_slope_intersect_to_MS1957(self.fitter_error.flatchain['slope'], self.fitter_error.flatchain['intersect'],timeaxis)
        self.data  , self.time= self.create_rt0_propabilities(self.converted_chain, params_dict)

        self.params_dict      = params_dict

    #def load(self, fitter_error, params_dict, timeaxis):

    def plot_inital(self, save_path=False):
        F=M.figure_axis_xy(x_size=5,  y_size=5, view_scale=.5)
        plt.suptitle(self.ID +' | Initital Intersect-Slope PDF \n' +
                ' correlation:' + str( round(self.fitter_error.params['slope'].correl['intersect'], 2)) , y=1.06)

        M.plot_scatter_2d(self.fitter_error.flatchain['intersect'], self.fitter_error.flatchain['slope'], xname='intersect', yname='slope')


        if save_path is not False:
            F.save_light(name=self.ID+'_intersect_slope_PDF', path=save_path)

    def convert_slope_intersect_to_MS1957(self, slope_chain, intersect_chain, realtime):
        result=convert_slope_intersect_to_MS1957(slope_chain,intersect_chain, realtime )
        result['t0_ns']=result['t0'].astype(int)

        return result

    def create_rt0_propabilities(self, table, params_dict, xarray=True):
        qlim                =params_dict['quantile_limits']
        radial_resolution   =params_dict['radial_resolution']
        time_resolution     =params_dict['time_resolution']

        xlim                =(np.datetime64( table['t0_ns'].astype('datetime64[ns]').quantile(qlim[0]) ).astype('M8[h]'),
                              np.datetime64( table['t0_ns'].astype('datetime64[ns]').quantile(qlim[1]) ).astype('M8[h]'))
        xbins               =np.arange(xlim[0], xlim[1],time_resolution ).astype('M8[m]')
        #print(xbins)
        r0_range    = np.arange(0, 180*110*1000, radial_resolution)
        a   =r0_range-table['r0'].quantile(qlim[0])
        b   =r0_range-table['r0'].quantile(qlim[1])

        ylim=(r0_range[np.unravel_index(np.abs(a).argmin(),np.transpose(a.shape))],  r0_range[np.unravel_index(np.abs(b).argmin(),np.transpose(b.shape))] )
        ybins=np.arange(ylim[0], ylim[1], radial_resolution)
        h=np.histogram2d(table['t0'].astype(int) ,table['r0'], bins=(xbins.astype('datetime64[ns]').astype(int), ybins) )

        if xarray:
            import xarray as xr
            import pandas as pd
            d=h[0]/h[0].sum()
            half_x_dt64=xbins[1:] + (xbins[:-1]-xbins[1:]) /2.0
            half_x_sec=half_x_dt64.astype(int)
            half_radius=(h[2][1:]+h[2][:-1])/2.0

            attrs=copy.copy(params_dict)
            attrs['time_resolution']=str(attrs['time_resolution'])


            #print(d.shape, half_x_dt64.shape, half_radius.shape)
            # G=xr.DataArray(d.T,name='fetchPDF', dims={'time':d.shape[0], 'radius':d.shape[1]},
            #               coords={'time':pd.to_datetime(half_x_dt64),
            #                       'time_sec': (('time') ,half_x_sec),
            #                       'radius': half_radius
            #                        },
            #                      attrs=attrs)

            #rr,  tt= np.meshgrid(half_radius,pd.to_datetime(half_x_dt64)  )
            G = xr.Dataset({'fetchPDF': (['radius', 'time'],  d.T)},
                coords={'radius': (('radius'), half_radius),
                'time': (('time'),pd.to_datetime(half_x_dt64) ),
                'time_sec': (('time') ,half_x_sec)},
                attrs=attrs)
                #'reference_time': pd.Timestamp('2014-09-05')}

            return G, half_x_dt64

        else:
            #h=np.histogram2d(result['t0'].astype(int) ,result['r0'] )
            half_x=(h[1][1:]+h[1][:-1])/2.0
            half_y=(h[2][1:]+h[2][:-1])/2.0
            half_x_time=xbins[1:] + (xbins[:-1]-xbins[1:]) /2.0

            return {'time_borders_sec':xbins.astype(int)  ,'time_borders_dt64':xbins  , 'time_postions':half_x_time ,
                    'radial_borders': ybins, 'data':h[0].T }



        #half_y.shape
        #half_x.shape
        #half_x_time.shape
        #h[0].T.shape
        #plt.contourf(half_x_time, half_y/1000.0 ,  h[0].T)

    def plot(self, save_path=False, F=None):

        from matplotlib import colors

        if F is None:
            F=M.figure_axis_xy(x_size=5,  y_size=5, view_scale=.5)

        if hasattr(self, 'fitter_error'):
            plt.suptitle(self.ID +' | Initital Intersect-Slope PDF \n' +
                    ' correlation:' + str( round(self.fitter_error.params['slope'].correl['intersect'], 2)) , y=1.06)
        else:
            plt.suptitle(self.ID +' | Initital Intersect-Slope PDF \n' +
                    '' , y=1.06)
        #MT.stats_format(self.data)

        if 'col' in globals() or  'col' in locals() :
            cmap= col.colormaps(20)
        else:
            cmap= plt.cm.Greys


        self.clevs=np.arange(0, 1, .05)
        plt.pcolormesh(self.data['time'], self.data['radius']/1000.0, self.data['fetchPDF'], norm=colors.LogNorm(),cmap = cmap)

        plt.xlabel('Time (6 hours)')
        plt.ylabel('Radial Distance (km)')

        ax=plt.gca()
        ax.xaxis_date()
        Month = dates.MonthLocator()
        Day = dates.DayLocator(interval=5)#bymonthday=range(1,32)
        Hour = dates.HourLocator(interval=6)#bymonthday=range(1,32)

        dfmt = dates.DateFormatter('%y-%m-%dT%H:%M')


        ax.xaxis.set_major_locator(Day)
        ax.xaxis.set_major_formatter(dfmt)
        ax.xaxis.set_minor_locator(Hour)

        # Set both ticks to be outside
        ax.tick_params(which = 'both', direction = 'out')
        ax.tick_params('both', length=6, width=1, which='major')
        ax.tick_params('both', length=3, width=1, which='minor')



        if save_path is not False:
            F.save_light(name=self.ID+'_org_PDF', path=save_path)


    def save(self, save_path, as_xarray=True, verbose=False):
        """
        this function save data with xarray the other option is not programmed jet
        """
        save_path_local=save_path+'/origin_PDF/'
        file_path=save_path_local +self.ID+'_origin_PDF.nc'
        #print(file_path)
        if not os.path.exists(save_path_local):
            os.makedirs(save_path_local)
        if as_xarray:
            self.data.to_netcdf(file_path)
            if verbose:
                print('save to:\n'+ file_path)

        else:
            raise Warning('Data is not an assumed to be an xarray.DataArray, saving in another format is not programmed jet')


    def load(self, load_path, verbose=False):
        """
        this function loads data with xarray
        """
        import xarray as xr

        #load_path_local=load_path+self.ID[0:4]+'.LHZ.StormD.resepl/origin_PDF/'
        #file_path=load_path_local +self.ID+'_origin_PDF.nc'
        save_path_local=load_path+'/origin_PDF/'
        file_path=save_path_local +self.ID+'_origin_PDF.nc'
        if verbose:
            print(file_path)
        self.data=xr.open_dataset(file_path)
        #self.data.close()

class Storm_match_pdf(Fetch_Propability):
    def __new__(cls, Fetch_Propability, tracks, tracks_name):
        Fetch_Propability.__class__ = cls

        #Fetch_Propability.data=Fetch_Propability.data/Fetch_Propability.data.sum()
        return Fetch_Propability

    def __init__(self, Fetch_Propability, tracks, tracks_name):
        """
        picks stormtracks that have at least one coman date with t_best_guess
        returns:
        SM.Tracks  dictionary of Systemtrack objects SM.Tracks.data is a Pandas Table that contains the track data

        """
        #self.plot_fitsimple()
        #Storm.Storm.__init__(self, SID)
        #self.load(save_path, verbose=True)
        self.add_tracks(tracks, tracks_name, time_axis=self.data['time'])

    def add_tracks(self, tracks, name, time_axis= None, station_pos=None):
        """
        This method add another set of track to the Tracks dict. according
        to cylones that overlab with the first and last entry of time_axis
        """

        time_axis=self.data['time'] if time_axis is None else time_axis
        if not hasattr(self, 'Tracks' ):
            self.Tracks=dict()

        selected_tracks=find_systems_by_time(tracks, np.array(time_axis[0]),np.array(time_axis[-1])  )
        if station_pos is not None:
            selected_tracks=generate_radial_coords(selected_tracks, station_pos)

        self.Tracks[name]=selected_tracks

    def define_erros(self, radial_quantile=(.1, .9) ,time_quantile= (.1 ,.9), for_plot=False ):
        """
        This Method defines the error ranges in space and time
        for cyclone matching
        inputs:
        radial_quantile    quantile limites for radial distance in self.data
        time_quantile    quantile limites for time stampas in self.data

        outputs:
        self.errorlimts     dict with 3 entry list (min, best estimate, max) for the keys r0 (meters) and t0 (datetime64 timestamp)
        """

        def find_timelim(data, varname, qlim=[.05, .5, .95]):
            xrdata=self.data[varname]
            qlims=list()
            data=(xrdata/xrdata.sum()).sum(dim='radius').cumsum(dim='time')
            masks_bt= data >= data.quantile(qlim)
            for i in range(masks_bt.shape[1]):
                qlims.append(np.array(data.time[masks_bt[:,i]][0]))
                #print(data.time[masks_bt[:,i]][0])
            return qlims


        def find_radialim(data, varname, qlim=[.05, .5, .95]):
            xrdata=data[varname]
            qlims=list()
            data=(xrdata/xrdata.sum()).sum(dim='time').cumsum(dim='radius')
            masks_bt= data >= data.quantile(qlim)
            for i in range(masks_bt.shape[1]):
                qlims.append(float(data.radius[masks_bt[:,i]][0]))
                #print(data.time[masks_bt[:,i]][0])
            return qlims



        def find_best_time(xrdata, varname):
            xrdata=(xrdata/xrdata.sum()).sum(dim='time').cumsum(dim='radius')

        error_time=find_timelim(self.data, 'fetchPDF', time_quantile)#+np.timedelta64(30, 'm')
        error_radial=find_radialim(self.data, 'fetchPDF', radial_quantile)

        t0=error_time
        radius=error_radial

        if for_plot:
            return {'r0':(error_radial[0], error_radial[-1]),
                    't0':(error_time[0], error_time[-1]) }

        else:
            self.error_time  =error_time
            self.error_radial=error_radial
            self.errorlimts={ 'r0':radius , 't0':t0 }


    def load_fetches_table(self,CID, path):
        """
        This function loads a fetch table
        """
        FT=MT.h5_load(CID.replace('.', '_'), path)
        return FT

    def add_analysis_fields(self,fetchtable, station_pos=None ):
        """
        This method adds addtional field to a fetch table

        inputs:
        fetchtable  pandas table with rows wspeed_long, wspeed_lat,
        station_pos (lon, lat ) of a receiving stations position


        added rows:
        U_X             windspeed times sqrt(area)= windspeed * fetch
        non_dim_fetch   g* sqrt(area) / U^2 . Non dimention al fetch fowllowing hasselmann et al

        Radial_to_S  radial distance from stations
        Angle_from_S Bearing from station

        returns:
        fetchtable
        """

        fetchtable['U_X']          = fetchtable['wspeed_mean']* [np.sqrt(i) for i in fetchtable['area_m2']]
        fetchtable['non_dim_fetch']= fetchtable.apply(lambda row: 9.81 *np.sqrt(row['area_m2']) / row['wspeed_mean']**2     , axis=1)
        fetchtable['non_dim_fetch_umax']= fetchtable.apply(lambda row: 9.81 *np.sqrt(row['area_m2']) / row['wspeed_max']**2     , axis=1)

        if station_pos is not None:
            # calculate radial distance to station and angle to station
            # wspeed_long [0, 360]
            # station_pos [0, 360, -90, 90]
            fetchtable['rad_dist']    =fetchtable.apply(lambda row:   M_geo.haversine(  station_pos[0],  station_pos[1], row['wspeed_long'], row['wspeed_lat'] ) *1000.0    , axis=1) # Haversine formula returns km
            fetchtable['Angle_from_S']   =fetchtable.apply(lambda row:   M_geo.bearing(    station_pos[0],  station_pos[1], row['wspeed_long'], row['wspeed_lat']  )    , axis=1) # Bering in degree deviation from north
        return fetchtable


    def load_fetches_from_dict(self,path, tracks_in=None):
        """
        This method loads the corresponding fetches to the tracks
        and converts in radial coordinates
        inputs:
        Station_pos   Tuple about Station Position
        tracks        a dict of Systemtrack objects
                      if None it takes self.Tracks dictonary
        path          path to .h5 files with fetch tables
        outputs:
        tracks        Systemtrack dict with
                      track.data as the table of the stormtracks
                      track.fetches as the table of the fetches belonging to this stormtrack
        fail_list     list of storm where fetches couldn't be found
         """

        tracks=self.Tracks if tracks_in is None else tracks_in
        fail_list=list()
        for TK_name, TK in tracks.iteritems():
            for CID, Track_system in TK.iteritems():
                print(CID)
                # load fetch table for CID
                #try:

                #FT=MT.h5_load(CID.replace('.', '_'), track_path+FetchcollID.string)
                # calculate radial distance to station and angle to station
                #FT['Radial_to_S']    =FT.apply(lambda row:   M_geo.haversine(   self.station_pos[0],   self.station_pos[1], row['wspeed_long'], row['wspeed_lat'] ) *1000.0    , axis=1) # Haversine formula returns km
                #FT['Angle_from_S']   =FT.apply(lambda row:   M_geo.bearing(   self.station_pos[0],   self.station_pos[1], row['wspeed_long'], row['wspeed_lat']  )    , axis=1) # Bering in degree deviation from north
                Track_system.fetches    =self.load_fetches_table(CID, path)
                Track_system.fetches    =self.add_analysis_fields(Track_system.fetches, station_pos=self.station_pos)

                TK[CID]=Track_system
                #except:
                #    print('cant find fetch file '+ CID + ' is skipped ')
                #    fail_list.append(CID)
            tracks[TK_name]=TK

        if tracks_in is not None:
            return tracks, fail_list
        else:
            self.Tracks=tracks
            return fail_list



    # def match_tracks(self, params, datapar, tracks=None, errorlimts=None):
    #     """
    #     This method finds tracks that are within the defined error
    #     inputs:
    #     Station_pos   Tuple about Station Position
    #     tracks        a Systemtrack object that contains tracks
    #                   if None it takes self.Tracks dictonary
    #     error_limits  a dict that contains error limits (output of method define_errors)
    #                   if None it takes self.error_limits dictonary
    #
    #     outputs:
    #     self.Tracks   Systemtrack dict with track that match at in space and time.
    #                   added coloumns to Track.data table:
    #                   'match_space' True if track position is within errors
    #                   'match_3d'    True if track postion and time within errors
    #                   'match_time'  True if track time within errors
    #     self.Tracks_stats
    #                   a dict with statistics about the matched storm Tracks_stats
    #
    #      """
    #     import xarray as xr
    #     from AA_gridded_data import grap_10wind_EASE
    #     from AA_cyclone_track_helpers import fetch_maker
    #     from pandas import concat
    #
    #     self.Tracks_stats=dict()
    #     self.Tracks_matched=dict()
    #     STID_local=self.ID[0:4]
    #     SID=self.ID
    #     self.station_pos=datapar['station_pos'][STID_local]
    #
    #
    #     tracks=self.Tracks_all if tracks is None else tracks
    #     errorlimts=self.errorlimts if errorlimts is None else errorlimts
    #
    #     fetch_mask_dict=dict()
    #
    #     FpSID=False
    #
    #     for TK_name, TK in tracks.iteritems():
    #         Fetches_per_track=dict()
    #         Track_system_matched=dict()
    #
    #         for CID, Track_system in TK.iteritems():
    #
    #             print(CID)
    #             Track=Track_system.data
    #             # calculate radial distance to Station
    #             Track['Radial_to_S']    =Track.apply(lambda row:   M_geo.haversine(   self.station_pos[0],   self.station_pos[1], row['long'], row['lat'] ) *1000.0    , axis=1) # Haversine formula returns km
    #             Track['Angle_from_S']   =Track.apply(lambda row:   M_geo.bearing(   self.station_pos[0],   self.station_pos[1], row['long'], row['lat'] )    , axis=1) # Bering in degree deviation from north
    #
    #
    #             # define radial Distance error in meters
    #             tmin = errorlimts['t0'][0]
    #             tmax = errorlimts['t0'][2]
    #
    #             #print(tmin, tmax)
    #             #ask if track is within these boundaries any time
    #             Track['match_time']    =  Track['timestamp'].ge(tmin) & Track['timestamp'].le(tmax)
    #
    #             #Track[Track['match_time']]['timestamp']
    #
    #             #Track['best_guess']=Track['lat'] > 100
    #
    #             keylist=['lat', 'long360', 'Radial_to_S',  'Angle_from_S', 'radius', 'timestamp']
    #             if sum(Track['match_time']) > 0:
    #
    #                 Winds_all=grap_10wind_EASE(str(tmin), str(tmax),
    #                         product=params['product'],  resolution=params['wind_data_timegrid'], in_parallel_pros=False, gridsize=str(datapar['grid_size'])+'km')
    #
    #
    #                 FpSID=False
    #
    #                 #print(Winds_all['time'])
    #                 #print(Track[Track['match_time']]['timestamp'])
    #                 for row, Tslice in Track[Track['match_time']].iterrows():
    #                     #Tslice=Track[Track['match_time']].iloc[0]
    #                     # load winds
    #                     print(Tslice.timestamp)
    #                     #print(type(Tslice.timestamp))
    #                     #print(Winds_all['time'])
    #                     #print(np.datetime64(Tslice.timestamp).astype('M8[ns]'))
    #                     Winds=Winds_all.sel(time=    slice( Tslice.timestamp-np.timedelta64(15,'m') , Tslice.timestamp+np.timedelta64(15,'m')  ) )
    #
    #                     #Winds=Winds_all.sel(time=Tslice.timestamp, method='nearest')
    #
    #                     tracks_fetch(Tslice, path=params['cyclone_path'] , iteration=2,
    #                                                         SID=SID, CID=CID )
    #                     #print(xr.__version__)
    #                     #return Winds_all, Tslice
    #                     #print( str(SID)  , STID_local, 'pos 2.6')
    #                     all_fetches=tracks_fetches.create_fetches( Winds, 'wspeed', quantile_thresh=params['quantile_thresh'],
    #                                         min_area=params['min_area'], grids=datapar['grids'], grid_size=datapar['grid_size'], parallel=params['parallel'] )
    #
    #                     if params['plot_figures']:
    #                         F=M.figure_axis_xy(x_size=5,  y_size=5, view_scale=.5)
    #                         tracks_fetches.plot_cyclone_area(datapar['grids']['long'], datapar['grids']['lats'])#,mmap=F.Smap.P.map)
    #                         #tracks_fetches.plot_data_above_tresh(datapar['grids']['long'], datapar['grids']['lats'],
    #                         #                                        np.squeeze(Winds['wspeed'])  )#, mmap=F.Smap.P.map
    #                         tracks_fetches.plot_all_fetches(datapar['grids']['long'], datapar['grids']['lats'])# mmap=F.Smap.P.map)
    #                         F.save_light(name=params['key_name'] + '_fetchOV' , path=params['plot_path'] +'/', verbose=False)
    #
    #
    #                     #print( str(SID)  , STID_local, 'pos 3.2')
    #                     # select fectches that fit
    #                     for fetchID, F1 in all_fetches.iteritems():
    #                         Fetches_per_track[CID+'.'+ fetchID]=F1.create_table(
    #                                                     grids=datapar['grids'],
    #                                                     lonlatbox=params['lonlatbox'],
    #                                                     land_mask=datapar['land_mask'],
    #                                                     radial_limits= (self.errorlimts['r0'][0], self.errorlimts['r0'][2]),
    #                                                     station_pos=datapar['station_pos'][STID_local]  )
    #
    #                         fetch_mask_dict[F1.ID.string]=F1.mask
    #                     #print( str(SID)  , STID_local, 'pos 3.5')
    #                     if Fetches_per_track:
    #                         Fptrack=concat([I.T for I in Fetches_per_track.itervalues()])
    #                         if FpSID is False:
    #                             FpSID=Fptrack
    #                         else:
    #                             FpSID=concat([FpSID, Fptrack])
    #                     #plt.xlim(150, 250)
    #                     #plt.ylim(200, 300)
    #
    #                 # collect fetches that match in time in new set of System_tracks
    #                 Track_system.fetches    =Fptrack
    #                 Track_system.data       =Track
    #                 Track_system_matched[CID]      =Track_system
    #
    #     self.fetch_mask_dict=fetch_mask_dict
    #     return Track_system_matched, FpSID


    def derive_best_match(self, FT, long_boundaries=None, prop_tresh=0.001):
        """
        the method derives the best match given
        inputs:
        self.data['fetchPDF'] dims: time (np.timestamp) x radial distance. self.data can be an as_xarray
        FT         fetch table where each row is a fetch.

        The best fetch is with
            max(prapability * non_dim_fetch)

        returns:
        best_CID                cyclones ID of best match
        best_fetch_index        index of FT table
        best_fetch_evolution    (beta) a table that only copntains the evolution of the
                                fetch that matches the best (there is no
                                unique identifyer for each fetch)

        prop_match              True if matched by propability, False else
        coast_dict
        """


        # calculate best fetch, sqrt(area)* U
        #import pandas as pd

        def get_prob(data, time, radius, topo_flag, long, long_boundaries ):
            rad_index=abs(data.radius.data - radius).argmin()

            if long_boundaries is not None:
                box_flag = (long >= long_boundaries[0]) & (long <= long_boundaries[1])
            else:
                box_flag = True


            if (data.time[0] < time)  & (data.time[-1] > time) & topo_flag & box_flag:
                #print('within time limits')
                #print((data.time[0] < time)  & (data.time[-1] > time))
                #rint(data.time[0])
                #print('time at this itteration:' + str(time))
                #print('radial index' + str(rad_index))

                #print(data.sel(time=slice(time), radius=data.radius[rad_index]).data)
                try:

                    #print('exact data point is')
                    #print(str(data.sel(time=time, radius=data.radius[rad_index]).data))

                    return data.sel(time=time, radius=data.radius[rad_index]).data

                except:
                    print('sliced timestep')
                    #print('slice data point is')
                    #print(str(data.sel(time=slice(time), radius=data.radius[rad_index]).mean().data))

                    return data.sel(time=slice(time), radius=data.radius[rad_index]).mean().data
            else:
                return 0

        def get_prob_at_coast(data, time, radius, topo_flag, long, long_boundaries):
            rad_index=abs(data.radius.data - radius).argmin()

            if long_boundaries is not None:
                box_flag = (long >= long_boundaries[0]) & (long <= long_boundaries[1])
            else:
                box_flag = True


            if (data.time[0] < time)  & (data.time[-1] > time) & ~topo_flag & box_flag:
                #print('within time limits')
                #print((data.time[0] < time)  & (data.time[-1] > time))
                return data.sel(time=time, radius=data.radius[rad_index]).data
            else:
                return 0

        def min_dist( max_radius, max_time, radius, timestamp, long, long_boundaries):

            if long_boundaries is not None:
                box_flag = (long >= long_boundaries[0]) & (long <= long_boundaries[1])
            else:
                box_flag = True

            dt=timedist_to_max(max_time,timestamp )
            #dt_time=(timestamp.astype('M8[h]') - np.datetime64( max_time.astype('M8[h]').item() ) ).astype('m8[s]')/np.timedelta64(1, 's') # in seconds
            #cyclones_speed=20.0 # m/s # assuming a mean cyclones speed of 80 km/h = 22 m/s
            #dt = dt_time* cyclones_speed

            dx=(radius - max_radius)
            if box_flag:
                return np.sqrt( (dt)**2 +   (dx)**2 ).item()
            else:
                return 1e15 # marker for fetchces that are either out of the longitudinal boundary


        def timedist_to_max(max_time, timestamp):
            dt_time=(timestamp.astype('M8[h]') - np.datetime64( max_time.astype('M8[h]').item() ) ).astype('m8[s]')/np.timedelta64(1, 's') # in seconds
            cyclones_speed=20.0 # m/s # assuming a mean cyclones speed of 80 km/h = 22 m/s
            dt = dt_time* cyclones_speed
            return dt

        pos=self.data['fetchPDF'].where(self.data['fetchPDF']==self.data['fetchPDF'].max(), drop=True)
        pos.time.data, pos.radius.data

        #row=FT.iloc[3]
        #print(get_prob(self.data['fetchPDF'],row['timestamp'],row['rad_dist'] ))
        FT['propability']=FT.apply(lambda row: get_prob(self.data['fetchPDF'],row['timestamp'], row['rad_dist'] , row['topo_flag'], row['wspeed_long'], long_boundaries )    , axis=1)
        FT['combined_propability']=FT['propability']*FT['non_dim_fetch']
        FT['combined_propability'][FT['propability'] < prop_tresh] = 0


        FT['propability_coast']=FT.apply(lambda row: get_prob_at_coast(self.data['fetchPDF'],row['timestamp'], row['rad_dist'] , row['topo_flag'], row['wspeed_long'], long_boundaries )    , axis=1)
        FT['combined_propability_coast']=FT['propability_coast']*FT['non_dim_fetch']
        FT['combined_propability_coast'][FT['propability'] < prop_tresh] = 0


        FT['dist_to_max']=FT.apply(lambda row: min_dist(pos.radius.data, pos.time.data, row['rad_dist'] , row['timestamp'] , row['wspeed_long'], long_boundaries )    , axis=1)

        FT['raddist_to_max']  = pos.radius.data-FT['rad_dist']
        FT['timedist_to_max'] = FT.apply(lambda row: timedist_to_max(pos.time.data, row['timestamp'] )    , axis=1)

        if (sum(FT['combined_propability'] !=0) != 0):
            prop_match=True
            #print('proparbility match is'+ str(prop_match) )
            best_fetch_loc=FT['combined_propability'].values.argmax()

        else:
            prop_match=False

            #print(FT['topo_flag'])
            # topo_flag = true for fetches over ocean!

            #print(FT[FT['topo_flag']]['dist_to_max'])
            # find minimum radial distance that is not over land.
            kk=FT[FT['topo_flag']]['dist_to_max'].idxmin()
            best_fetch_loc=FT.index.get_loc(kk)

            # find minimum radial distance that could be over land
            #best_fetch_loc=FT['dist_to_max'].values.argmin()

        #print(type(best_fetch_loc))
        #if ~((type(best_fetch_loc) == np.int64) | (type(best_fetch_loc) == int)):
        print('proparbility match is '+ str(prop_match) )
        if ((type(best_fetch_loc) is list) or (type(best_fetch_loc) is np.ndarray)):
            print(' best position is a list or array ')
            ain=np.arange(len(best_fetch_loc))
            best_fetch_loc=int(np.round(ain[best_fetch_loc].mean()))

        print('best position:' + str(best_fetch_loc))
        print(' ----------')

        best_fetch_index=FT.iloc[best_fetch_loc].name

        best_fetch=FT.iloc[best_fetch_loc]
        best_CID=FT.iloc[best_fetch_loc]['CID']
        best_fetchID=FT.iloc[best_fetch_loc]['fetchID']

        best_fetch_evolution=FT[(FT['CID'] == best_CID) & (FT['fetchID'] == best_fetchID) ]
        best_fetch_evolution.reset_index(inplace=True)

        #return best_fetch, best_fetch_ID, best_fetch_evolution

        if prop_match:
            tmask=best_fetch_evolution['timestamp']==best_fetch['timestamp']
            tmask.name='propability_match'

            b=best_fetch_evolution.T
            b.loc['propability_match']=tmask
            best_fetch_evolution=b.T
            #best_fetch_evolution.T['propability_match']=tmask

        else:
            tmask=best_fetch_evolution['timestamp']==best_fetch['timestamp']
            tmask.name='distance_match'

            b=best_fetch_evolution.T
            b.loc['distance_match']=tmask
            best_fetch_evolution=b.T
            #best_fetch_evolution.T['propability_match']=tmask

        ## ---------------------------------------
        # additional derive coastal match
        if sum(FT['combined_propability_coast'] !=0) == 0:
            coast_prop_match=False

            coast_best_CID=None
            coast_best_fetch_index=None
            coast_best_fetch_evolution=None

        else:
            coast_prop_match=True
            coast_best_fetch_loc=FT['combined_propability_coast'].values.argmax()

            #print(coast_best_fetch_loc)
            coast_best_fetch_index=FT.iloc[coast_best_fetch_loc].name

            coast_best_fetch=FT.iloc[coast_best_fetch_loc]
            coast_best_CID=FT.iloc[coast_best_fetch_loc]['CID']
            coast_best_fetchID=FT.iloc[coast_best_fetch_loc]['fetchID']

            coast_best_fetch_evolution=FT[(FT['CID'] == coast_best_CID) & (FT['fetchID'] == coast_best_fetchID) ]
            coast_best_fetch_evolution.reset_index(inplace=True)

            #return best_fetch, best_fetch_ID, best_fetch_evolution

            tmask=coast_best_fetch_evolution['timestamp']==coast_best_fetch['timestamp']
            tmask.name='coast_propability_match'

            b=coast_best_fetch_evolution.T
            b.loc['coast_propability_match']=tmask
            coast_best_fetch_evolution=b.T
            #best_fetch_evolution.T['propability_match']=tmask


        coast_dict={'best_CID':coast_best_CID ,
                    'best_fetch_ID':coast_best_fetch_index,
                    'best_fetch_evolution':coast_best_fetch_evolution,
                    'match':coast_prop_match}

        return best_CID, best_fetch_index, best_fetch_evolution, prop_match, coast_dict

def generate_radial_coords(trackdict, station_pos):

    for CID, Track in trackdict.iteritems():
        # calculate radial distance to Station.
        Track.data['Radial_to_S']    =Track.data.apply(lambda row:   M_geo.haversine(   station_pos[0],   station_pos[1], row['long'], row['lat'] ) *1000.0    , axis=1) # Haversine formula returns km
        Track.data['Angle_from_S']   =Track.data.apply(lambda row:   M_geo.bearing(   station_pos[0],   station_pos[1], row['long'], row['lat'] )    , axis=1) # Bering in degree deviation from north

        trackdict[CID]=Track

    return trackdict

def plot_matched_fetches(sm_pdf, best_fetch, rqlims=(0.001, 0.999), tqlim=(.01,.99), col=None, no_limits=False):

    def create_weight_single(x):
        #size_list=list(pair1['x']['data_peak'][size_var])/np.std((pair1['x']['data_peak'][size_var])) + list(pair1['y']['data_peak'][size_var])/np.std((pair1['y']['data_peak'][size_var]))
        size_list=  - 10*np.log(list(x))
        size_list+= - size_list.min()
        size_list*= 1 / 2.0
        return size_list

    from matplotlib import colors

    F=M.figure_axis_xy(x_size=5,  y_size=5, view_scale=.5)

    plt.suptitle(sm_pdf.ID +' | Intersect-Slope PDF \n' +
                'Fetched and Match' , y=1.06)
    #MT.stats_format(sm_pdf.data)

    cmap= plt.cm.Greys
    sm_pdf.clevs=np.arange(0, 1, .05)
    plt.pcolormesh(sm_pdf.data['time'], sm_pdf.data['radius']/1000.0, sm_pdf.data['fetchPDF'], norm=colors.LogNorm(),cmap = cmap, alpha=0.5)
    #plt.contourf(sm_pdf.data['time'], sm_pdf.data['radius']/1000.0, sm_pdf.data['fetchPDF'],cmap = cmap, alpha=0.5)

    leadsize=create_weight_single(sm_pdf.AllFetches['non_dim_fetch'])*5

    propper_fetches=(sm_pdf.AllFetches['combined_propability']  != 0) & (sm_pdf.AllFetches['propability']  >= 1e-3)
    other_fetches=~propper_fetches#sm_pdf.AllFetches['combined_propability']  == 0
    plt.scatter( list(sm_pdf.AllFetches[other_fetches]['timestamp'].astype('M8[m]') ) ,list(sm_pdf.AllFetches[other_fetches]['rad_dist']/1000.0), s= leadsize[other_fetches.tolist()], color=col.aug2, alpha=0.4, zorder=4)

    # select fetches with propabitliy
    #propper_fetches=(sm_pdf.AllFetches['combined_propability']  != 0) & (sm_pdf.AllFetches['propability']  >= 1e-2)
    plt.scatter( list(sm_pdf.AllFetches[propper_fetches]['timestamp'].astype('M8[m]') ) ,list(sm_pdf.AllFetches[propper_fetches]['rad_dist']/1000.0), s= leadsize[propper_fetches.tolist()], color=col.lead1, alpha=0.8, zorder=4)

    best_fetch_mask=(sm_pdf.AllFetches.index == list(best_fetch['index'])[0]) & (sm_pdf.AllFetches['timestamp'] == list(best_fetch['timestamp'])[0])

    plt.scatter( list(best_fetch['timestamp'].astype('M8[m]')) ,list(best_fetch['rad_dist']/1000.0), s= leadsize[best_fetch_mask]*3 ,marker='v' , color=col.lead2, zorder=5)

    plt.xlabel('Time (6 hours)')
    plt.ylabel('Radial Distance (km)')

    ax=plt.gca()
    ax.xaxis_date()
    Month = dates.MonthLocator()
    Day = dates.DayLocator(interval=5)#bymonthday=range(1,32)
    Hour = dates.HourLocator(interval=6)#bymonthday=range(1,32)

    dfmt = dates.DateFormatter('%y-%m-%dT%H:%M')


    ax.xaxis.set_major_locator(Day)
    ax.xaxis.set_major_formatter(dfmt)
    ax.xaxis.set_minor_locator(Hour)

    # Set both ticks to be outside
    ax.tick_params(which = 'both', direction = 'out')
    ax.tick_params('both', length=6, width=1, which='major')
    ax.tick_params('both', length=3, width=1, which='minor')

    if no_limits is False:
        lims=sm_pdf.define_erros(radial_quantile=rqlims ,time_quantile=tqlim, for_plot=True)
        #print(lims)
        plt.ylim( lims['r0'][0]/1000.0 , lims['r0'][1]/1000.0 )
        plt.xlim( [i.astype('M8[m]').item() for i  in lims['t0']] )

    return F
