class plot_spectrogram(object):

            def __init__(self,time,fs, data,clevs=None, sample_unit=None, data_unit=None,
                        ylim=None, time_unit=None, cmap=None):
                import matplotlib.pyplot as plt
                import numpy as np
                self.fs=fs[1:]
                self.time=time
                self.data=data[:,1:]
                self.clevs=clevs
                self.sample_unit=sample_unit if sample_unit is not None else 'df'

                self.data_unit=data_unit if data_unit is not None else 'X'
                self.time_unit=time_unit if time_unit is not None else 'dt'

                self.cmap=cmap if cmap is not None else plt.cm.ocean_r
                self.ylim=ylim if ylim is not None else [fs[0],fs[-1]]

            def loglog(self):
                self.F=figure_axis_xy(fig_scale=2)

                plt.loglog(self.fs[1:],(self.Xdata[1:]))

                plt.ylabel(('|X|^2/f (' + self.data_unit + '^2/' + self.sample_unit+ ')'))
                plt.xlabel(('f  (' + self.sample_unit+ ')'))
                plt.xlim(self.fs[1] ,self.fs[-1])

                self.F.make_clear()
                plt.grid()

            def linear(self):
                self.F=figure_axis_xy(10,4,fig_scale=2)
                dd=10*np.log10(self.data[:-2,:]).T

                self.clevs=self.clevs if self.clevs is not None else clevels(dd)
                self.F.ax.set_yscale("log", nonposy='clip')
                tt = self.time.astype(DT.datetime)

                self.cs=plt.contourf(tt[:-2], self.fs[:],dd, self.clevs,cmap=self.cmap)
                #self.cs=plt.pcolormesh(self.time[:-2], self.fs[:],dd,cmap=self.cmap,shading='gouraud')
                print(self.clevs)
                plt.ylabel(('Power db(' + self.data_unit + '^2/' + self.sample_unit+ ')'))
                plt.xlabel(('f  (' + self.sample_unit+ ')'))
                self.cbar= plt.colorbar(self.cs,pad=0.01)#, Location='right')#
                self.cbar.ax.aspect=100
                self.cbar.outline.set_linewidth(0)
                self.cbar.set_label('('+self.data_unit+')')


                ax = plt.gca()
                #Set y-lim
                ax.set_ylim(self.ylim[0], self.ylim[1])

                #format X-Axis
                ax.xaxis_date()
                Month = dates.MonthLocator()
                Day = dates.DayLocator(interval=5)#bymonthday=range(1,32)
                dfmt = dates.DateFormatter('%y-%b')

                ax.xaxis.set_major_locator(Month)
                ax.xaxis.set_major_formatter(dfmt)
                ax.xaxis.set_minor_locator(Day)

                # Set both ticks to be outside
                ax.tick_params(which = 'both', direction = 'out')
                ax.tick_params('both', length=6, width=1, which='major')
                ax.tick_params('both', length=3, width=1, which='minor')

                # Make grid white
                ax.grid()
                gridlines = ax.get_xgridlines() + ax.get_ygridlines()

                for line in gridlines:
                    line.set_color('white')
                    #line.set_linestyle('-')

            def power(self,  anomalie=False):
                self.F=figure_axis_xy(10,4,fig_scale=2)
                dd=10*np.log10(self.data[:-1,:])

                if anomalie is True:
                    dd_tmp=dd.mean(axis=0).repeat(self.time.size-1)
                    dd=dd- dd_tmp.reshape(self.fs.size,self.time.size-1).T
                    dd=dd

                self.clevs=self.clevs if self.clevs is not None else clevels(dd)
                self.F.ax.set_yscale("log", nonposy='clip')
                tt = self.time.astype(DT.datetime)

                print(tt[:-1].shape, self.fs[:].shape,dd.T.shape)
                self.cs=plt.contourf(tt[:-1], self.fs[:],dd.T, self.clevs,cmap=self.cmap)
                self.x=np.arange(0,tt[:-1].size)
                #self.cs=plt.pcolormesh(self.time[:-2], self.fs[:],dd,cmap=self.cmap,shading='gouraud')
                print(self.clevs)
                plt.xlabel('Time')
                plt.ylabel(('f  (' + self.sample_unit+ ')'))
                self.cbar= plt.colorbar(self.cs,pad=0.01)#, Location='right')#
                self.cbar.ax.aspect=100
                self.cbar.outline.set_linewidth(0)
                self.cbar.set_label('Power db(' + self.data_unit + '^2/f )')


                ax = plt.gca()
                #Set y-lim
                ax.set_ylim(self.ylim[0], self.ylim[1])

                #format X-Axis
                ax.xaxis_date()
                Month = dates.MonthLocator()
                Day = dates.DayLocator(interval=5)#bymonthday=range(1,32)
                dfmt = dates.DateFormatter('%y-%b')

                ax.xaxis.set_major_locator(Month)
                ax.xaxis.set_major_formatter(dfmt)
                ax.xaxis.set_minor_locator(Day)

                # Set both ticks to be outside
                ax.tick_params(which = 'both', direction = 'out')
                ax.tick_params('both', length=6, width=1, which='major')
                ax.tick_params('both', length=3, width=1, which='minor')

                # Make grid white
                ax.grid()
                gridlines = ax.get_xgridlines() + ax.get_ygridlines()

                for line in gridlines:
                    line.set_color('white')
                    line.set_linestyle('--')
            def imshow(self, shading=None, downscale_fac=None, anomalie=False, downscale_type=None, fig_size=None, ax=None,cbar=True):
                nopower=True
                self.power_imshow(shading, downscale_fac , anomalie, downscale_type, fig_size , nopower, ax=ax, cbar=cbar)
                self.cbar.set_label('Power (' + self.data_unit + '^2/f )')
            def power_imshow(self, shading=None, downscale_fac=None, anomalie=False,
                            downscale_type=None, fig_size=None , nopower=False, ax=None, cbar=True):

                import matplotlib.pyplot as plt
                import datetime as DT
                import matplotlib.colors as colors
                from matplotlib import dates
                import time
                import scipy.signal as signal
                import matplotlib.ticker as ticker
                import numpy as np
                from .tools import stats_format

                shading='gouraud' if shading is True else 'flat'
                fig_size=[10,4] if fig_size is None else fig_size
                if ax:
                    assert type(ax) is tuple, "put ax as tuple ax=(ax,F)"
                    self.F=ax[1]
                    ax_local=ax[0]
                else:
                    self.F=figure_axis_xy(fig_size[0], fig_size[1], fig_scale=2)
                    ax_local=self.F.ax

                if nopower is True:
                    dd=self.data
                else:
                    dd=10*np.log10(self.data[:-1,:])

                if anomalie is True:
                    dd_tmp=dd.mean(axis=0).repeat(self.time.size-1)
                    dd=dd- dd_tmp.reshape(self.fs.size,self.time.size-1).T

                self.clevs=self.clevs if self.clevs is not None else clevels(dd)

                norm = colors.BoundaryNorm(boundaries=self.clevs, ncolors=256)

                tt = self.time

                #tvec=np.arange(0,tt.size,1)
                ax_local.set_yscale("log", nonposy='clip')

                if downscale_fac is not None:
                    if downscale_type =='inter':
                        fn=[]
                        for yr in np.arange(0,self.fs.size,downscale_fac):
                            fn.append(np.mean(self.fs[yr:yr+downscale_fac]))
                    else:

                        ddn=np.empty((self.time.size-1))
                        fsn_p=gen_log_space(self.fs.size,int(np.round(self.fs.size/downscale_fac)))
                        fsn_p_run=np.append(fsn_p,fsn_p[-1])
                        dd=dd.T
                        #print(ddn.shape, fsn_p.shape)
                        for fr in np.arange(0,fsn_p.size,1):
                            #print(np.mean(dd[fsn_p[fr]:fsn_p[fr+1], :],axis=0).shape)

                            ddn=np.vstack((ddn, np.mean(dd[fsn_p_run[fr]:fsn_p_run[fr+1], :],axis=0)))
                        ddn=np.delete(ddn, 0,0)
                        #print(ddn.shape)
                        dd2=ddn
                        fn=self.fs[fsn_p]

                        if nopower is True:
                            tt=tt
                        else:
                            tt=tt[:-1]
                        #print(dd2.shape, fn.shape, tt.shape)

                else:
                    if nopower is True:
                        tt=tt
                    else:
                        tt=tt[:-1]
                    dd2=dd.T
                    fn=self.fs

                if isinstance(tt[0], np.datetime64):
                    print('time axis is numpy.datetime64, converted to number for plotting')
                    ttt=dates.date2num(tt.astype(DT.datetime))
                    #print(ttt)
                elif isinstance(tt[0], np.timedelta64):
                    print('time axis is numpy.timedelta64, converted to number for plotting')
                    #print(tt)
                    ttt=tt
                else:
                    #print(type(tt[0]))
                    #print(tt)

                    print('time axis is not converted')
                    ttt=tt

                stats_format(dd2)
                self.cs=plt.pcolormesh(ttt,fn ,dd2,cmap=self.cmap , norm=norm,
                shading=shading)#, origin='lower',aspect='auto',
                    #interpolation='none',
                    #extent=[tvec.min(),tvec.max(),self.fs.min(),self.fs.max()])

                #self.F.ax.set_yscale("log", nonposy='clip')
                #self.cs=plt.pcolormesh(self.time[:-2], self.fs[:],dd,cmap=self.cmap,shading='gouraud')
                #print(self.clevs)
                plt.ylabel(('f  (' + self.sample_unit+ ')'))
                if cbar is True:
                    self.cbar= plt.colorbar(self.cs,pad=0.01)#, Location='right')#
                    self.cbar.ax.aspect=100
                    self.cbar.outline.set_linewidth(0)
                    self.cbar.set_label('Power db(' + self.data_unit + '^2/f )')

                ax =ax_local#plt.gca()
                if isinstance(tt[0], np.datetime64):
                    plt.xlabel('Time')
                    #Set y-lim
                    ax.set_ylim(self.ylim[0], self.ylim[1])
                    ax.set_xlim(ttt[0], ttt[-1])

                    #format X-Axis
                    #ax.xaxis_date()
                    Month = dates.MonthLocator()
                    Day = dates.DayLocator(interval=5)#bymonthday=range(1,32)
                    dfmt = dates.DateFormatter('%b/%y')

                    ax.xaxis.set_major_locator(Month)
                    ax.xaxis.set_major_formatter(dfmt)
                    ax.xaxis.set_minor_locator(Day)
                elif isinstance(tt[0], np.float64):
                    plt.xlabel('Time')
                    #Set y-lim
                    ax.set_ylim(self.ylim[0], self.ylim[1])
                    ax.set_xlim(ttt[0], ttt[-1])

                    #format X-Axis
                    #ax.xaxis_date()
                    Month = dates.MonthLocator()
                    Day = dates.DayLocator(interval=5)#bymonthday=range(1,32)
                    dfmt = dates.DateFormatter('%b/%y')

                    ax.xaxis.set_major_locator(Month)
                    ax.xaxis.set_major_formatter(dfmt)
                    ax.xaxis.set_minor_locator(Day)
                else:
                    plt.xlabel('Time (' + self.time_unit+ ')')
                    ax.set_ylim(self.ylim[0], self.ylim[1])
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

                    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

                # Set both ticks to be outside
                ax.tick_params(which = 'both', direction = 'out')
                ax.tick_params('both', length=6, width=1, which='major')
                ax.tick_params('both', length=3, width=1, which='minor')

                # Make grid white
                ax.grid()
                self.ax=ax
                gridlines = ax.get_xgridlines() + ax.get_ygridlines()

                for line in gridlines:
                    line.set_color('white')
                    #line.set_linestyle('-')

                self.x=ttt


            def linear_imshow(self, shading=None, downscale_fac=None, anomalie=False, downscale_type=None, fig_size=None , nopower=False, ax=None):
                import matplotlib.colors as colors
                from matplotlib import dates
                import time
                import scipy.signal as signal
                import matplotlib.ticker as ticker
                import numpy as np

                shading='gouraud' if shading is True else 'flat'
                fig_size=[10,4] if fig_size is None else fig_size
                if ax:
                    assert type(ax) is tuple, "put ax as tuple ax=(ax,F)"
                    self.F=ax[1]
                    ax_local=ax[0]
                else:
                    self.F=figure_axis_xy(fig_size[0], fig_size[1], fig_scale=2)
                    ax_local=self.F.ax

                if nopower is True:
                    dd=self.data
                else:
                    dd=10*np.log10(self.data[:-1,:])

                if anomalie is True:
                    dd_tmp=dd.mean(axis=0).repeat(self.time.size-1)
                    dd=dd- dd_tmp.reshape(self.fs.size,self.time.size-1).T

                self.clevs=self.clevs if self.clevs is not None else clevels(dd)

                norm = colors.BoundaryNorm(boundaries=self.clevs, ncolors=256)

                tt = self.time

                #tvec=np.arange(0,tt.size,1)
                self.F.ax.set_yscale("log", nonposy='clip')

                if downscale_fac is not None:
                    if downscale_type =='inter':
                        fn=[]
                        for yr in np.arange(0,self.fs.size,downscale_fac):
                            fn.append(np.mean(self.fs[yr:yr+downscale_fac]))
                    else:

                        ddn=np.empty((self.time.size-1))
                        fsn_p=gen_log_space(self.fs.size,int(np.round(self.fs.size/downscale_fac)))
                        fsn_p_run=np.append(fsn_p,fsn_p[-1])
                        dd=dd.T
                        #print(ddn.shape, fsn_p.shape)
                        for fr in np.arange(0,fsn_p.size,1):
                            #print(np.mean(dd[fsn_p[fr]:fsn_p[fr+1], :],axis=0).shape)

                            ddn=np.vstack((ddn, np.mean(dd[fsn_p_run[fr]:fsn_p_run[fr+1], :],axis=0)))
                        ddn=np.delete(ddn, 0,0)
                        #print(ddn.shape)
                        dd2=ddn
                        fn=self.fs[fsn_p]

                        if nopower is True:
                            tt=tt
                        else:
                            tt=tt[:-1]
                        #print(dd2.shape, fn.shape, tt.shape)

                else:
                    if nopower is True:
                        tt=tt
                    else:
                        tt=tt[:-1]
                    dd2=dd.T
                    fn=self.fs

                if isinstance(tt[0], np.datetime64):
                    print('numpy.datetime64')
                    ttt=dates.date2num(tt.astype(DT.datetime))
                    #print(ttt)
                elif isinstance(tt[0], np.timedelta64):
                    print('numpy.timedelta64')
                    #print(tt)
                    ttt=tt
                else:
                    #print(type(tt[0]))
                    #print(tt)

                    print('something else')
                    ttt=tt


                self.cs=plt.pcolormesh(ttt,fn ,dd2,cmap=self.cmap , norm=norm,
                shading=shading)#, origin='lower',aspect='auto',
                    #interpolation='none',
                    #extent=[tvec.min(),tvec.max(),self.fs.min(),self.fs.max()])

                #self.F.ax.set_yscale("log", nonposy='clip')
                #self.cs=plt.pcolormesh(self.time[:-2], self.fs[:],dd,cmap=self.cmap,shading='gouraud')
                #print(self.clevs)


                plt.ylabel(('f  (' + self.sample_unit+ ')'))
                self.cbar= plt.colorbar(self.cs,pad=0.01)#, Location='right')#
                self.cbar.ax.aspect=100
                self.cbar.outline.set_linewidth(0)
                self.cbar.set_label('Power db (' + self.data_unit + '^2/f )')

                ax = plt.gca()
                if isinstance(tt[0], np.datetime64):
                    plt.xlabel('Time')
                    #Set y-lim
                    ax.set_ylim(self.ylim[0], self.ylim[1])
                    ax.set_xlim(ttt[0], ttt[-1])

                    #format X-Axis
                    #ax.xaxis_date()
                    Month = dates.MonthLocator()
                    Day = dates.DayLocator(interval=5)#bymonthday=range(1,32)
                    dfmt = dates.DateFormatter('%b/%y')

                    ax.xaxis.set_major_locator(Month)
                    ax.xaxis.set_major_formatter(dfmt)
                    ax.xaxis.set_minor_locator(Day)
                else:
                    plt.xlabel('Time (' + self.time_unit+ ')')
                    ax.set_ylim(self.ylim[0], self.ylim[1])
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

                    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

                # Set both ticks to be outside
                ax.tick_params(which = 'both', direction = 'out')
                ax.tick_params('both', length=6, width=1, which='major')
                ax.tick_params('both', length=3, width=1, which='minor')

                # Make grid white
                ax.grid()
                self.ax=ax
                gridlines = ax.get_xgridlines() + ax.get_ygridlines()

                for line in gridlines:
                    line.set_color('white')
                    #line.set_linestyle('-')

                self.x=np.arange(0,ttt.size+1)


            def set_xaxis_to_days(self, **kwargs):
                set_timeaxis_days(self.ax, **kwargs)


def cut_nparray(var, low, high, verbose=False):
    import numpy as np
    if low < high:
        if low < var[0]:
            if verbose:
                print('out of lower limit!')
        if high > var[-1]:
            if verbose:
                print('out of upper limit!')
                print(high ,'>', var[-1])
        return (var >=low) & (var <=high)

    elif high < low:
        if high < var[0]:
            print('limits flipped, out of lower limit!')
        if low > var[-1]:
            print('limits flipped, out of lower limit!')

        return (var >=high) & (var <=low)

    elif high == low:
        if verbose:
            print('find nearest')
        a=var-low
        return np.unravel_index(np.abs(a).argmin(),np.transpose(a.shape))

    else:
        print('error')
        return



class figure_axis_xy(object):
        """define standart  XY Plot with reduced grafics"""

        def __init__(self,x_size=None,y_size=None,view_scale=None, fig_scale=None, container=False, dpi=180):
                import matplotlib.pyplot as plt
                xsize=x_size if x_size is not None else 8
                ysize=y_size if y_size is not None else 5
                viewscale=view_scale if view_scale is not None else 0.5
                fig_scale=fig_scale if fig_scale is not None else 1
                if container:
                    self.fig=plt.figure(edgecolor='None',dpi=dpi*viewscale,figsize=(xsize*fig_scale, ysize*fig_scale),facecolor='w')
                else:
                    self.fig, self.ax=plt.subplots(num=None, figsize=(xsize*fig_scale, ysize*fig_scale), dpi=dpi*viewscale, facecolor='w', edgecolor='None')


        def make_clear_weak(self):
                #turn off axis spine to the right
                #self.fig.tight_layout()
                self.ax.spines['right'].set_color("none")
                self.ax.yaxis.tick_left() # only ticks on the left side
                self.ax.spines['top'].set_color("none")
                self.ax.xaxis.tick_bottom() # only ticks on the left side
        def make_clear(self):
                self.make_clear_weak()

        def make_clear_strong(self):
                #turn off axis spine to the right
                #self.fig.tight_layout()
                self.ax.spines['right'].set_color("none")
                self.ax.spines['left'].set_color("none")
                self.ax.yaxis.tick_left() # only ticks on the left side
                self.ax.spines['top'].set_color("none")
                self.ax.spines['bottom'].set_color("none")
                self.ax.xaxis.tick_bottom() # only ticks on the left side

        def tight(self):
                #turn off axis spine to the right
                self.fig.tight_layout()

        def label(self, x='x',y='y',t=None):

                self.ax.set_xlabel(x)
                self.ax.set_ylabel(y)
                self.ax.set_title(t, y=1.04)

        def save(self,name=None,path=None, verbose=True):
                import datetime
                import os

                savepath=path if path is not None else os.path.join(os.path.dirname(os.path.realpath('__file__')),'plot/')
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                #os.makedirs(savepath, exist_ok=True)
                name=name if name is not None else datetime.date.today().strftime("%Y%m%d_%I%M%p")
                #print(savepath)
                #print(name)
                extension='.pdf'
                full_name= (os.path.join(savepath,name)) + extension
                #print(full_name)
                self.fig.savefig(full_name, bbox_inches='tight', format='pdf', dpi=180)
                if verbose:
                    print('save at: '+name)

        def save_pup(self,name=None,path=None, verbose=True):
                import datetime
                import re
                import os
                name=re.sub("\.", '_', name)

                savepath=path if path is not None else os.path.join(os.path.dirname(os.path.realpath('__file__')),'plot/')
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                #os.makedirs(savepath, exist_ok=True)
                name=name if name is not None else datetime.date.today().strftime("%Y%m%d_%I%M%p")
                #print(savepath)
                #print(name)
                extension='.pdf'
                full_name= (os.path.join(savepath,name)) + extension
                #print(full_name)
                self.fig.savefig(full_name, bbox_inches='tight', format='pdf', dpi=300)
                if verbose:
                    print('save at: ',full_name)

        def save_light(self,name=None,path=None, verbose=True):
                import datetime
                import os
                savepath=path if path is not None else os.path.join(os.path.dirname(os.path.realpath('__file__')),'plot/')
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                #os.makedirs(savepath, exist_ok=True)
                name=name if name is not None else datetime.date.today().strftime("%Y%m%d_%I%M%p")
                #print(savepath)
                #print(name)
                extension='.png'
                full_name= (os.path.join(savepath,name)) + extension
                #print(full_name)
                self.fig.savefig(full_name, bbox_inches='tight', format='png', dpi=180)
                if verbose:
                    print('save with: ',name)

class subplot_routines(figure_axis_xy):
        def __init__(self, ax):
            self.ax=ax

def runningmean(var, m, tailcopy=False):
    m=int(m)
    s =var.shape
    if s[0] <= 2*m:
        print('0 Dimension is smaller then averaging length')
        return
    rr=np.asarray(var)*np.nan
    #print(type(rr))
    var_range=np.arange(m,int(s[0])-m-1,1)
    #    print(var_range)
    #    print(np.isfinite(var))
    #    print(var_range[np.isfinite(var[m:int(s[0])-m-1])])
    for i in var_range[np.isfinite(var[m:int(s[0])-m-1])]:
        #rm.append(var[i-m:i+m].mean())
        rr[int(i)]=np.nanmean(var[i-m:i+m])

    if tailcopy:
    #        print('tailcopy')
        rr[0:m]=rr[m+1]
        rr[-m-1:-1]=rr[-m-2]

    return rr


def detrend(data, od=None, x=None, plot=False, verbose=False):
    # data  data that should be detrended
    #od order of polynomial
    #x  optional xaxis, otherwise equal distance is assument
    #plot True for plot
    od=0 if od is None else od

    if od == 0:
        d_detrend=data-np.nanmean(data)
        d_org=[]
        dline=[]


    elif od > 0 :
        if verbose: print('assume data is equal dist. You can define option x= if not.')

        d_org=data-np.nanmean(data)
        x=np.arange(0,d_org.size,1) if x is None else x

        #print(np.isnan(x).sum(), np.isnan(d_org).sum())
        idx = np.isfinite(x) & np.isfinite(d_org)
        px=np.polyfit(x[idx], d_org[idx], od)
        dline=np.polyval( px, x)
        d_detrend = d_org -dline

    if plot == True:
        F=figure_axis_xy(15, 5)
        if od > 0:
            plt.plot(d_org, Color='black')
            plt.plot(dline, Color='black')
        plt.plot(d_detrend, Color='r')
        F.make_clear()
        plt.grid()
        plt.legend(['org', 'line', 'normalized'])


    stats=dict()
    stats['org']=d_org
    stats['std']=np.nanstd(d_detrend)
    if od > 0:
        stats['line']=dline
        stats['polynom order']=od
        stats['polyvals']=px
    if verbose: print(stats)
    return d_detrend/np.nanstd(d_detrend) , stats

def normalize(data):
    return detrend(data)[0]
def nannormalize(data):
    return ( data-np.nanmean(data) ) /np.nanstd(data)

class plot_polarspectra(object):
        def __init__(self,f, thetas, data,unit=None, data_type='fraction' ,lims=None,  verbose=False):

            self.f=f
            self.data=data
            self.thetas=thetas

            #self.sample_unit=sample_unit if sample_unit is not None else 'df'
            self.unit=unit if unit is not None else 'X'

            # decided on freq limit
            lims=[self.f.min(),self.f.max()] if lims is None else lims
            self.lims=lims

            freq_sel_bool=M.cut_nparray(self.f,1./lims[1], 1./lims[0])

            self.min=np.nanmin(data[freq_sel_bool,:])#*0.5e-17
            self.max=np.nanmax(data[freq_sel_bool,:])
            if verbose:
                print(str(self.min), str(self.max) )

            self.ylabels=np.arange(10, 100, 20)
            self.data_type=data_type
            if data_type == 'fraction':
                self.clevs=np.linspace(0.01, self.max*.5, 21)
            elif data_type == 'energy':
                self.ctrs_min=self.min+self.min*.05
                #self.clevs=np.linspace(self.min, self.max, 21)
                self.clevs=np.linspace(self.min+self.min*.05, self.max*.60, 21)


        def linear(self, radial_axis='period', circles =None, ax=None ):




            if ax is None:
                ax = plt.subplot(111, polar=True)
                self.title=plt.suptitle('  Polar Spectrum', y=0.95, x=0.5 , horizontalalignment='center')
            else:
                ax=ax
            ax.set_theta_direction(-1)  #left turned postive
            ax.set_theta_zero_location("N")


            #cm=plt.cm.get_cmap('Reds')
            #=plt.cm.get_cmap('PuBu')


            plt.ylim(self.lims)

            #ylabels=np.arange(10, 100, 20)
            #ylabels = ([ 10, '', 20,'', 30,'', 40])
            ax.set_yticks(self.ylabels)
            ax.set_yticklabels(' '+str(y)+ ' s' for y in self.ylabels)

            ## Set titles and colorbar
            #plt.title(STID+' | '+p + ' | '+start_date+' | '+end_date, y=1.11, horizontalalignment='left')


            grid=ax.grid(color='k', alpha=.5, linestyle='--', linewidth=.5)

            if self.data_type == 'fraction':
                cm=brewer2mpl.get_map( 'RdYlBu','Diverging', 4, reverse=True).mpl_colormap
                colorax = ax.contourf(self.thetas, 1/self.f, self.data, self.clevs, cmap=cm, zorder=1)# ,cmap=cm)#, vmin=self.ctrs_min)
            elif self.data_type == 'energy':
                cm=brewer2mpl.get_map( 'Paired','Qualitative', 8).mpl_colormap

                cm.set_under='w'
                cm.set_bad='w'
                colorax = ax.contourf(self.thetas, 1/self.f, self.data, self.clevs,cmap=cm, zorder=1)#, vmin=self.ctrs_min)
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.05)

            if circles is not None:
                theta = np.linspace(0, 2 * np.pi, 360)
                r1 =theta*0+circles[0]
                r2 =theta*0+circles[1]
                plt.plot(theta, r1,  c='red', alpha=0.6,linewidth=1, zorder=10)
                plt.plot(theta, r2,  c='red', alpha=0.6,linewidth=1, zorder=10)



            cbar = plt.colorbar(colorax, fraction=0.046, pad=0.06, orientation="horizontal")
            if self.data_type == 'fraction':
                cbar.set_label('Fraction of Energy', rotation=0, fontsize=MEDIUM_SIZE)
            elif self.data_type == 'energy':
                cbar.set_label('Energy Density ('+self.unit+')', rotation=0, fontsize=MEDIUM_SIZE)
            cbar.ax.get_yaxis().labelpad = 30
            cbar.outline.set_visible(False)
            #cbar.ticks.
            #cbar.outline.clipbox
            degrange = range(0,360,30)

            lines, labels = plt.thetagrids(degrange, labels=None, frac = 1.07)

            for line in lines:
                #L=line.get_xgridlines
                line.set_linewidth(5)
                #line.set_linestyle(':')

            ax.spines['polar'].set_color("none")
            ax.set_rlabel_position(87)
            self.ax=ax

def set_timeaxis_days(ax, int1=1, int2=2, bymonthday=None):
    # int1 interval of the major (labeld) days
    # int2 intercal of the minar (only ticks) days
    from matplotlib import dates
    bymonthday=bymonthday if bymonthday is not None else range(1,32)
    Month = dates.MonthLocator()
    Month_dfmt = dates.DateFormatter('%b/%y')
    Day = dates.DayLocator(interval=int2, bymonthday=bymonthday)#bymonthday=range(1,32)
    Day_dfmt = dates.DateFormatter('%d')
    Day2 = dates.DayLocator(interval=int1, bymonthday=bymonthday)#bymonthday=range(1,32)
    Day2_dfmt = dates.DateFormatter('')

    ax.xaxis.set_major_locator(Day)
    ax.xaxis.set_major_formatter(Day_dfmt)
    ax.xaxis.set_minor_locator(Day2)
    ax.xaxis.set_minor_formatter(Day2_dfmt)


def log_power(data):
    return 10*np.log10(data)

def echo_dt(a, as_string=False):
    string=str(a.astype('timedelta64[s]'))+'/'+str(a.astype('timedelta64[m]'))+'/'+str(a.astype('timedelta64[h]'))+'/'+str(a.astype('timedelta64[D]'))
    #print(string)
    if as_string:
        return string
    else:
        print(string)
def easy_dtstr(a):
    if a.astype('timedelta64[s]') < np.timedelta64(60,'s'):
        return str(a.astype('timedelta64[s]'))
    elif a.astype('timedelta64[m]') < np.timedelta64(60,'m'):
        return str(a.astype('timedelta64[m]'))
    elif a.astype('timedelta64[h]') < np.timedelta64(24,'h'):
        return str(a.astype('timedelta64[h]'))
    elif a.astype('timedelta64[D]') < np.timedelta64(365,'D'):
        return str(a.astype('timedelta64[D]'))
    elif a.astype('timedelta64[M]') < np.timedelta64(24,'M'):
        return str(a.astype('timedelta64[M]'))
    else:
        return str(a.astype('timedelta64[Y]'))
def clevels(data, dstep=None):
    import numpy as np
    dstep=dstep if dstep is not None else 21
    mmax=np.ceil(np.nanmax(data))
    mmin=np.floor(np.nanmin(data))
    clim=np.linspace(mmin,mmax,dstep)
    return clim

def save_anyfig(fig,name=None,path=None):
                import datetime

                savepath=path if path is not None else os.path.join(os.path.dirname(os.path.realpath('__file__')),'plot/')
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                name=name if name is not None else datetime.date.today().strftime("%Y%m%d_%I%M%p")
                #print(savepath)
                #print(name)
                extension='.png'
                full_name= (os.path.join(savepath,name)) + extension
                #print(full_name)
                fig.savefig(full_name, bbox_inches='tight', format='png', dpi=180)
                print('save at: ',full_name)



def find_max_ts(data_org, threshold=None, jump=None, smooth=True, spreed=None, plot=False, nocopy=False, verbose=True):

    """
    This function finds local minima in a 1-dimensional array by asking where the gradient of the data changes sign

    input:
        data_org    data array, like a time series or so. (even or uneven distributed?)

        threshold   (None) Only concider data above a threshold
        jump        (None) minimal distance in data points two minima are allowed to be appart .
        smooth      (True) if True smoothing the time series using a running mean
            spreed      (None) the with of the running mean. If None its set to 2 data point.

        plot        (False) if True it plots somethinhe (not implemented jet)
        nocopy      if True, the time series is not coyed and altered in this function (be cause python is updatedingh links)

        verbose     prints statements if True

    returns:
        jump is None:       tuple with  (index, data, data[index])
                index           index points of maxima,
                data            the modified 1d data array
                data[index]     values of the maxima points

        jump is not None:   tuple with  (index_reduced, data, data[index], index)
                index_reduced   index points of maxima according to jump condition
                data            the modified 1d data array
                data[index]     values of the maxima points
                index           all indexes without the jump condition

    """
    if nocopy:
        data=data_org
    else:
        data=np.copy(data_org)
    spreed=2 if spreed is None else spreed

    if smooth is True:
        data=runningmean(data,spreed)
    #print(threshold is not None and threshold > np.nanmin(data))
    #if threshold is not None and numpy.ndarray

    if threshold is not None and threshold > np.nanmin(data):
        data[np.isnan(data)]=0
        data[data<threshold]=0#np.nan
    else:
        #print(type(data.astype('float64')))
        data[np.isnan(data)]=0

    index=np.where(np.diff(np.sign(np.gradient(data)))== -2)[0]+1

    if index.size == 0:
        index=np.where(data==data.max())[0]

    index2=list()
    for i in index:

        adjustment=data[i-1:i+2].argmax()-1
        if adjustment != 0:
            #print(str(i) +' adjusted by ' + str(adjustment))
            index2.append(i+data[i-1:i+2].argmax()-1)
        else:
            index2.append(i)

    index=index2

    if jump is None:
        if verbose:
            print('index, data, edit ts (index)')
        return index, data, data[index]
    else:
        c=np.diff(index)
        b=[]
        i=0
        while i < index.size-1:
            #  print(i, index.size-2, c[i:])
            if c[i] < jump:
                if i >= index.size-2:
                    nc=1
                elif sum(c[i:] >= jump) == 0:
                    nc=c[i:].size
                else:
                    # print(np.nonzero(c[i:] >= jump))
                    nc=np.nonzero(c[i:] >= jump)[0][0]
                b=np.append(b, np.round(np.mean(index[i:i+nc+1]))).astype(int)
                #print(nc, index[i:i+nc+1], ' new', np.round(np.mean(index[i:i+nc+1])))
                i=i+nc+1
            else:
                b=np.append(b, index[i]).astype(int)
                #print(' ', index[i], ' new', index[i])
                i=i+1
        if verbose:
            print('index, edited ts, edit ts (index), org_index')

        return b, data, data[b], index
