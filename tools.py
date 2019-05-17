
import numpy as np

import os
import pickle

def dt_form_timestamp(timestamp, unit=None):
    unit='h' if unit is None else unit
    return (timestamp[1]-timestamp[0]).astype('m8['+unit+']')


def tick_formatter(a, interval=2, rounder=2, expt_flag=True):

    O=int(np.log10(a.max()))
    fact=10**(O-1)
    b=np.round(a/fact, rounder+1)*fact
    ticklabels=[' ' for i in range(len(b))]
    N=int(np.ceil(len(b)/interval))

    tt=np.arange(0,len(b),interval)

    for t in tt:
        if expt_flag:
            ticklabels[int(t)]='{:.2e}'.format(b[t])
        else:

            ticklabels[int(t)]=format(b[t], '.2f').rstrip('0').rstrip('.')#'{:.{2}f}'.format(b[t])

    #ticks=a
    return ticklabels, b


def freq_lim_string(low, high):
	a='%2.1e' % low
	b='%2.1e' % high

	return a[0:3] +'-'+ b +' Hz'

def mkdirs_r(path):
    if not os.path.exists(path):
                os.makedirs(path)

def check_year(inputstr, yearstring):
    a=np.datetime64(inputstr).astype(object).year
    ref=np.datetime64(yearstring).astype(object).year
    if a == ref:
        return True
    else:
        return False

def datetime64_to_sec(d):
    return d.astype('M8[s]').astype('float')
def datetime64_to_day(d):
    return d.astype('M8[D]').astype('float')

def float_plot_time_to_sec(pp):
    return np.datetime64(dates.num2date(pp)).astype('M8[s]').astype('float')
def float_plot_time_to_dt64(pp):
    return np.datetime64(dates.num2date(pp)).astype('M8[s]')

def sec_to_dt64(pp):
    return pp.astype('M8[s]')

def sec_to_float_plot(pp):
    from matplotlib import dates
    import datetime as DT
    return dates.date2num(pp.astype('M8[s]').astype(DT.datetime))
def sec_to_float_plot_single(pp):
    from matplotlib import dates
    import datetime as DT
    return dates.date2num(np.datetime64(int(pp), 's').astype('M8[s]').astype(DT.datetime))

def fake_2d_data(verbose=True, timeaxis=False):
    import matplotlib.pyplot as plt

    x=np.arange(0,100,1)
    y=np.arange(0,40,1)
    XX, YY= np.meshgrid(x,y)

    mu=x.size/2
    sigma=x.size/5
    z2= 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (XX - mu)**2 / (2 * sigma**2) )
    z2=z2/z2.max()

    mu=y.size/2
    sigma=y.size/5
    z3= 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (YY - mu)**2 / (2 * sigma**2) )
    z3=z3/z3.max()
    if verbose:
        print('x' , x.shape)
        print('y' , y.shape)
        print('z' , z3.shape)

        plt.contourf(x, y,z2/2+z3/2)
        plt.colorbar()
        plt.axis('scaled')
        plt.show()


    return x, y, z3

def pickle_save(name, path, data, verbose=True):
    if not os.path.exists(path):
        os.makedirs(path)
    full_name= (os.path.join(path,name+ '.npy'))


    with open(full_name, 'wb') as f2:
        pickle.dump(data, f2)
    if verbose:
        print('save at: ',full_name)

def pickle_load(name, path, verbose=True):
    #if not os.path.exists(path):
    #    os.makedirs(path)
    full_name= (os.path.join(path,name+ '.npy'))

    with open(full_name, 'r') as f:
        data=pickle.load(f)

    if verbose:
        print('load from: ',full_name)
    return data


def json_save(name, path, data, verbose=False, return_name=False):
    import json
    #import simplejson as json
    if not os.path.exists(path):
        os.makedirs(path)
    full_name_root=os.path.join(path,name)
    full_name= (os.path.join(full_name_root+ '.json'))
    with open(full_name, 'w') as outfile:
        json.dumps(data, outfile)
    if verbose:
        print('save at: ',full_name)
    if return_name:
        return full_name_root
    else:
        return


def json_load(name, path, verbose=False):
    import json
    full_name= (os.path.join(path,name+ '.json'))

    with open(full_name, 'r') as ifile:
        data=json.load(ifile)
    if verbose:
        print('loaded from: ',full_name)
    return data

def h5_load(name, path, verbose=False):
    import pandas as pd
    full_name= (os.path.join(path,name+ '.h5'))
    data=pd.read_hdf(full_name)
    #with pd.HDFStore(full_name) as data:
    #    data = pd.HDFStore(path+self.ID+'.h5')
    #    data.close()
    return data

def h5_load_v2(name, path, verbose=False):
    import h5py

    h5f = h5py.File(path + name + '.h5','r')
    if verbose:
        print(h5f.keys())

    data_dict=dict()
    for k, I in h5f.iteritems():
        data_dict[k] =I[:]

    h5f.close()

    return data_dict

def h5_save(name, path, data_dict, verbose=False, mode='w'):
    import h5py

    mode = 'w' if mode is None else mode
    if not os.path.exists(path):
        os.makedirs(path)

    full_name= (os.path.join(path,name+ '.h5'))
    store = h5py.File(full_name, mode)



    for k, I in list(data_dict.items()):
            store[k]=I

    store.close()

    if verbose:
        print('saved at: ' +full_name)

def h5_save(name, path, data_dict, verbose=False, mode='w'):
    import h5py

    mode = 'w' if mode is None else mode
    if not os.path.exists(path):
        os.makedirs(path)

    full_name= (os.path.join(path,name+ '.h5'))
    store = h5py.File(full_name, mode)



    for k, I in list(data_dict.items()):
            store[k]=I

    store.close()

    if verbose:
        print('saved at: ' +full_name)


def load_pandas_table_dict(name , save_path):
    import warnings
    from pandas import HDFStore
    from pandas.io.pytables import PerformanceWarning
    warnings.filterwarnings('ignore',category=PerformanceWarning)

    return_dict=dict()
    with HDFStore(save_path+'/'+name+'.h5') as store:
        #print(store)
        #print(store.keys())
        for k in store.keys():
            return_dict[k[1:]]=store.get(k)

    return return_dict

def save_pandas_table(table_dict, ID , save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    import warnings
    from pandas import HDFStore
    from pandas.io.pytables import PerformanceWarning
    warnings.filterwarnings('ignore',category=PerformanceWarning)

    with HDFStore(save_path+'/'+ID+'.h5') as store:
        for name,table in table_dict.iteritems():
                store[name]=table


def write_log(hist, string, verbose=False, short=True , date=True):
    import datetime as datetime
    if short:
        now = datetime.datetime.now().strftime("%Y%m%d")
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    if date:
        message='\n'+now+' '+string
    else:
        message='\n '.ljust(16)+' '+string

    if verbose== True:
        print(message)
    elif verbose == 'all':
        print(hist+message)
    return hist+message

def write_variables_log(hist, var_list, locals, verbose=False, date=False):
    import datetime as datetime

    now = datetime.datetime.now().strftime("%Y%m%d")

    var_dict=dict( (name,locals[name]) for name in var_list )
    stringg=''
    for name,I in var_dict.iteritems():
        stringg=stringg+ '\n '+name.ljust(20) + str(I)

    if date:
        message='\n'+now+' '+stringg
    else:
        message='\n '.ljust(16)+' '+stringg

    if verbose== True:
        print(message)
    elif verbose == 'all':
        print(hist+message)
    return hist+message


def save_log_txt(name, path, hist,verbose=False):
    if not os.path.exists(path):
        os.makedirs(path)
    full_name= (os.path.join(path,name+ '.hist.txt'))
    with open(full_name, 'w') as ifile:
        ifile.write(str(hist))
    if verbose:
        print('saved at: ',full_name)

def load_log_txt(name, path):
    import glob
    hist_file=name#'DR01.LHN.stormdetect.A02_geometry_cut_storm.hist.txt' #ID.string+'.A02**.txt'
    f=[]
    for h in glob.glob(os.path.join(path,hist_file)):
        f.append(open(h, 'r').read())
    return '\n'.join(f)

def shape(a):
	for i in a:
		print(i.shape)

def find_O(a, case='round'):
    if case=='round':
        for k in np.logspace(0,24,25):
            if np.ceil(a/k) == 1:
                return k
                break
    elif case=='floor':
        for k in np.logspace(0,24,25):
            if np.ceil(a/k) == 1:
                return k
                break

    elif case=='ceil':
        for k in np.logspace(0,24,25):
            if np.ceil(a/k) == 1:
                return k
                break
    else:
        raise Warning('no propper case')

def stats(a):
	print('shape' , a.shape)
	print('Nans',np.sum(np.isnan(a)))
	print('max' , np.nanmax(a))
	print('min' ,np.nanmin(a))
	print('mean' ,np.nanmean(a))

def stats_format(a, name=None):
	print('Name:', str(name),'   Shape:' , a.shape ,'   NaNs:',np.sum(np.isnan(a)),' max:', np.nanmax(a),' min', np.nanmin(a),' mean:', np.nanmean(a))


def lanczos_1d(width,  dx, a=2):
    """
    This is a 1D lanczos Filter for time series analysis.
    it generates the Filter to be convolved with the timeseries
    https://en.wikipedia.org/wiki/Lanczos_resampling
    inputs:
    width  width of the filter in units of the timeseries
    a      Lanczos parameter (default =2). the length of the filter is a*width
    dx     delta x of the to be filtered timeseries

    returns:
    L       Lanczos Filter with the length a*width and dx.

    """
    # width= 2   # width of the filter in units of the timeseries
    # a=     1   # Lanczos parameter. the length of the filter is a*width
    # dx=   .1   # deltax of the to be filtered timeseries

    r=width/2.0
    xl=a*r
    x= np.arange(-xl, xl, dx)
    xprime=x/r

    # define the filter
    L = np.sinc(xprime) * np.sinc(xprime/a)
    L = np.where((xprime > -a) & (xprime < a),L, 0)

    return x, L/L.sum()

def lanczos_filter_1d(x, data, width, a=2 , mode='same', method='direct'):
    """
    colvolves the lanzcos filter with data.
    inputs
    x       independent variaable, dimension for data
    data    to be smoothed data, same dimensions a x
    width   width of the lanzos filter in dimensions of x
    a       lanzcos parameters. default 2. Integer.

    mode    passed to signal.convolve() 'full', 'valid','same'
    method  'direct', 'fft', 'auto'

    returns
    data_lp low-passed data, same size as before.
    """
    import scipy.signal as signal
    dx     =   np.diff(x).mean()
    x , L  =    lanczos_1d(width,  dx, a=a)


    data_lp= signal.convolve(data, L, mode=mode, method=method)#*

    return data_lp


def lateral_boundary_noise(xx, data, n=4,  lanzos_width=0.015,  mean_method=np.min):

    """
    this method creates a noise model from the first and last valid point
    at each index in the 2nd dimennsion of a 2d array.
    It estimates the "noise" at the lateral boundaries and returns an
    array of shape data that only varies in the second dimension

    It uses a 1d lanzcvos filter to create a low-pass field.

    inputs:
    x
    data
    n       number of valid gridpooints used at the boundaries
    lanzos_width    = 0.15 width of the lanzos filter in units of x
    mean_method     method how to derive the value at each y-index.
                    can be np.min, np.mean, ...

    return
    data_boundary_model   array of same size as data.

    """

    base=list()
    for i in np.arange(data.shape[1]):
        ll =data[:,i]
        aa = ll[~np.isnan(ll)][range(n) +  range(-n, 0)]
        base.append(mean_method(aa))

    a3 =lanczos_filter_1d( xx, np.array(base), lanzos_width, a=2 , mode='same', method='auto')

    __ , data_lb = np.meshgrid(np.arange(data.shape[0]), a3)

    return data_lb
