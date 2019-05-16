
if __name__ == '__main__':

    execfile(os.environ['PYTHONSTARTUP'])
    execfile(STARTUP_IG2018)

    %matplotlib inline
    #import m_general as M
    #import m_earth_geometry as M_geo
    #import m_spectrum as spec
    #import m_tools as MT


import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio
#import seaborn as sns

#import scipy.special as special
from mpl_toolkits.basemap import cm
#import pygrib
import os.path
import sys


import imp
import pickle
import datetime as DT
import matplotlib.dates as dates
from lmfit import minimize, Parameters
import scipy as sp

# In[] Plot data
def normalize_time(time):
    time=np.copy(time)
    dt=np.diff(time).mean()#G.dt_periodogram
    time=(time-time[0])/dt#np.arange(0,time.size, 1)
    return (time)/(time[-1]) , dt

if __name__ == '__main__':
    ifile='/work/mhell_work/RIS2016/processed//curvefitting/test_curvefitting.npy'
    plotpath=mconfig['paths']['plot']+'curvefitting/tests/'
    f = open(ifile, 'r')
    G=pickle.load(f)
    f.close()


    mask=M.cut_nparray(G.time, np.datetime64('2015-03-13T00:00:00'), np.datetime64('2015-03-14'))
    datasub=G.data[mask,:]
    time=G.time[mask]
    f=G.f[M.cut_nparray(G.f, 0, .5/(2*(np.pi)))]
    datasub=datasub[:,M.cut_nparray(G.f, 0, .5/(2*(np.pi)))]
    datasub=datasub[:,1:]
    f=f[1:]
    w=f*2*np.pi
    error_high=np.repeat((G.error_El[mask],), f.size, 0).T
    error_low=np.repeat((G.error_Ey[mask],), f.size, 0).T

    sample_unit='Hz'
    data_unit='m'
    G.data.sum()*G.df

    F=M.figure_axis_xy(10,5)
    #plt.subplot(1,2, 1)
    time.shape,f.shape,datasub.T.shape
    time, dt=normalize_time(G.time[mask])

    print(time)
    #plt.plot(time,slope0*time+.042)
    plt.contourf(time,f,datasub.T, color=cm.s3pcpn(2+1*2))
    xlabelstr=('  ( time)')
    plt.ylabel(('|X|^2/f (' + data_unit + '^2/' + sample_unit+ ')'))
    plt.xlabel(xlabelstr)
    plt.ylim(0.04,0.08)
    plt.colorbar()
    plt.grid()
    #F.make_clear()
    #plt.legend(

# In[]

def gamma_time(time,amp=1, gammapar=2, loc=.2, scale=0.1):
    from scipy.stats import gamma
    """
    configured for normalized time scale (0, 1)
    gammapar > 1
    time > 0
    loc is position 0 ..1
    scale  >0 .. \approx 0.3
    """
    gamma_mod=gamma(gammapar, loc = loc, scale = scale)
    return amp * gamma_mod.pdf(time)


def gamma_freq(f,amp=1, gammapar=2, loc=0.01, scale=0.001):
    from scipy.stats import gamma
    """
    configured for freq. timescale (0, 0.1)
    gammapar > 1
    time > 0
    loc is position 0 ..1
    scale  >0 .. \approx 0.3
    """
    gamma_mod=gamma(gammapar, loc = loc, scale = scale)
    return amp * gamma_mod.pdf(f)

def doublegamma(time, f,
                    slope, intersect,
                    tamp, tgammapar, tscale,
                    famp, fgammapar, floc, fscale,
                    plot=False):
    slopef=1/slope
    intersectf=-intersect/slope
    pfreq=f*slopef+intersectf
    #rint(pfreq.shape)
    tt, line=np.meshgrid( time, pfreq)
    #print(line.shape)
    #print(tt.shape)

    func_t= gamma_time(tt,amp=tamp, gammapar=tgammapar, loc=line, scale=tscale)
    #func_t_temp= (tamp*np.exp(- (time-t_center )**4 / tsigma ))
    #print(func_t.shape)
    func_freq_temp= gamma_freq(f,amp=famp, gammapar=fgammapar, loc=floc, scale=fscale)
    #func_freq_temp=   (famp*np.exp(- (f-fcenter)**2 / fsigma ))
    tt, func_freq= np.meshgrid( time, func_freq_temp)

    if plot:
        F=M.figure_axis_xy(8,10)
        plt.subplor(2, 1, 1)
        plt.contourf(time,f, func_freq*func_t)
        plt.subplor(2, 1, 1)
        plt.contourf(time,f, func_freq)
        plt.subplor(2, 1, 1)
        plt.contourf(time,f, func_t)

    return (func_freq*func_t).T


# In[] Test gamma model
if __name__ == '__main__':
    slope0=9*1e-7*G.dt_periodogram*time.size/2
    intersect0=.05

    tamp0=1e-9
    tgammapar0=2
    tscale0=.1

    famp0=1
    fgammapar0=2
    floc0=0.05
    fscale0=0.004

    #time_non_dim=normalize_time(time)[0]

    F=M.figure_axis_xy(10,5)
    model=doublegamma(time, f,
                        slope0, intersect0,
                        tamp0, tgammapar0, tscale0,
                        famp0, fgammapar0, floc0, fscale0, plot=False )
    model.max()
    plt.plot(time,slope0*time+intersect0)
    plt.contourf(time,f,datasub.T, color=cm.s3pcpn(2+3*2))
    plt.colorbar()
    plt.contour(time,f, model.T, colors='r')#, np.arange(0, .01, 0.001))
    plt.ylim(0.04,0.08)


# In[] build residual functions
if __name__ == '__main__':
    #http://cars9.uchicago.edu/software/python/lmfit/fitting.html
    params=Parameters()
    params.add('slope', value= slope0)# min=0.01, max=1)
    params.add('intersect', value= intersect0)#, min=0, max=.002)

    params.add('tamp', value= tamp0)#, min=0., max=1)
    params.add('tgammapar', value= tgammapar0, min=0.0001, max=2.5)
    params.add('tscale', value= tscale0, min=0, max=.1)

    params.add('famp', value= famp0)#, min=0., max=1)
    params.add('fgammapar', value= fgammapar0, min=0.0001, max=25)
    params.add('floc', value= floc0)#, min=0.01, max=1)
    params.add('fscale', value= fscale0)#, min=0.01, max=1)

def residual_gamma2d(params, time, f, data=None, eps=None):
    """ eps is the error/weighting"""
    vdict=params.valuesdict()

    model=doublegamma(time, f,
                        vdict['slope'], vdict['intersect'],
                        vdict['tamp'], vdict['tgammapar'], vdict['tscale'],
                        vdict['famp'], vdict['fgammapar'], vdict['floc'], vdict['fscale'],
                        plot=False )

    #tt, tt= np.meshgrid(time, ff)
    model1d=model.reshape(model.shape[0]*model.shape[1])

    if data is not None:
        if np.size(data.shape) != 1:
            if model.shape == data.shape:
                data1d=data.reshape(data.shape[0]*data.shape[1])
            elif model.shape == data.T.shape:
                data1d=data.T.reshape(data.T.shape[0]*data.T.shape[1])
            else:
                raise TypeError("data shape does not match")

    if data is None:
        return model1d
    if eps is None:
        return (model1d - data1d)
    return ( (model1d - data1d)/eps)

# In[] cheak plot
if __name__ == '__main__':
    model=residual_gamma2d(params, time, f , data=None, eps=None)
    model.max()

    F=M.figure_axis_xy(8,10, view_scale=.5)
    plt.subplot(2,1 ,1)
    plt.plot(model, c='r', label='model')
    plt.plot(datasub.reshape(datasub.shape[0]*datasub.shape[1]), c='b', label='data')
    plt.legend()

    plt.subplot(2,1, 2)

    plt.contourf(time,f,datasub.T)
    plt.colorbar()
    plt.contour(time, f, model.reshape(time.size, f.size).T, colors='r')
    plt.plot(time,params['slope'].value*time+params['intersect'])

    xlabelstr=('  ( time)')
    plt.ylabel(('|X|^2/f (' + data_unit + '^2/' + sample_unit+ ')'))
    plt.xlabel(xlabelstr)
    plt.ylim(0.04,0.08)
    plt.grid()
    F.save(path=plotpath, name='gamma_gamma_bulk')

# In[] minimize Gamma gamma model

if __name__ == '__main__':
    #Rinv=1/(np.identity(datasub.shape[0]*datasub.shape[1])*datasub.reshape(datasub.shape[0]*datasub.shape[1])*1e-2)
    data1d=datasub.reshape(datasub.shape[0]*datasub.shape[1])
    Rinv=1/(data1d/np.std(data1d))
    weight=M.runningmean(data1d, 30)
    weight[np.isnan(weight)]=0
    Rinv=1/(weight/np.std(data1d))

    #plt.plot(Rinv)
    #model=gauss2d(time, f, m0, b0, s0, a0, t0, tsigma, tA, plot=False)
    #model1d=model.reshape(model.shape[0]*model.shape[1])
    #data1d=datasub.reshape(datasub.shape[0]*datasub.shape[1])
def reduce_fct(r, Rinv):
    return np.matmul(np.matmul(r.T,Rinv),r)

if __name__ == '__main__':
    fitter = minimize(residual_gamma2d, params, args=(time, f,), kws={'data':datasub, 'eps':Rinv}, nan_policy='omit')#, reduce_fcn=reduce_fct)#, fcn_args=args, fcn_kws=kws,
                       #iter_cb=iter_cb, scale_covar=scale_covar, **fit_kws)

    time_syntetic=np.arange(0,1, .01)
    model_result=residual_gamma2d(fitter.params, time_syntetic, f )
    model_result_corse=residual_gamma2d(fitter.params, time, f )

    # #In[] plot Gamma Gamma model
    F=M.figure_axis_xy(10,18, view_scale=0.4)
    plt.subplot(5,1 ,1)
    plt.plot(Rinv, c='g', label='weight')
    plt.subplot(5,1 ,2)
    plt.plot(model_result_corse, c='k', label='model')
    plt.plot(datasub.reshape(datasub.shape[0]*datasub.shape[1]), c='b', label='data')
    plt.legend()

    plt.subplot(5,1, 3)
    cval=np.linspace(-7e-7, 7e-7, 41)
    cmap=plt.cm.RdBu_r
    #cmap = sns.diverging_palette(220, 20, n=41, as_cmap=True)
    plt.contourf(time,f,datasub.T, cval, cmap=cmap)
    plt.colorbar()

    plt.contour(time_syntetic,f,model_result.reshape(time_syntetic.size, f.size).T, colors='black', alpha=0.5)
    plt.plot(time, fitter.params['slope'].value*time+fitter.params['intersect'])
    xlabelstr=('  ( time)')
    plt.ylabel(('|X|^2/f (' + data_unit + '^2/' + sample_unit+ ')'))
    plt.xlabel(xlabelstr)
    plt.ylim(0.04,0.08)
    plt.grid()

    plt.subplot(5,1, 4)

    plt.contourf(time,f,fitter.residual.reshape(time.size, f.size).T, cval, cmap=cmap)
    plt.colorbar()
    #plt.contour(time,f,, colors='w')
    plt.plot(time, fitter.params['slope'].value*time+fitter.params['intersect'])
    xlabelstr=('  ( time)')
    plt.ylabel(('|X|^2/f (' + data_unit + '^2/' + sample_unit+ ')'))
    plt.xlabel(xlabelstr)
    plt.ylim(0.04,0.08)
    plt.grid()

    plt.subplot(5,1, 5)
    model_tmean=model_result.reshape(time_syntetic.size, f.size).T.mean(1)
    residual_tmean=fitter.residual.reshape(time.size, f.size).T.mean(1)
    data_tmean=datasub.T.mean(1)
    plt.plot(f,model_tmean , label='model')
    plt.plot(f,residual_tmean, label='residual func')
    plt.plot(f,data_tmean-model_tmean, label='residual')
    plt.plot(f,data_tmean, label='data' )
    plt.legend()

    #F.save(path=plotpath, name='gamma_gamma_result')


# In[]
if __name__ == '__main__':
    Rinv=(data1d/np.std(data1d))
    weight_tmean=Rinv.reshape(time.size, f.size).T.mean(1)
    plt.plot(f,-residual_tmean/weight_tmean, label='residual func')
    plt.plot(f,data_tmean-model_tmean, label='residual')
    plt.legend()

# In[] Use JONWAP in freq and gamma in time
def JONSWAP_bulk(f,floc=0.04, famp=1e-2,  gamma=3.3, peak_std=1e-1, stretch=5/4):
    """
    see Ocean Surface waves - S. R. Massel eq.3.69 and eq.3.81

    """
    B=0.74
    g=9.81
    w=f*2*np.pi
    wp=floc*2*np.pi

    #peak_std=1e-01 #0.00555
    #scale_std=1e-2
    alpha=famp
    #stretch=5/4

    delta=np.exp(-(w-wp)**2 / ( 2*peak_std**2 *wp**2 ) )
    peak_factor=gamma**delta
    return alpha * w**(-5) * np.exp(-stretch*  (w/wp)**-4)*peak_factor # units of m^2 / Hz
    #np.exp(-B *(g/(w*U))**4)
# In[]


def JONSWAP_bulk_acc(f,floc=0.04, famp=1e-2,  gamma=3.3, peak_std=1e-1, stretch=5/4):
    """
    see Ocean Surface waves - S. R. Massel eq.3.69 and eq.3.81

    """
    B=0.74
    g=9.81
    w=f*2*np.pi
    wp=floc*2*np.pi

    alpha=famp

    delta=np.exp(-(w-wp)**2 / ( 2*peak_std**2 *wp**2 ) )
    peak_factor=gamma**delta
    return alpha * w**(-3) * np.exp(-stretch*  (w/wp)**-4)*peak_factor # units of (m/s)^2 / Hz


def plot_line_params(time, slope, intersect, **kargs):
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
    intersectF=- intersect*slope
    pfreq=time*slope+intersectF
    plt.plot(time, pfreq, **kargs)

def gamma_time_JONSWAO_freq(time, f,
                    slope_t, intersectT,
                    tamp, tgammapar, tscale,
                    floc=0.04, famp=1e-2, gamma_peak=3.3,  peak_std=1e-1, stretch=.09,
                    plot=False, acc=False):

    """
    This method calculated a 2D shaped function given the parameters:
    inputs:
    time        normalized time [0, 1] np.array
    f           frequency in the swell band, np.array,
    slope_t     slope of the "dispersed peak frequencies" df/dt [Hz/ normalized time]
    intersectT  intersect of that line in units of normalized time
    tamp        amplitude of the gamma function, adds upt with famp!
    tgammapar   gamma parameter of the gamma function in time
    tscale      scaling parameter of the gamma function
    floc        =0.04, location of the peak frequency on the JONSWAP spectrum
    famp        =1e-2, amplitude scaling of the JONSWAP spectrum
    gamma_peak  =3.3,  gamma peak parameter of the JONSWAP spectrum
    peak_std    =1e-1, width of the additioanl peak in the JONSWAP spectrum
    stretch     =.09,  stretching parameter inthe JONSWAP spectrum

    plot        True, False. Simple plot of the output function

    return:
                2d function with the shape of [time,freq]
    """

    #intersectf=intersect-intersect/slopet
    intersectF=-intersectT*slope_t
    pfreq=time*slope_t+intersectF
    #print('intersect F=' + str(intersectF))

    #intersectf=intersect#-intersect/slope
    slopeF=1/slope_t
    pfreq_forgamma=f*slopeF+intersectT

    #rint(pfreq.shape)
    tt, line=np.meshgrid(time, pfreq_forgamma)
    #print(line)
    #print(tt)

    func_t= gamma_time(tt,amp=tamp, gammapar=tgammapar, loc=line, scale=tscale)
    #func_t_temp= (tamp*np.exp(- (time-t_center )**4 / tsigma ))
    #print(func_t.shape)
    #func_freq_temp= gamma_freq(f,amp=famp, gammapar=fgammapar, loc=floc, scale=fscale)
    if acc:
        func_freq_temp= JONSWAP_bulk_acc(f,floc=floc,  famp=famp,gamma=gamma_peak, peak_std=peak_std, stretch=stretch)
    else:
        func_freq_temp= JONSWAP_bulk(f,floc=floc,  famp=famp,gamma=gamma_peak, peak_std=peak_std, stretch=stretch)
    #func_freq_temp=   (famp*np.exp(- (f-fcenter)**2 / fsigma ))
    tt, func_freq= np.meshgrid( time, func_freq_temp)

    if plot:
        F=M.figure_axis_xy(8,10)
        plt.subplot(3, 1, 1)
        plt.contourf(time,f, func_freq*func_t)
        plt.plot(time, pfreq)
        plt.grid()

        plt.subplot(3, 1, 2)
        plt.plot(f, func_freq.mean(1))
        plt.contourf(time,f, func_freq)
        plt.subplot(3, 1, 3)
        plt.contourf(time,f, tt)
        plt.ylim(0.04,0.08)
        plt.grid()

    return (func_freq*func_t).T




# In[]
    #http://cars9.uchicago.edu/software/python/lmfit/fitting.html
    params=Parameters()
    params.add('slope', value= slope0, min=slope0*.1 , max=slope0*10)
    params.add('intersect', value= intersect0, min=-0.5, max=.5)

    #params.add('tamp', value= tamp0, min=0., max=1)
    params.add('tgammapar', value= tgammapar0, min=0.0001, max=4)
    params.add('tscale', value= tscale0, min=0, max=.1)

    params.add('floc', value= floc0, min=0., max=1)
    params.add('famp', value= famp0, min=0.0001, max=25)
    params.add('gamma_peak', value= gamma_peak0, min=0, max=4)
    params.add('peak_std', value= peak_std0, min=0.0001, max=6)
    params.add('stretch', value= stretch0, min=-.001, max=4)

def residual_JANSWAP_gamma(params_local, time, f, data=None, eps=None, weight=None):
    """ eps is the error/weighting"""
    vdict=params_local.valuesdict()
    #+vdict['level']
    tscale=0.07
    model=gamma_time_JONSWAO_freq(time, f,
                        vdict['slope'], vdict['intersect'],
                        1, vdict['tgammapar'], tscale,
                        vdict['floc'], vdict['famp'], vdict['gamma_peak'], vdict['peak_std'],vdict['stretch'],
                        plot=False )

    #tt, tt= np.meshgrid(time, ff)
    model1d=model.reshape(model.shape[0]*model.shape[1])

    if data is not None:
        if np.size(data.shape) != 1:
            if model.shape == data.shape:
                data1d=data.reshape(data.shape[0]*data.shape[1])
                nan_track=np.isnan(data1d)
            elif model.shape == data.T.shape:
                data1d=data.T.reshape(data.T.shape[0]*data.T.shape[1])
                nan_track=np.isnan(data1d)
            else:
                raise TypeError("data shape does not match")


    if data is None:
        return model1d
    if (eps is not None) and (weight is None):
        d=(model1d - data1d)/eps
        d[nan_track]=np.nan
        return d
    if (weight is not None) and (eps is None):
        d=(model1d - data1d)*weight
        d[nan_track]=np.nan
        return d
    if (eps is None) and (weight is None):
        d= model1d - data1d
        d[nan_track]=np.nan
        return d


def residual_JANSWAP_gamma_regularization_acc(params_local, time, f, data=None, eps=None, weight=None, prior=None):
    return residual_JANSWAP_gamma_regularization(params_local, time, f, data=None, eps=None, weight=None, prior=None, acc=True)

def residual_JANSWAP_gamma_regularization(params_local, time, f, data=None, eps=None, weight=None, prior=None, acc=False):
    """ eps is the error/weighting"""
    vdict=params_local.valuesdict()
    #vdict0=params0.valuesdict()
    #+vdict['level']
    tscale=0.07
    model=gamma_time_JONSWAO_freq(time, f,
                        vdict['slope'], vdict['intersect'],
                        1, vdict['tgammapar'], tscale,
                        vdict['floc'], vdict['famp'], vdict['gamma_peak'], vdict['peak_std'],vdict['stretch'],
                        plot=False , acc=acc)

    #tt, tt= np.meshgrid(time, ff)
    model1d=model.reshape(model.shape[0]*model.shape[1])

    if data is not None:
        if np.size(data.shape) != 1:
            if model.shape == data.shape:
                data1d=data.reshape(data.shape[0]*data.shape[1])
                nan_track=np.isnan(data1d)
            elif model.shape == data.T.shape:
                data1d=data.T.reshape(data.T.shape[0]*data.T.shape[1])
                nan_track=np.isnan(data1d)
            else:
                raise TypeError("data shape does not match")

    #modelparameter costfunction
    if prior is not None:
        Jm=Jm_regulizer(vdict, prior, alpha=5.0)

    #print(Jm)

    if data is None:
        return model1d
    if (eps is not None) and (weight is None):
        d=(model1d - data1d)/eps
        d[nan_track]=np.nan
        return np.concatenate(( d, Jm ))

    if (weight is not None) and (eps is None):
        d=(model1d - data1d)*weight
        d[nan_track]=np.nan
        #print(np.concatenate(( d,Jm )) )
        return np.concatenate(( d, Jm ))

    if (eps is None) and (weight is None):
        d= model1d - data1d
        d[nan_track]=np.nan
        return d

def Jm_regulizer(vdict, prior, alpha=5.0):
    """
    returns a Model cost function as list. each item is the cost for each prior given the parameter value in vdict
    """
    Jm=list()
    for k,I in prior.iteritems():
        if type(I['m_err']) is float:
            Jm.append(     alpha * (I['m0']- vdict[k] ) / I['m_err']    )
        else:
            if vdict[k] >= I['m0']:
                Jm.append( alpha * (I['m0']- vdict[k] ) / I['m_err'][1] )
            else:
                Jm.append( alpha * (I['m0']- vdict[k] ) / I['m_err'][0] )
    return Jm




# In[] cheak plot integrated in
if __name__ == '__main__':
    model=residual_JANSWAP_gamma(params, time, f , data=None, eps=None)

    # #In[] minimize
    # Build function for miniizing  called in Storm with data, time, f, rm_distance
    data1d=datasub.reshape(datasub.shape[0]*datasub.shape[1])
    weight=M.runningmean(data1d, 10)
    weight[np.isnan(weight)]=0
    lower_bound=error_low.reshape(datasub.shape[0]*datasub.shape[1])
    Rinv=(1/(weight/np.std(data1d)))+lower_bound*0
    #Rinv=None#1/(data1d)*1e-2

    #,,  method='differential_evolution'\

    # datasub.shape
    # datasub.size
    # Rinv.shape
    # time.shape
    # f.shape
    # plt.plot(residual_JANSWAP_gamma(params, time, f, data=datasub, eps=None, weight=None))
    fitter2 = minimize(residual_JANSWAP_gamma, params,method='least_squares', args=(time, f,), kws={'data':datasub, 'eps':Rinv}, nan_policy='omit')#, reduce_fcn=reduce_fct)#, fcn_args=args, fcn_kws=kws,
                       #iter_cb=iter_cb, scale_covar=scale_covar, **fit_kws)

    time_syntetic=np.arange(0,1, .01)
    model_result=residual_JANSWAP_gamma(fitter2.params, time_syntetic, f )
    model_result_corse=residual_JANSWAP_gamma(fitter2.params, time, f )

    fitter2.params.pretty_print()


# In[]

# if __name__ == '__main__':
#     F=plot_fitted_model()
#     F.save_light(path=plotpath, name='gamma_JANSWA_result_all')
#
#     fitter2.params.pretty_print()
#     print('---')
#
#
#     # Draw Chi surface in the slope- intersect space
#     fitter2.redchi
#     #np.sum(fitter2.residual**2)/(fitter2.ndata-fitter2.nvarys)

def redchi_manuel(time, f, model, datasub, eps ,parms,  var1, value1, var2, value2, ndata, nvarys ):

    parms.add(var1, value=value1)
    parms.add(var2, value=value2)

    resid_manual=model(parms, time, f , data=datasub, eps=eps)
    return np.sum(resid_manual**2)/(fitter2.ndata-fitter2.nvarys)


if __name__ == '__main__':
    bestfit=fitter2.params.valuesdict().copy()
    s_vary=np.linspace(bestfit['slope']/10,bestfit['slope']*10,100)
    inter_vary=np.linspace(bestfit['intersect']-0.1,bestfit['intersect']+0.1,100)

    var_surface=np.zeros([s_vary.size])
    for s in s_vary:
        interl=[]
        for inter in inter_vary:
            #var_surface[s, inter]=
            interl.append(redchi_manuel(time, f, residual_JANSWAP_gamma, datasub, Rinv ,fitter2.params.copy(),
                        'slope', s, 'intersect' , inter,
                        fitter2.ndata, fitter2.nvarys ))
        var_surface=np.vstack((var_surface,interl ))

    var_surface=np.delete(var_surface,0,0)
    MT.stats(var_surface)
    # #In[]
    F=M.figure_axis_xy(6,6, view_scale=.5)

    cval=np.linspace(4e-15, 9e-14, 21)
    #cmap = sns.light_palette(220, 20, n=41, as_cmap=True)
    #cmap=sns.color_palette("GnBu_d", 41,as_cmap=True)
    cmap=plt.cm.RdBu_r
    #cmap=sns.cubehelix_palette(8,light=.9, dark=0.1, as_cmap=True, rot=-.4)
    plt.contourf(s_vary,inter_vary,var_surface.T, cval, cmap=cmap)

    plt.colorbar()

    #plt.contour(time_syntetic,f,model_result.reshape(time_syntetic.size, f.size).T, colors='black', alpha=0.5)
    plt.plot(fitter2.params['slope'], fitter2.params['intersect'], '.', c='r')
    xlabelstr=(' slope')
    plt.ylabel(('intersect'))
    plt.xlabel(xlabelstr)
    #plt.xlim(0,0.2)
    plt.grid()
    plt.title('Chi^2 plane')
#    F.save(path=plotpath, name='slope_intersect_chi2')
