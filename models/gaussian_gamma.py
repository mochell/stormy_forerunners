
if __name__ == '__main__':

    execfile(os.environ['PYTHONSTARTUP'])
    execfile(STARTUP_IG2018)

    %matplotlib inline


import sys, imp
#import pickle
#from lmfit import minimize, Parameters


# %%
def normalize_time(time):
    """
    returns a time vector from 0 to 1 of the length of time
        and the time step dt in dimentional units
    """
    time=np.copy(time)
    dt=np.diff(time).mean()#G.dt_periodogram
    time=(time-time[0])/dt#np.arange(0,time.size, 1)
    return (time)/(time[-1]) , dt

def a_gaussian(x,x0,sigma):
  return np.exp(-np.power((x - x0)/sigma, 2.)/2.)

#%% generate some data_ano
from scipy.stats import gamma

f=np.arange(1/1000.0, .2, 0.001)
time=np.arange(0, 1, 0.001)

tt, ff = np.meshgrid(time, f)
fake_data=  np.sin(tt*np.pi)**4 * a_gaussian( ff, .05, 0.03)

plt.contour(tt, ff, fake_data, colors='k')


# %% basic functions
def gamma_time_normlized_amp(time, gammapar=2, loc=.2, scale=0.1):
    from scipy.stats import gamma
    """
    configured for normalized time scale (0, 1)
    gammapar > 1
    time > 0
    loc is position 0 ..1
    scale  >0 .. \approx 0.3
    """
    gamma_mod=gamma(gammapar, loc = loc, scale = scale).pdf(time)
    return  gamma_mod/gamma_mod.max()

plt.plot(time, gamma_time_normlized_amp(time))
plt.plot(time, gamma_time_normlized_amp(time, gammapar=3, loc=.2, scale=0.1) )


# %% Exponentially modified Gaussian distribution
# https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution


def gaussian(x,nu,sigma):
  return np.exp(-np.power((x - nu)/sigma, 2.)/2.)


def EMG(x, nu, sigma, lamb):

    """ """
    from scipy.special import erfcx,erfc
    vfac = lamb / 2.0
    #disp= vfac*  np.exp( vfac * (2 * nu  +  lamb * sigma**2.0  -  2*x)) * erfc( (nu + lamb * sigma**2 - x ) / ( np.sqrt(2) * sigma )  )
    h=1.0
    tau=1/lamb
    disp= (h*sigma/tau) * np.sqrt(np.pi/2.0) *  np.exp( 0.5 * (sigma/tau)**2.0  - (x - nu) / tau ) * erfc( (sigma/tau  - (x- nu )/sigma ) / np.sqrt(2))
    #disp=  np.exp(- vfac *( ((x - nu  + lamb) /sigma ) **2 )/2. )

    return disp

#plt.plot(f, gaussian(f, 0.09, 0.01   ))
plt.plot(f,      EMG(f, 0.02, 0.005, 1))
#plt.plot(time, gaussian(time, gammapar=3, loc=.2, scale=0.1) )


# %% a simpler version
f
def gaussian_skew(x,nu,sigma, n1):
  factor= n1*(2* np.pi *x )**(-1)
  #factor= 1/ (2* np.pi *x )*n1
  return factor + np.exp(-np.power((x - nu)/sigma, 2.)/2.)


#x=np.arange(0, 5 ,0.01)

plt.plot(f, gaussian(f, 0.09, 0.01   ))
b=gaussian_skew(f, 0.09, 0.01,1)
plt.plot(f, b /b.max() )

# %%

def pierson_moskowitz_default(f, U):
    """
    see Ocean Surface waves - S. R. Massel eq.3.79 and eq.3.80

    """
    g=9.81
    wp=0.879*g / U
    w=2.*np.pi*f
    sigma=0.04 *g / wp**2.0
    alpha=5.0* (wp**2.0 *sigma / g)**2.0

    return  alpha * w**(-5.0) * g**2.0 *np.exp(-5./4.0 * (w/wp)**-4)#

def PM_IG(f,fp ,alpha,  n1=5.0, n2=4.0):

    """ This Function generates a gestalt that is informed by the PM spectrum """

    return  alpha * (2* np.pi *f )**(-n1) *np.exp(-5./4. * (f/fp)**-n2)#


def PM_IG_normalized(f,fp , n1=5.0, n2=4.0):
    """ This Function generates a gestalt that is informed by the PM spectrum """
    a=(2* np.pi *f )**(-n1) *np.exp(-5./4. * (f/fp)**-n2)
    a[np.isnan(a)]=0
    a=a/np.nanmax(a)
    return  a #


f2=np.arange(0, .1 ,0.001)[1:]

a= PM_IG(f2, 0.06 , 1, 5, 4)
plt.plot(f2, a/np.nanmax(a))

b= PM_IG_normalized(f2, 0.005 , 1, 2)
plt.plot(f2, b, 'r')




# %%

def gamma_time_PM_IG_default(time, f,
                    slope_t, intersectT,
                    tgammapar, tscale,
                    f_max=0.01, power_slope = 5., power_exp = 4.,
                    amplitude=1.,
                    plot=False):
    #time, f,slope_t, intersectT,tgammapar, tscale,f_max, power_slope, power_exp = time, f2, slope0, intersect0, tgammapar0, tscale0, 0.01,  2,  2
    """
    This method calculated a 2D shaped function given the parameters:
    inputs:

    time        normalized time [0, 1] np.array
    f           frequency in the swell band, np.array,

    slope_t     slope of the "dispersed peak frequencies" df/dt [Hz/ normalized time]
    intersectT  intersect of that line in units of normalized time

    tgammapar   gamma parameter of the gamma function in time
    tscale      scaling parameter of the gamma function

    f_max       =0.01, location of the peak frequency in frequency dimension

    gamma_peak  =3.3,  gamma peak parameter of the JONSWAP spectrum
    power_slope = 5, power of the slope factor in front of the exponential
    power_exp   = 4, power of the f/fmax term in the exponential
    amplitude   amplitude of the whole function.  if =1 , peak amplitude corresponds to JONSWAPs values

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
    func_t= gamma_time_normlized_amp(tt, gammapar=tgammapar, loc=line, scale=tscale)
    #func_t_temp= (tamp*np.exp(- (time-t_center )**4 / tsigma ))
    #print(func_t.shape)

    """ Define X(f_max and U) here """


    func_freq_temp= PM_IG_normalized(f, f_max,  power_slope, power_exp)

    #func_freq_temp=   (famp*np.exp(- (f-fcenter)**2 / fsigma ))
    tt, func_freq= np.meshgrid( time, func_freq_temp)

    if plot:
        tt, ff=np.meshgrid(time, f)
        F=M.figure_axis_xy(8,10)
        plt.subplot(3, 1, 1)
        plt.contourf(tt, ff, func_t)
        plt.plot(time, pfreq)
        plt.ylim(f.min(), f.max())


        plt.subplot(3, 1, 2)
        #plt.plot(f, func_freq.mean(1))
        plt.contourf(tt, ff,func_freq)

        plt.subplot(3, 1, 3)
        plt.contourf(tt, ff, func_t * func_freq)
        #plt.ylim(0.04,0.08)
        plt.grid()

    return (func_t * func_freq).T



# %%

slope0= 1.2
intersect0=.3
tgammapar0=2
tscale0=.1
f2=np.arange(0, .1 ,0.001)[1:]

model_func =gamma_time_PM_IG_default(time, f2, slope0, intersect0, tgammapar0, tscale0, f_max=0.01, power_slope = 2, power_exp = 2, plot=True)
#plt.contour(tt, ff, fake_data, colors='k')

# %% build residual
def residual_PM_IG_gamma(params_local, time, f, data=None, eps=None, weight=None):
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


# %%

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
