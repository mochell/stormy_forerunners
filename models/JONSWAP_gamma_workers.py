#import os
#execfile(os.environ['PYTHONSTARTUP'])
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

#%matplotlib Agg
import matplotlib
#get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt
import matplotlib.dates as dates


import multiprocessing as mp

import numpy as np
from .. import general as M
from .. import tools as MT
from .. import stormrecon_ph3 as Storm
#import scipy.special as special

#import pygrib
import os.path
import sys
#import datetime as DT
import numpy as np

#sys.tracebacklimit = 0
from matplotlib.colors import LinearSegmentedColormap
#from scipy import signal
import imp
import concurrent.futures
import time
import random
import shutil


def storm_fitter_JONSWAP_gamma(S, params,cont):
    # container need the following keys
    #allow_negative_slopes
    #substract_plain_simple
    #save_path
    #model_type
    #error_estimate
    #error_N

    #prior_flag= True if cont['priors'] is not None else False

    ###  Init paramters ###
    S.normalize_time() , S.write_log('- normalized time')


    #changes default params


    # 1.a slope inital

    S.write_log('- use estimated slopes and intersects for inital model')  # convert to freq(normalized_time)
    slope0=S.slope_to_dfdt_normalized()
    if np.isinf(S.slope_to_dfdt_normalized() ):
        slope0=10
    if cont['allow_negative_slopes']:
        smax=slope0*3 if slope0*.5 < slope0*3 else slope0*.5
        smin=slope0*0.5 if slope0*3 > slope0*0.5 else slope0*3
    else:
        smax=slope0*3
        smin=slope0*0.01
    params['slope'].set(value= slope0)#, min=smin , max=smax)

    # 1.b slope priors
    if ('priors' in cont.keys()) and (cont['priors'] is not False):
        if cont['priors']['slope']:
            cont['priors']['slope']=[smin/2.0, smax/2.0]
            S.write_log('slope priors set to min=' + str(smin/2.0) + ' max=' +  str(smax/2.0) )
            #print('slope priors set to min=' + str(smin/2.0) + ' max=' +  str(smax/2.0) )


    # 2. intetect intial value
    t0_temp=S.geo['t0L']+(S.geo['t0']-S.geo['t0L'])/2
    #intersect0=S.intersect_sec_to_dfdt_normalized(t0_temp) # convert to intersect with freqency axis
    intersect0=S.normalize_time_unit(S.geo['t0L']+(S.geo['t0']-S.geo['t0L'])/2) # set intersect with normalized time axis
    params['intersect'].set(value= intersect0-0)



    # 3. normalize data
    S.write_log('- normalization Data:')
    if not hasattr(S, 'factor'):
        S.data_std=np.nanstd(S.masked_data)#1e9
        S.factor= 1.0 / S.data_std

        S.write_log('  calculated normalization factor: '+ str(S.factor))
        S.write_log('  normalize S.masked_data*s.factor')
        S.data_normalized=S.masked_data / S.data_std
    else:
        S.write_log('  S.data_normalized already exist, no change to the data')


    # 4. Choose noise model
    ###### substract plain or not
    if cont['noise_model'] is False:
        data_normalized=S.data_normalized
        S.data_normalized_less_noise= data_normalized
        S.write_log('- no noise model applied ')
        S.noise_model= {'type':cont['noise_model'], 'tmean': None}


    elif cont['noise_model'] == 'substract_plain_simple':
        data_normalized=S.substract_plain(datasub=S.data_normalized, verbose=False)
        S.write_log('- substracted efolding plain')
        #S.substract_plain_simple( datasub=S.data*S.factor, verbose=True)
        #S.write_log('  substracted simple plain')
        S.data_normalized_less_noise= data_normalized
        S.noise_model= {'type':cont['noise_model'], 'tmean': self.plain_fitter.model_timemean}

    elif cont['noise_model'] == 'lateral_boundary_noise':

        noise_model = MT.lateral_boundary_noise(S.f, S.data_normalized, n=3, lanzos_width=0.015,  mean_method=np.min ).T
        temp1 = S.data_normalized - noise_model

        noise_model_add_slope = MT.top_bottom_tap(temp1.T,  mean_method=np.nanmin  )
        data_normalized = S.data_normalized - (noise_model + noise_model_add_slope.T)

        S.noise_model= {'type':cont['noise_model'], 'tmean': np.nanmean(noise_model, 0) }
        S.write_log('- substracted lateral boundaries model')
        #S.substract_plain_simple( datasub=S.data*S.factor, verbose=True)
        #S.write_log('  substracted simple plain')
        data_normalized[data_normalized < -.2] = 0
        S.data_normalized_less_noise = data_normalized


    else:

        raise Warning('Noise model not defined. No Noise model is applied.')
        data_normalized=S.data_normalized
        S.data_normalized_less_noise= data_normalized

        S.noise_model= {'type':cont['noise_model'], 'tmean': None}
        S.write_log('- no noise model defined, no model applied ')


    #S.clevs=M.clevels(data_normalized)
    S.clevs=np.linspace(-0.2, np.nanmax(S.data_normalized),21)
    #S.clevs=np.arange(-0.2, np.nanmax(data_normalized), )
    S.write_log('  redefine self.clevs to normalized scale')
    #S.write_log('added factor '+ str(S.factor) +' to local data_normalized, S.clevs and model, called. S.factor')


    # 4. fmax
    f_max0=S.f[M.find_max_ts(np.nanmean(data_normalized,0), smooth=True,spreed=round(S.f.size/10.), verbose=False )[0][0]]-0.01 #0.05
    if f_max0 < params['f_max'].min:
        f_max0=params['f_max'].min
    elif f_max0 > params['f_max'].max:
        f_max0=params['f_max'].max
    #print(f_max0)
    #params['f_max'].set(value= f_max0, min=S.geo['f_low'], max=S.geo['f_high'])
    params['f_max'].set(value= f_max0)#, min=S.geo['f_low'], max=S.geo['f_high'])
    S.write_log('- adjusted f_max to f_max0:'+ str(f_max0) + ' min:'+ str(S.geo['f_low']) + ' max:' + str(S.geo['f_high']) )

    # 5. adjust Amplitude
    amp=np.nanpercentile(data_normalized,99)
    params['amp_nondim'].set(value= amp)#, max=amp*100.0, min=amp*0.01)
    S.write_log('- adjusted non dimensional amp to :'+ str(amp) + ' min:'+ str(amp*100.0) + ' max:' + str(amp/100.0))


    # 6. save inital parameters
    parms_path_ID=cont['save_path']+S.ID+'_init_parms.json'
    params.dump(open(parms_path_ID, 'w'))
    S.write_log('init parameters and save at: '+ parms_path_ID)


    #test if prior is in container, it true set priors
    if ('priors' in cont.keys()) and (cont['priors'] is not False):
        priors=dict()

        for k,I in cont['priors'].items():
            priors[k]={'m_err':I, 'm0':params[k].value}
        #print(priors)
    else:
        priors=False

    S.write_log('- Priors:'+ str(priors))
    S.write_log('- Fit model: '+cont['model_type'] )


    # just a check if flag is set ot not
    if not 'MCMC' in cont.keys():
        print('error_workers is not set')
        cont['MCMC']=None
    #params.pretty_print()

    # submit all to fit model
    S.fit_model(params,ttype=cont['model_type'], datasub=data_normalized,  weight_opt=cont['weight_opt'], model=cont['gradient_method'],
                prior=priors,
                error_estimate=cont['MCMC']['vars'],
                error_N=cont['MCMC']['N'],
                error_workers= cont['MCMC']['workers'],
                error_nwalkers= cont['MCMC']['nwalkers'],
                set_initial=cont['MCMC']['set_initial'],
                error_opt=cont['MCMC']['emcee_kargs'],
                basinhopping_opt=cont['basinhopping_opt'])

    return S
