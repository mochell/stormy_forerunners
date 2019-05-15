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
import m_general as M
import m_tools as MT
import AA_plot_base as AA
import AA_stormrecon as Storm
#import scipy.special as special

#import pygrib
import os.path
import sys
#import datetime as DT


#sys.tracebacklimit = 0
from matplotlib.colors import LinearSegmentedColormap
#from scipy import signal
import imp
import concurrent.futures
import time
import random
import shutil


def storm_fitter_stage2(S, params,cont):
    # container need the following keys
    #allow_negative_slopes
    #substract_plain_simple
    #save_path
    #model_type
    #error_estimate
    #error_N

    ###  Init paramters ###
    S.normalize_time() , S.write_log('- normalized time')

    #changes to default params
    slope0=S.slope_to_dfdt_normalized()
    if np.isinf(S.slope_to_dfdt_normalized() ):
        slope0=10
    S.write_log('- use estimated slopes and intersects for inital model')  # convert to freq(normalized_time)
    if cont['allow_negative_slopes']:
        smax=slope0*3 if slope0*.5 < slope0*3 else slope0*.5
        smin=slope0*0.5 if slope0*3 > slope0*0.5 else slope0*3
    else:
        smax=slope0*3
        smin=slope0*0.5
    params['slope'].set(value= slope0, min=smin , max=smax)

    if cont['priors']['slope']:
        cont['priors']['slope']=[smin/2.0, smax/2.0]
        S.write_log('slope priors set to min=' + str(smin/2.0) + ' max=' +  str(smax/2.0) )
        #print('slope priors set to min=' + str(smin/2.0) + ' max=' +  str(smax/2.0) )


    t0_temp=S.geo['t0L']+(S.geo['t0']-S.geo['t0L'])/2
    #intersect0=S.intersect_sec_to_dfdt_normalized(t0_temp) # convert to intersect with freqency axis
    intersect0=S.normalize_time_unit(S.geo['t0L']+(S.geo['t0']-S.geo['t0L'])/2) # set intersect with normalized time axis
    params['intersect'].set(value= intersect0+0.1)

    # add factor , back to nm
    #print(S.factor)
    #print(np.nanstd(S.masked_data))
    S.write_log('- normalization Data:')
    if not hasattr(S, 'factor'):
        S.factor=1.0/np.nanstd(S.masked_data)#1e9

        S.write_log('  calculated normalization factor: '+ str(S.factor))
        S.write_log('  normalize S.masked_data*s.factor')
        S.masked_data=S.masked_data*S.factor
    else:
        S.write_log('  S.masked_data already normalized, no change to the data')

    ###### substract plain or no!
    if cont['substract_plain_simple']:
        masked_data=S.substract_plain(datasub=S.masked_data, verbose=False)
        S.write_log('  substracted efolding plain')
        #S.substract_plain_simple( datasub=S.data*S.factor, verbose=True)
        #S.write_log('  substracted simple plain')
        S.masked_data_less_noise= masked_data
    else:
        masked_data=S.masked_data
        S.masked_data_less_noise= masked_data

    S.clevs=M.clevels(masked_data)
    S.write_log('  redefine self.clevs to normalized scale')
    #S.write_log('added factor '+ str(S.factor) +' to local masked_data, S.clevs and model, called. S.factor')

    floc0=S.f[M.find_max_ts(np.nanmean(masked_data,0), smooth=True,spreed=round(S.f.size/10.), verbose=False )[0][0]] #0.05
    if floc0 < params['floc'].min:
        floc0=params['floc'].min
    elif floc0 > params['floc'].max:
        floc0=params['floc'].max
    #params['floc'].set(value= floc0, min=S.geo['f_low'], max=S.geo['f_high'])
    params['floc'].set(value= floc0)#, min=S.geo['f_low'], max=S.geo['f_high'])
    S.write_log('- adjusted floc to floc0:'+ str(floc0) + ' min:'+ str(S.geo['f_low']) + ' max:' + str(S.geo['f_high']) )

    # adjust Amplitude
    amp=0.076* (floc0*50* 2*9.81 / 7.0  )**0.66 / 100.0
    params['famp'].set(value= amp, max=amp*100.0, min=amp*0.01)
    S.write_log('- adjusted famp to famp:'+ str(amp) + ' min:'+ str(amp*100.0) + ' max:' + str(amp/100.0))

    parms_path_ID=cont['save_path']+'init_parms_'+S.ID+'.json'
    params.dump(open(parms_path_ID, 'w'))
    S.write_log('init parameters and save at: '+ parms_path_ID)

    #test if prior is in container
    if 'priors' in cont.keys():
        priors=dict()

        for k,I in cont['priors'].iteritems():
            priors[k]={'m_err':I, 'm0':params[k].value}
        #print(priors)
    else:
        priors=None

    S.write_log('- Priors:'+ str(priors))
    S.write_log('- Fit model: '+cont['model_type'] )

    if not 'error_workers' in cont.keys():
        print('error_workers is not set')
        cont['error_workers']=None
    #params.pretty_print()
    S.fit_model(params,ttype=cont['model_type'], datasub=masked_data,  wflag='combined', model='least_squares',
                error_estimate=cont['error_estimate'], error_N=cont['error_N'],error_workers= cont['error_workers'],
                error_nwalkers= cont['error_nwalkers'], prior=priors, set_initial=cont['set_initial'],
                error_opt=cont['error_opt'])

    return S



# def storm_fitter(SID,queue,q_post,container=None):
#
#     if container is None:
#         raise Warning('pars is not set. provide pars dict')
#
#     #locals().update(container)
#     for key,val in container.items():
#         exec(key + '=val')
#
#
#     from lmfit import minimize, Parameters
#     try:
#         print(SID)
#
#         #load default_parameters:
#         params=Parameters()
#         params.load( open(params_path+'parms_basic_adjusted_rms.json', 'r' ))
#
#         # #In[]
#         imp.reload(Storm)
#         S=Storm.Storm(SID)
#
#         S.load(save_path)
#         S.write_log(' ---- load data for data fitting')
#
#     except:
#         print("--------Fail initializing:"+ S.ID)
#         #raise Warning('fail to initialize '+SID)
#
#
#         # #In[]
#         ###  Init paramters ###
#         S.normalize_time() , S.write_log('normalized time, use estimated slopes and intersects')
#
#         #changes to default params
#         slope0=S.slope_to_dfdt_normalized()  # convert to freq(normalized_time)
#         if allow_negative_slopes:
#             smax=slope0*1.5 if slope0*.5 < slope0*1.5 else slope0*.5
#             smin=slope0*0.5 if slope0*1.5 > slope0*0.5 else slope0*1.5
#         else:
#             smax=slope0*1.5
#             smin=slope0*0.5
#         params['slope'].set(value= slope0, min=smin , max=smax)
#
#         t0_temp=S.geo['t0L']+(S.geo['t0']-S.geo['t0L'])/2
#         intersect0=S.intersect_sec_to_dfdt_normalized(t0_temp) # convert to intersect with freqency axis
#         params['intersect'].set(value= intersect0)
#
#         # add factor , back to nm
#         S.factor=1/np.nanstd(S.masked_data)#1e9
#         #print('init' ,np.nanmax(S.masked_data))
#         S.masked_data=S.masked_data*S.factor
#
#         ###### substract plain or no!
#         if substract_plain_simple:
#             S.substract_plain_simple( datasub=S.data*S.factor, verbose=False)
#             S.write_log('substracted simple plain')
#
#         S.clevs=M.clevels(S.masked_data)
#         S.write_log('adjusted self.clevs to normalized scale')
#         S.write_log('added factor '+ str(S.factor) +' to local masked_data, S.clevs and model, called. S.factor')
#
#         floc0=S.f[M.find_max_ts(np.nanmean(S.masked_data,0), smooth=True,spreed=round(S.f.size/10.), verbose=False )[0][0]] #0.05
#         if floc0 < params['floc'].min:
#             floc0=params['floc'].min
#         elif floc0 > params['floc'].max:
#             floc0=params['floc'].max
#         #params['floc'].set(value= floc0, min=S.geo['f_low'], max=S.geo['f_high'])
#         params['floc'].set(value= floc0)#, min=S.geo['f_low'], max=S.geo['f_high'])
#         S.write_log('adjusted floc to floc0:'+ str(floc0) + ' min:'+ str(S.geo['f_low']) + ' max:' + str(S.geo['f_high']))
#
#         parms_path=save_path+'init_parms_'+ID.string+'.json'
#         params.dump(open(parms_path, 'w'))
#         S.write_log('init parameters and save at: '+ parms_path)
#
#     try:
#
#         S.write_log('init parameters')
#         S.write_log(model_type)
#         S.fit_model(params,ttype=model_type, datasub=S.masked_data,  wflag='combined', model='least_squares',
#                     error_estimate=error_estimate, error_N=error_N)
#     except:
#         print("--------Fail fitting:"+ S.ID)
#         return S.ID
#
#     #try:
#
#     if no_plots:
#         print('no plot')
#         S.masked_data=S.masked_data/S.factor
#         S.clevs=S.clevs/S.factor
#         S.model_result=S.model_result/S.factor
#         S.model_result_corse=S.model_result_corse/S.factor
#         S.residual_2d=S.residual_2d/S.factor
#
#         #print(np.nanmax(S.masked_data) , np.nanmax(S.model_init))
#         S.write_log('adjust for factor '+ str(1/S.factor) +' to S.clevs and S.model, called. S.factor')
#         ## save Data
#         time.sleep(random.random()*2)
#         S.write_log('fit model and saved')
#
#         S.convert_normalized_intersect_slope()
#
#         S.save(save_path)
#         time.sleep(1)
#         #return SID
#
#     else:
#
#         queue.put(S)
#         S.write_log('plot saved at: '+S.ID+'_fit_stat')
#         time.sleep(.2)
#         #print('wait fot plotter')
#         while True:
#             print('waiting', S.ID)
#             post = q_post.get()
#             if post is 'fail':
#                 print('fail in plotter', S.ID)
#                 break
#             if post == S.ID:
#                 print('plotter send back')
#                 S.write_log('plot saved at: '+S.ID+'_fit_stat')
#                 #time.sleep(2)
#                 # reaajust for scalar
#                 #print(np.nanmax(S.masked_data) , np.nanmax(S.model_init))
#                 S.masked_data=S.masked_data/S.factor
#                 S.clevs=S.clevs/S.factor
#                 S.model_result=S.model_result/S.factor
#                 S.model_result_corse=S.model_result_corse/S.factor
#                 S.residual_2d=S.residual_2d/S.factor
#
#                 #print(np.nanmax(S.masked_data) , np.nanmax(S.model_init))
#                 S.write_log('adjust for factor '+ str(1/S.factor) +' to S.clevs and S.model, called. S.factor')
#                 ## save Data
#                 time.sleep(random.random()*2)
#                 S.write_log('fit model and saved')
#
#                 S.convert_normalized_intersect_slope()
#
#                 S.save(save_path)
#                 time.sleep(1)
#
#                 return SID
#     #except:
#     #    print("--------Fail plotting:"+ S.ID)
#     #    return S.ID
#
#
# def worker2(qq, SID_plotlist, q_back, plot_path ):
#     print('worker runs')
#
#     for SS in iter(qq.get, None):
#
#         #print(np.nanmax(SS.masked_data), SS.masked_data.shape)
#         print("worker received:", SS.ID)
#         try:
#             Figure=SS.plot_fitted_model(flim=(SS.geo['f_low'], SS.geo['f_high']), datasub=SS.masked_data)
#
#
#             Figure.save(name=SS.ID+'_fit_stat', path=plot_path+'fitted_models/', verbose=True)
#             plt.close('all')
#
#             SID_plotlist.append(SS.ID)
#
#             print("worker send:", SS.ID)
#             q_back.put(SS.ID)
#
#         except:
#             print('plot of '+ SS.ID +'Failed')
#             plt.close('all')
#             q_back.put(SS.ID)
#             pass
#
#     q_back.put('fail')
#     print('worker quits')
#     #return
#
# def worker_linear(qq,SID_plotlist, q_back, save_flag=False):
#     SS=qq.get()
#     Figure=SS.plot_fitted_model(flim=(SS.geo['f_low'], SS.geo['f_high']), datasub=SS.masked_data)
#
#     if save_flag:
#         Figure.save(name=SS.ID+'_fit_stat', path=plot_path+'fitted_models/', verbose=True)
#     #plt.close('all')
#     q_back.put(SS.ID)
