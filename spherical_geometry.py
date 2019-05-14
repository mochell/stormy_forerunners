import sys
sys.path.append('/Users/laure/Desktop/stage/travail/2019_swell_NP/modules/')

import matplotlib.pyplot as plt
import numpy as np

import datetime as DT
import stormy_forerunners.general as M


# estimating distances from sloped arrivals


def MS1957(m, b, verbose=False):
    g=9.8196
    r0=g /(4*np.pi*m)
    t0=-b/m
    if verbose:
        print('Radius (m):', r0 )
        print('t0 (sec):', t0)
    return r0, t0

def normalized_fetch(area, u):
    g=9.81
    return np.sqrt(area)*g / u**2


def event_orign_MS1957(timestamp,f, data, f1, f2, f_delta=0.04, spreed=10, verbose=False, mode='free_limits'):
    dtstamp_local=(timestamp[1]-timestamp[0]).astype('m8[s]')
    dt_local=dtstamp_local.astype(int)
    print(dtstamp_local, dt_local )
    time_lin=np.arange(0,timestamp.size*dt_local, dt_local) # seconds

    STR=M.find_max_along_line(time_lin, f, data, f1, f2, f_delta=f_delta, spreed=spreed,  plot=verbose, mode=mode)

    index=STR['index']

    time_low=(time_lin-dt_local*index).astype(int)
    #print(time_low[0:5])

    time_lowstamp=time_low.astype('timedelta64[s]').astype('m8[h]')
    time_lowstamp_sec=time_low.astype('timedelta64[s]').astype('m8[s]')
    #print(time_lowstamp[0])

    #dt=np.timedelta64(timestamp[1]-timestamp[0]).astype('timedelta64[s]').astype(int)
    slope, intercept, line_time, line_predicted=M.RAMSAC_regression_bootstrap(time_low[STR['t_pos']],
        np.asarray(STR['freq']).T,
        time_lin_arg=time_low,
        plot=False,  alpha=0.05, n_samples=600)
    #slope, intercept, line_time, line_predicted=M.robust_regression(np.asarray(STR['t_pos']).T*dt,
    #    np.asarray(STR['freq']).T, plot=False)

    print('slope:', slope)
    print('intercept:', intercept)

    MS1957_results=dict()
    org = {'slope': slope, 'slope_unit':'1/s/s',
        'intercept': intercept, 'line_time': line_time,
        'line_predicted':line_predicted,
        'timestamp':timestamp, 'time_sec':time_lin}

    MS1957_results['org']=org

    r0, t0=MS1957(slope,intercept , verbose=True)
    MS1957_results['r0']=r0
    MS1957_results['r0_unit']='m'
    MS1957_results['t0']=t0
    MS1957_results['t0_unit']='sec'
    MS1957_results['t0_stamp']=t0.astype('timedelta64[s]')
    MS1957_results['t0_hours']=t0.astype('timedelta64[s]').astype('m8[h]')
    print('t0 mean:', M.echo_dt(MS1957_results['t0_stamp'][0]))
    print('t0 5%:', M.echo_dt(MS1957_results['t0_stamp'][1]))
    print('t0 95%:', M.echo_dt(MS1957_results['t0_stamp'][2]))


    MS1957_results['time_index']=time_low
    MS1957_results['time_indextamp']=time_lowstamp
    MS1957_results['time_indextamp_sec']=time_lowstamp_sec

    if verbose:
        #print(time_lin.shape)
        #print(line_time.shape)
        #print(line_predicted)

        F=plt.figure()
        plt.pcolor(time_low, f, data,cmap='Greys')
        plt.scatter(time_low[STR['t_pos']], STR['freq'], s=30, color='blue', alpha=.8)

        plt.plot(STR['left_limit']-(dt_local*index).astype(int), f, LineWidth=2, Color='green', alpha=.5)
        plt.plot(STR['right_limit']-(dt_local*index).astype(int), f, LineWidth=2, Color='green', alpha=.5)
        print(time_low.shape, line_predicted.shape)
        plt.plot(time_low, line_predicted.T, LineWidth=1, Color='red', alpha=1)

        plt.plot(time_low, time_low*0+f1, LineWidth=2, Color='black', alpha=1)
        plt.plot(f*0, f, LineWidth=2, Color='black', alpha=1)
        plt.plot(time_low, time_low*0+f2, LineWidth=2, Color='green', alpha=.5)

        #plt.plot(time_low, time_low*slope[0]+intercept[0], LineWidth=2, Color='black', alpha=.8)
        #plt.plot(time_lin, line_predicted[1,:], LineWidth=2, Color='orange', alpha=.8)
        #plt.plot(time_lin, line_predicted[2,:], LineWidth=2, Color='orange', alpha=.8)
        plt.xlim(line_predicted[0,0]-(dt_local*index).astype(int), line_predicted[2,-1]-(dt_local*index).astype(int))
        plt.ylim(f[0] , np.max(STR['freq'])*1.1)#STR['freq'].max())#$line_predicted.max())
        #plt.set_yscale("log", nonposy='clip')

        ax=plt.gca()

        ax.set_xticks(MS1957_results['time_index'][3+5::8], minor=False)
        ax.set_xticklabels(MS1957_results['time_indextamp'][3+5::8], minor=False)
    return MS1957_results, STR

# Using great circles distances to get circles of equal distance on the globe
# from http://www.geophysique.be/2011/02/20/matplotlib-basemap-tutorial-09-drawing-circles/
def shoot(lon, lat, azimuth, maxdist=None):
    """Shooter Function
    Original javascript on http://williams.best.vwh.net/gccalc.htm
    Translated to python by Thomas Lecocq
    """
    glat1 = lat * np.pi / 180.
    glon1 = lon * np.pi / 180.
    s = maxdist / 1.852
    faz = azimuth * np.pi / 180.

    EPS= 0.00000000005
    if ((np.abs(np.cos(glat1))<EPS) and not (np.abs(np.sin(faz))<EPS)):
        alert("Only N-S courses are meaningful, starting at a pole!")

    a=6378.13/1.852
    f=1/298.257223563
    r = 1 - f
    tu = r * np.tan(glat1)
    sf = np.sin(faz)
    cf = np.cos(faz)
    if (cf==0):
        b=0.
    else:
        b=2. * np.arctan2 (tu, cf)

    cu = 1. / np.sqrt(1 + tu * tu)
    su = tu * cu
    sa = cu * sf
    c2a = 1 - sa * sa
    x = 1. + np.sqrt(1. + c2a * (1. / (r * r) - 1.))
    x = (x - 2.) / x
    c = 1. - x
    c = (x * x / 4. + 1.) / c
    d = (0.375 * x * x - 1.) * x
    tu = s / (r * a * c)
    y = tu
    c = y + 1
    #print('in shoot', y, c)
    while (np.abs (y - c) > EPS):

        sy = np.sin(y)
        cy = np.cos(y)
        cz = np.cos(b + y)
        e = 2. * cz * cz - 1.
        c = y
        x = e * cy
        y = e + e - 1.
        y = (((sy * sy * 4. - 3.) * y * cz * d / 6. + x) *
              d / 4. - cz) * sy * d + tu

    b = cu * cy * cf - su * sy
    c = r * np.sqrt(sa * sa + b * b)
    d = su * cy + cu * sy * cf
    glat2 = (np.arctan2(d, c) + np.pi) % (2*np.pi) - np.pi
    c = cu * cy - su * sy * cf
    x = np.arctan2(sy * sf, c)
    c = ((-3. * c2a + 4.) * f + 4.) * c2a * f / 16.
    d = ((e * cy * c + cz) * sy * c + y) * sa
    glon2 = ((glon1 + x - (1. - c) * d * f + np.pi) % (2*np.pi)) - np.pi

    baz = (np.arctan2(sa, b) + np.pi) % (2 * np.pi)

    glon2 *= 180./np.pi
    glat2 *= 180./np.pi
    baz *= 180./np.pi

    return (glon2, glat2, baz)

def equi_north(m, centerlon, centerlat, radius, angle_range=None, *args, **kwargs):
    glon1 = centerlon
    glat1 = centerlat
    X = []
    Y = []
    angle_range=angle_range if angle_range is not None else (-120, 120)
    #print(angle_range)
    for azimuth in range(angle_range[0], angle_range[1]):
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        X.append(glon2)
        Y.append(glat2)
    #X.append(X[0])
    #Y.append(Y[0])

    #~ m.plot(X,Y,**kwargs) #Should work, but doesn't...
    #print(X)
    X,Y = m(X,Y)

    plt.plot(X,Y,**kwargs)

def equi_north_360(m, centerlon, centerlat, radius, angle_range=None,deg_res=1, *args, **kwargs):
    # radius in km
    glon1 = centerlon+180
    glat1 = centerlat
    X = []
    Y = []
    angle_range=angle_range if angle_range is not None else (-180, 180)
    #print(angle_range)
    for azimuth in np.arange(angle_range[0], angle_range[1], deg_res):
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        X.append(glon2)
        Y.append(glat2)
    #X.append(X[0])
    #Y.append(Y[0])

    X180 = [i+180 for i in X]
    Xm,Ym = m(X180,Y)

    plt.plot(Xm,Ym,**kwargs)
    return X180, Y


def create_great_cirle_on_map(m, centerlon, centerlat, radius, angle_range=None,deg_res=1):

    # radius in km
    glon1 = centerlon+180
    glat1 = centerlat
    X = []
    Y = []
    angle_range=angle_range if angle_range is not None else (-180, 180)
    #print(angle_range)
    for azimuth in np.arange(angle_range[0], angle_range[1], deg_res):
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        X.append(glon2)
        Y.append(glat2)
    #X.append(X[0])
    #Y.append(Y[0])

    X180 = [i+180 for i in X]
    Xm,Ym = m(X180,Y)
    return Xm, Ym

def equidistant_circle_lon_lat(centerlon, centerlat, radius, *args, **kwargs):
    ''' from http://www.geophysique.be/2011/02/20/matplotlib-basemap-tutorial-09-drawing-circles/
        radius in km

    '''
    glon1 = centerlon
    glat1 = centerlat
    X = []
    Y = []
    for azimuth in range(0, 360):
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        X.append(glon2)
        Y.append(glat2)
    X.append(X[0])
    Y.append(Y[0])

    return X, Y

def equi(m, centerlon, centerlat, radius, *args, **kwargs):
    ''' from http://www.geophysique.be/2011/02/20/matplotlib-basemap-tutorial-09-drawing-circles/
    '''
    glon1 = centerlon
    glat1 = centerlat
    X = []
    Y = []
    for azimuth in range(0, 360):
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        X.append(glon2)
        Y.append(glat2)
    X.append(X[0])
    Y.append(Y[0])

    #~ m.plot(X,Y,**kwargs) #Should work, but doesn't...
    X,Y = m(X,Y)
    plt.plot(X,Y,**kwargs)
    #return X, Y

def haversine(lon1, lat1, lon2, lat2, arc=False):
    from math import radians, cos, sin, asin, sqrt
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    arc     True: returns radians [0 , pi]
            False: returns km
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    if arc:
        return c
    else:
        return c * r

def bearing(lon1, lat1, lon2, lat2):
    """
    calculates the angle between two points on the globe
    output:
    initial_bearing  from -180 to + 180 deg
    """
    from math import radians, cos, sin, atan2, degrees
    #if (type(pointA) != tuple) or (type(pointB) != tuple):
    #    raise TypeError("Only tuples are supported as arguments")

    lat1 = radians(lat1)
    lat2 = radians(lat2)

    diffLong = radians(lon2 - lon1)

    x = sin(diffLong) * cos(lat2)
    y = cos(lat1) * sin(lat2) - (sin(lat1)
            * cos(lat2) * cos(diffLong))

    initial_bearing = atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180 to + 180 which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return initial_bearing
