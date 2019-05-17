



def load_IRIS_seismic_data(arg):
    """
    This is a dummy for loading time series data of the IRIS data like.
    """
    pass


def load_wave_bouy_spectrograms(arg):
    """
    This is a dummy for loading data of from CDIP/NOAA wave bouys
    """
    pass


def build_timestamp(time, unit, start, verbose=True): #G.time,'h','1900-01-01' ):
    import numpy as np

    timestamp=np.datetime64(start)+time[:].astype('m8['+unit+']')
    #print('timespan:', timestamp[0], '-', timestamp[-1])
    if verbose is True:
        print(timestamp)
    return timestamp

def stats_format(a, name=None):
	print('Name:', str(name),'   Shape:' , a.shape ,'   NaNs:',np.sum(np.isnan(a)),' max:', np.nanmax(a),' min', np.nanmin(a),' mean:', np.nanmean(a))


def init_from_input(arguments):
    if (len(arguments) <= 1) | ('-f' in set(arguments) ) :
        print('no config file found :/' )
        STID='DR01'
        POL='LHN'
        params_file='test1'

        print('use standard values')

    else:

        STID=arguments[1]
        POL=arguments[2]
        params_file=arguments[3]
        #$(hemisphere) $(coords) $(config)

        print('read vars from file: ' +str(arguments[3]) )

    print(STID, POL, params_file)

    if (len(arguments) == 5):
        num = arguments[4]
        print('ID number found, ID#= '+str(num) )

    #ID=STID+'.'+coords+'.'+params_file

    if (len(arguments) == 5):
        return STID, POL, params_file, year
    else:
        return STID, POL, params_file

# def grap_winds(file, startdate=None, enddate=None):
#
#     G=grap_file(file, startdate=startdate, enddate=enddate)
#     k=list(G.variables.keys())
#     print(k)
#     G=gridded_data_netCDF(G, data=k[4:], time=k[3],  lon=k[0], lat=k[1], level=k[2])
#
#     G.timestamp=np.array(build_timestamp(G.time[:],'h','1900-01-01T00:00:00', verbose=False))
#     #print(type(G.timestamp))
#     # if type(G.timestamp) is not np.array:
#     #     G.timestamp = G.timestamp.data
#
#     return G



# class gridded_data_netCDF(object):
#     def __init__(self, G, data=None, unit=None,lat=None, lon=None, time=None, meta=None, mask=None, level=None):
#         # Takes Variable key and links them to standart names
#         self.netCDF=G if G is not None else None
#
#         self.lat=G.variables[lat] if lat is not None else None
#         self.lon=G.variables[lon] if lon is not None else None
#         self.time=G.variables[time] if time is not None else None
#         self.mask=G.variables[mask] if mask is not None else None
#         self.level=G.variables[level] if level is not None else None
#         #print(len(data))
#         if isinstance(data, list):
#             self.var=dict()
#             for i in data:
#                 self.var[i]=G.variables[i]
#         else:
#             self.var=G.variables[data]
#
#         self.data=self.var
#
#         self.meta=meta
#         self.unit=unit
#     def info(self):
#         print(self.netCDF)
#
#     def stats(self):
#         if isinstance(self.data, dict):
#             for key in self.data.keys():
#                 MT.stats_format(self.data[key], name=key)
#         else:
#             MT.stats_format(self.data, name='Data')
#         MT.stats_format(self.lat, name='lat')
#         MT.stats_format(self.lon, name='lon')
#         MT.stats_format(self.time, name='time')
#     def apply_mask(self, fill_value=0):
#         print('use var_filled')
#         self.var_filled=self.var[:].filled(fill_value)
#         self.data_filled=self.var_filled


# def grap_file(file, startdate=None, enddate=None, var=None):
#     from cdo import *   # python version
#     cdo = Cdo()
#     if startdate == None:
#         return cdo.readCdf(file)
#     elif enddate == None:
#         datestring=startdate
#         return cdo.seldate(datestring, input =file, returnCdf  =  True)
#     else:
#         datestring=startdate+','+enddate
#         return cdo.seldate(datestring, input =file, returnCdf  =  True)
