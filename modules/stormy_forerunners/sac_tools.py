import matplotlib.pyplot as plt
import numpy as np
import os,sys

from obspy import read, read_inventory
import stormy_forerunners.tools as MT

def sac_load_all(path, noplot=None):
	st = read(path)
	print(st)
	#st.stats
	G=st
	if noplot==True:
		G.plot()
	return G


def sac_load(path, noplot=None):
	st = read(path)
	print(st)
	#st.stats
	G=st[0]
	if noplot==True:
		G.plot()
	return G
#def convert_timesstamp(var):


def build_time(G,start_time=None, end_time=None, dt=None, verbose=True):
	"""build timeseries from given SAC file or define start_time or end_time
	dates can be defines as: '2014-12-01T12:00:14.000004'
	or simpler: 			 '2014-12-01'
	dt is format [1, 's']  or similar"""
	file_start=np.datetime64(G.meta.starttime).astype(np.datetime64)
	file_end=np.datetime64(G.meta.endtime).astype(np.datetime64)
	if verbose == True:
		print('File')
		print('start:', file_start)
		print('end:', file_end)

	dt_data=np.timedelta64(int(G.stats.sampling_rate),'s')  if dt is None else np.timedelta64(dt[0],dt[1])
	start_time=start_time if start_time is not None else file_start
	end_time=end_time if end_time is not None else file_end

	timearray = np.arange(file_start, file_end+dt_data,dt_data)
	G.dt_data=dt_data

	timecut1=[timearray >= np.datetime64(start_time)]
	timecut2=[timearray <= (np.datetime64(end_time)-dt_data)]
	timecut=[timecut1 & timecut2 for timecut1, timecut2 in zip(timecut1, timecut2)]
	G.data=G.data[timecut]
	G.timestamp=timearray[timecut]
	if verbose == True:
		if np.datetime64(start_time) == file_start:
			if np.datetime64(end_time) == file_end:
				print('No new time range set')
			else:
				print('new start date:', start_time)
				print('      end date:', end_time)
		else:
			print('new start date:', start_time)
			print('      end date:', end_time)

		#update startime
		#G.meta.starttime=start_time
		#G.meta.endtime=end_time

		print(G.timestamp)
	return G

def Station(path,start_time=None, end_time=None, verbose=True, unit=None, factor=None):
	if unit is not None:
		unit=unit
	else:
		if path.find('DIS') != -1:
			unit='m'
		elif path.find('VEL') != -1:
			unit='m/s'
		elif path.find('ACC') != -1:
			unit='m/s/s'
		else:
			unit='?'

	unit='m' if unit is None else unit

	G=sac_load(path, noplot=verbose)

	if factor is not None:
		G.data=G.data*factor
		G.factor=factor
	G.path=path
	G.unit=unit
	G=build_time(G,start_time,end_time, verbose=verbose)
	return G

def plot_station(DR01, res=None):
	import datetime as DT
	import m_general as M

	DR01.unit='?' if DR01.unit is None else DR01.unit
	res=.1 if res is None else res
	if DR01.unit is '?':
		print('Please set unit')

	F=M.figure_axis_xy(12,5)

	tt = DR01.timestamp.astype(DT.datetime)
	resr=int(round(1/res))
	print('only every ', str(resr), 'timestep is plottet')
	plt.plot(tt[::resr], DR01.data[::resr], 'k')
	F.label('time', DR01.unit, DR01.meta.station+' '+DR01.meta.channel+' '+str(DR01.meta.starttime.date)+' '+str(DR01.meta.endtime.date))
	return F
