'''
Required utility functions from molsim (https://github.com/bmcguir2/molsim)
'''


from numba import njit
import math
from collections import defaultdict
import numpy as np
from scipy import signal
import re
from ..constants import ccm, cm, ckm, h, k, kcm 



def _read_txt(filein):
	'''Reads in any txt file and returns a line by line array'''
	
	return_arr = []
	
	with open(filein, 'r') as input:
		for line in input:
			return_arr.append(line)

	return return_arr


@njit
def get_rms(y,sigma=3):
	tmp_y = np.copy(y)
	i = np.nanmax(tmp_y)
	rms = np.sqrt(np.nanmean(np.square(tmp_y)))
	
	while i > sigma*rms:
		tmp_y = tmp_y[tmp_y<sigma*rms]
		rms = np.sqrt(np.nanmean(np.square(tmp_y)))
		i = np.nanmax(tmp_y)

	return rms


def _find_ones(arr):
	'''
	Find the start,[stop] indices where value is present in arr
	'''

	# Create an array that is 1 where a is 0, and pad each end with an extra 0.
	# here .view(np.int8) changes the np.equal output from a bool array to a 1/0 array
	new_arr = np.copy(arr)
	iszero = np.concatenate(([0], np.equal(arr, 1).view(np.int8), [0]))
	absdiff = np.abs(np.diff(iszero))
	# Runs start and end where absdiff is 1.
	ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

	lls = [x[0] for x in ranges]
	uls = [x[1] for x in ranges]

	return lls,uls


def _trim_arr(arr,lls,uls,key_arr=None,return_idxs=False,ll_idxs=None,ul_idxs=None,return_mask=False):
	'''
	Trims the input array to the limits specified.  Optionally, will get indices from
	the key_arr for trimming instead.
	'''

	if ll_idxs is not None:
		return np.concatenate([arr[ll_idx:ul_idx] for ll_idx,ul_idx in zip(ll_idxs,ul_idxs)])
	# modified to set as False to begin with, and working with
	# booleans instead of numbers
	mask_arr = np.zeros_like(arr, dtype=bool)
	if key_arr is None:
		for x,y in zip(lls,uls):
			mask_arr[(arr>x) & (arr<y)] = True
	else:
		for x,y in zip(lls,uls):
			mask_arr[(key_arr>x) & (key_arr<y)] = True
	if return_mask:
		return mask_arr
	if return_idxs is False:
		return arr[mask_arr]
	else:
		ll_idxs_out = _find_ones(mask_arr)[0]
		ul_idxs_out = _find_ones(mask_arr)[1]
		return arr[mask_arr],ll_idxs_out,ul_idxs_out
	
def find_nearest(arr,val):
	idx = np.searchsorted(arr, val, side="left")
	if idx > 0 and (idx == len(arr) or math.fabs(val - arr[idx-1]) \
		 < math.fabs(val - arr[idx])):
		return idx-1
	else:
		return idx
	
def find_nearest_vectorized(arr,val_arr):
    idxs = np.searchsorted(arr, val_arr, side="left")
    for i in range(len(val_arr)):
        if idxs[i] > 0 and (idxs[i] == len(arr) or math.fabs(val_arr[i] - arr[idxs[i]-1]) \
                           < math.fabs(val_arr[i] - arr[idxs[i]])):
            idxs[i] = idxs[i]-1
        return list(idxs)
	
@njit
def _make_gauss(freq0,int0,freq,dV,ckm):
	return int0*np.exp(-((freq-freq0)**2/(2*((dV*freq0/ckm)/2.35482)**2)))

def _apply_vlsr(frequency,vlsr):
	'''
	Applies a vlsr shift to a frequency array.  Frequency in [MHz], vlsr in [km/s]
	'''
	return frequency - vlsr*frequency/ckm

def _apply_beam(freq_arr,int_arr,source_size,dish_size,return_beam=False):
	beam_size = 206265 * 1.22 * (cm/(freq_arr * 1E6)) / dish_size #get beam size in arcsec
	beam_dilution = source_size**2 / (beam_size**2 + source_size**2)
	if return_beam is False:
		return int_arr*beam_dilution
	else:
		return int_arr*beam_dilution,beam_dilution
	
def _make_fmted_qnstr(qns,qnstr_fmt=None):
	'''
	Given a qnstr_fmt formatter declaration, turns a set of quantum numbers into a
	human readable output.

	For example, for methanol with some conditions, we want the final output to look like:

	1(1)-A vt=0 for the upper state of the 834.28 transition that has catalog qns of "1 1 - 0," we would use:

	'/#1/(/#2/)/#3[+=+ A,-=- A,= E]/ vt=/#4/'

	'''

	#if a qnstr_fmt is not given, return a nicely cleaned up string
	if qnstr_fmt is None:
		qn_bits = [f'{x:>3}' for x in qns]
		qnstr = ' '
		return qnstr.join(qn_bits)

	#Clean up the formatting input a bit
	base_str = qnstr_fmt.split('/')

	if base_str[0] == '':
		del base_str[0]
	if base_str[-1] == '':
		del base_str[-1]

	#apply the formatting
	for x in range(len(base_str)):
		if '#' in base_str[x] and '[' not in base_str[x]:
			base_str[x] = base_str[x].replace('#','')
			idx = int(base_str[x])
			base_str[x] = str(qns[idx-1])

		if '#' in base_str[x] and '[' in base_str[x]:
			conditions = base_str[x].split('[')[1].replace(']','').split(',')
			idx = int(base_str[x].split('[')[0].replace('#',''))

			for y in range(len(conditions)):
				conditions[y] = conditions[y].split('=')
			value = str(qns[idx-1])

			for y in range(len(conditions)):
				if conditions[y][0] == value:
					base_str[x] = str(conditions[y][1])

	#make the string and return it
	qnstr = ''
	return qnstr.join(base_str)

def _read_txt(filein):
	'''Reads in any txt file and returns a line by line array'''
	
	return_arr = []
	
	with open(filein, 'r') as input:
		for line in input:
			return_arr.append(line)

	return return_arr

def _read_xy(filein):
	'''Reads in a two column x y file and returns the numpy arrays	'''
	
	x = []
	y = []
	
	with open(filein, 'r') as input:
		for line in input:
			x.append(float(line.split()[0].strip()))
			y.append(float(line.split()[1].strip()))
	
	x = np.asarray(x)
	y = np.asarray(y)
			
	return x,y		


def _read_spectrum(filein):
	'''
	Reads in an npz saved spectrum.  Returns a spectrum object.
	'''
	from .molsim_classes import Spectrum
	npz_dict = np.load(filein,allow_pickle=True)
	new_dict = defaultdict(lambda: None)
	for x in npz_dict:
		new_dict[x] = npz_dict[x]
	#sort some back to strings from numpy arrays
	entries = ['name','notes']
	for entry in entries:
		if entry in new_dict:
			new_dict[entry] = str(new_dict[entry])
	
	spectrum = Spectrum(freq0 = new_dict['freq0'],
						frequency = new_dict['frequency'],
						Tb = new_dict['Tb'],
						Iv = new_dict['Iv'],
						Tbg = new_dict['Tbg'],
						Ibg = new_dict['Ibg'],
						tau = new_dict['tau'],
						tau_profile = new_dict['tau_profile'],
						freq_profile = new_dict['freq_profile'],
						int_profile = new_dict['int_profile'],
						Tbg_profile = new_dict['Tbg_profile'],
						velocity = new_dict['velocity'],
						int_sim = new_dict['int_sim'],
						freq_sim = new_dict['freq_sim'],
						snr = new_dict['snr'],
						noise = new_dict['noise'],
						id = new_dict['id'],
						notes = new_dict['notes'],
						name = new_dict['name']
						)
						
	return spectrum		

def load_obs(filein=None,xunits='MHz',yunits='K',id=None,notes=None,spectrum_id=None,spectrum_notes=None,source_dict=None,source=None,continuum_dict=None,continuum=None,observatory_dict=None,observatory=None,type='molsim'):
	
	'''
	Reads in an observations file and initializes an observation object with the given attributes.
	'''
	from .molsim_classes import Observation, Spectrum, Source, Continuum, Observatory 
	#initialize an Observation object
	obs = Observation(spectrum=Spectrum())
	
	type = type.lower()
	
	#read in the data if there is any
	if filein is not None:
		#if the file was previously a molsim formatted .npz file
		if type == 'molsim':
			obs.spectrum = _read_spectrum(filein)
		#if the file is an alma ispec file, we get some of the info from the header	
		elif type == 'ispec':
			#read in the file into a temporary array
			raw_arr = _read_txt(filein)
			#eliminate all empty lines
			raw_arr = [x for x in raw_arr if x != '\n']
			#separate out the comments that will have metadata, then make a dictionary
			metadata_keys = [x.split(':')[0].strip('#').strip().lower() for x in raw_arr if x[0] == '#' and ':' in x]
			#the extra joining here preserves the colons in coordinate designations
			metadata_vals = [':'.join(x.split(':')[1:]).strip().lower() for x in raw_arr if x[0] == '#' and ':' in x]
			metadata = dict(zip(metadata_keys,metadata_vals))
			#separate out the data
			x = np.array([float(line.split()[0].strip()) for line in raw_arr if line[0] != '#'])
			y = np.array([float(line.split()[1].strip()) for line in raw_arr if line[0] != '#'])
			#make sure the data is in increasing frequency order
			sort_idx = np.argsort(x)
			x = x[sort_idx]
			y = y[sort_idx]
			#sort through the metadata
			#look for frequency units
			if 'xlabel' in metadata.keys():
				if '[' in metadata['xlabel']:
					xunits = metadata['xlabel'].split('[')[1].strip(']')
				elif '(' in metadata['xlabel']:
					xunits = metadata['xlabel'].split('(')[1].strip(')')
				else:
					print('Unable to interpret frequency units from *.ispec file, assumed to be MHz')
					xunits = 'mhz'
			#change to MHz if required
			convert_dict = {'mhz' : 1.0,
							'ghz' : 1000.,
							'thz' : 1000000.,
							'khz' : 0.001}
			if xunits:
				x *= convert_dict[xunits]
			#look for intensity units
			if 'ylabel' in metadata.keys():
				yunits = metadata['ylabel'].split('[')[1].split(']')[0]
			
			obs.spectrum = Spectrum(frequency=x,Tb=y,notes=yunits)							
		else:
			x,y = _read_xy(filein)
			if xunits == 'GHz':
				x*=1000
				xunits = 'MHz'
			obs.spectrum.frequency = x
			if yunits == 'K':
				obs.spectrum.Tb = y
			elif yunits.lower() == 'jy/beam':
				obs.spectrum.Iv = y		
	
	if id is not None:
		obs.id = id
	if spectrum_id is not None:
		obs.spectrum.id = spectrum_id
	if notes is not None:
		obs.notes = notes
	if spectrum_notes is not None:
		obs.spectrum.notes = spectrum_notes
		
	if source is not None:
		obs.source = source
	elif source_dict is not None:
		obs.source = Source(
								name = source_dict['name'] if 'name' in source_dict else None,
								coords = source_dict['coords'] if 'coords' in source_dict else None,
								velocity = source_dict['velocity'] if 'velocity' in source_dict else 0.,
								size = source_dict['size'] if 'size' in source_dict else 1E20,
								solid_angle = source_dict['solid_angle'] if 'solid_angle' in source_dict else None,
								column = source_dict['column'] if 'column' in source_dict else 1.E13,
								Tex = source_dict['Tex'] if 'Tex' in source_dict else 300,
								Tkin = source_dict['Tkin'] if 'Tkin' in source_dict else None,
								dV = source_dict['dV'] if 'dV' in source_dict else 3.,
								notes = source_dict['notes'] if 'notes' in source_dict else None,	
							)
							
	if continuum is not None:
		obs.source.continuum = continuum
	elif continuum_dict is not None:
		obs.source.continuum = Continuum(
											cont_file = continuum_dict['cont_file'] if 'cont_file' in continuum_dict else None,
											type = continuum_dict['type'] if 'type' in continuum_dict else 'thermal',
											params = continuum_dict['params'] if 'params' in continuum_dict else [2.7],
											freqs = continuum_dict['freqs'] if 'freqs' in continuum_dict else None,
											temps = continuum_dict['temps'] if 'temps' in continuum_dict else None,
											fluxes = continuum_dict['fluxes'] if 'fluxes' in continuum_dict else None,
											notes = continuum_dict['notes'] if 'notes' in continuum_dict else None,
										)
	if observatory is not None:
		obs.observatory = observatory									
	elif observatory_dict is not None:
		obs.observatory = Observatory(
										name = observatory_dict['name'] if 'name' in observatory_dict else None,
										id = observatory_dict['id'] if 'id' in observatory_dict else None,
										sd = observatory_dict['sd'] if 'sd' in observatory_dict else True,
										array = observatory_dict['array'] if 'array' in observatory_dict else False,
										dish = observatory_dict['dish'] if 'dish' in observatory_dict else 100.,
										synth_beam = observatory_dict['synth_beam'] if 'synth_beam' in observatory_dict else [1.,1.],
										loc = observatory_dict['loc'] if 'loc' in observatory_dict else None,
										eta = observatory_dict['eta'] if 'eta' in observatory_dict else None,
										eta_type = observatory_dict['eta_type'] if 'eta_type' in observatory_dict else 'Constant',
										eta_params = observatory_dict['eta_params'] if 'eta_params' in observatory_dict else [1.],
										atmo = observatory_dict['atmo'] if 'atmo' in observatory_dict else None,
									)	
	elif type == 'ispec':
		obs.observatory = Observatory(
										name = 'ALMA',
										sd = False,
										array = True,
									)																

	return obs		

def find_limits(freq_arr,spacing_tolerance=100,padding=0):
	'''
	Finds the limits of a set of data, including gaps over a width, determined by the
	spacing tolerance.  Optional padding to each side to allow user to change vlsr and get
	the simulation within the right area.
	'''

	# if len(freq_arr) == 0:
	# 	print('The input array has no data.')
	# 	return

	# this algorithm compares each gap against the average spacing of spacing_tolerance nearby points
	# if the gap is larger than spacing_tolerance * average spacing in both directions, it is considered as the actual gap
	# simplifying it gives the following expressions

	t = spacing_tolerance
	center = freq_arr[t+1:-t] - freq_arr[t:-t-1]
	left = freq_arr[t:-t-1] - freq_arr[:-2*t-1]
	right = freq_arr[2*t+1:] - freq_arr[t+1:-t]

	gaps = np.concatenate(([-1], np.where(np.logical_and(center > left, center > right))[0] + t, [-1]))

	ll = freq_arr[gaps[:-1]+1]
	ul = freq_arr[gaps[1:]]

	# the original expressions require 1 addition, 1 multiplication and 1 division operations per element, i.e. 3N operations, plus memory allocation and deallocation of temporary array
	# the following use 1 multiplication per element plus 1 addition and 1 division, i.e. N+2 operations, without allocating temporary array
	ll *= 1 - padding/ckm
	ul *= 1 + padding/ckm

	return ll,ul

def find_peaks(freq_arr,int_arr,res,min_sep,is_sim=False,sigma=3,kms=True, rms=None):
	'''
	'''
	#if is_sim == False:
	#	print('molsim find peaks rms no sim', rms)
	#else:
	#		print('molsim find peaks rms sim', rms)


	if kms is True:
		max_f = np.amax(freq_arr)
		min_f = np.amin(freq_arr)
		cfreq = (max_f + min_f)/2
		v_res = res*ckm/max_f #finest velocity spacing
		v_span = (max_f - min_f) * ckm/(cfreq) #total velocity range spanned, setting cfreq at v=0.
		v_samp = np.arange(-v_span/2,v_span/2+v_res,v_res) #create a uniformly spaced velocity array
		freq_new = v_samp*cfreq/ckm + cfreq #convert it back to frequency
		int_new = np.interp(freq_new,freq_arr,int_arr,left=0.,right=0.)
		chan_sep = min_sep/v_res
	else:
		freq_new = freq_arr
		int_new = int_arr
		chan_sep = min_sep/res

	indices = signal.find_peaks(int_new,distance=chan_sep)

	if kms is True:
		indices = [find_nearest(freq_arr,freq_new[x]) for x in indices[0]] #if we had to re-sample things

	if is_sim is True:
		return np.asarray(indices)

	if rms == None:
		rms = get_rms(int_arr)
		indices = [x for x in indices if int_arr[x]>sigma*rms ]
	else:
		indices = [x for x in indices if int_arr[x]>sigma*rms ]

	return np.asarray(indices)

def load_mol(filein,type='molsim',catdict=None,id=None,name=None,formula=None,
				elements=None,mass=None,A=None,B=None,C=None,muA=None,muB=None,
				muC=None,mu=None,Q=None,qnstrfmt=None,partition_dict=None,
				qpart_file=None):

	'''
	Loads a molecule in from a catalog file.  Default catalog type is molsim.  Override
	things with catdict.  Generates energy level objects, transition objects, a partition
	function object and	a molecule object which it returns.
	'''
	from .molsim_classes import Level, Molecule, PartitionFunction
	#load the catalog in
	cat = _load_catalog(filein,type=type,catdict=catdict)

	#now we have to make a hash for every entries upper and lower state.  If this already
	#exists in the catalog, great.  If not, we have to make it.
	if cat.qnup_str is None:
		qnups = [cat.qn1up,cat.qn2up,cat.qn3up,cat.qn4up,cat.qn5up,cat.qn6up,cat.qn7up,
					cat.qn8up]	
		qn_list_up = [_make_qnstr(qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8) for qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8 in zip(cat.qn1up,cat.qn2up,cat.qn3up,cat.qn4up,cat.qn5up,cat.qn6up,cat.qn7up,cat.qn8up)]
	else:
		qn_list_up = cat.qnup_str
		
	if cat.qnlow_str is None:
		qnlows = [cat.qn1low,cat.qn2low,cat.qn3low,cat.qn4low,cat.qn5low,cat.qn6low,
				cat.qn7low,cat.qn8low]
		qn_list_low = [_make_qnstr(qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8) for qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8 in zip(cat.qn1low,cat.qn2low,cat.qn3low,cat.qn4low,cat.qn5low,cat.qn6low,cat.qn7low,cat.qn8low)]
	else:
		qn_list_low = cat.qnlow_str
		
	level_qns = np.concatenate((qn_list_low,qn_list_up),axis=0)
	level_qns = list(set(list(level_qns))) #get the unique ones
	level_dict = dict.fromkeys(level_qns)			
	
	#now we find unique energy levels.  We just get the dictionary of levels, since
	#that function is computationally intensive and we want to njit it.
	
	level_dict = _make_level_dict(
									cat.qn1low,
									cat.qn2low,
									cat.qn3low,
									cat.qn4low,
									cat.qn5low,
									cat.qn6low,
									cat.qn7low,
									cat.qn8low,
									cat.qn1up,
									cat.qn2up,
									cat.qn3up,
									cat.qn4up,
									cat.qn5up,
									cat.qn6up,
									cat.qn7up,
									cat.qn8up,
									cat.frequency,
									cat.elow,
									cat.gup,									
									qn_list_low,
									qn_list_up,
									level_qns,
									level_dict,
									qnstrfmt,
								)
								
	#load those levels into level objects and add to a list
	levels = []
	for x in level_dict:
		levels.append(Level(
							energy = level_dict[x]['energy'],
							g = level_dict[x]['g'],
							g_flag = level_dict[x]['g_flag'],
							qn1 = level_dict[x]['qn1'],
							qn2 = level_dict[x]['qn2'],
							qn3 = level_dict[x]['qn3'],
							qn4 = level_dict[x]['qn4'],
							qn5 = level_dict[x]['qn5'],
							qn6 = level_dict[x]['qn6'],
							qn7 = level_dict[x]['qn7'],
							qn8 = level_dict[x]['qn8'],
							qnstrfmt = level_dict[x]['qnstrfmt'],
							id = level_dict[x]['id']
							))
	levels.sort(key=lambda x: x.energy) #sort them so the lowest energy is first
	
	#we'll now update the catalog with some of the things we've calculated like eup and
	#glow unless they're already present
	if type != 'molsim':
		level_ids = np.array([x.id for i,x in np.ndenumerate(levels)])
		tmp_dict = {}
		for x in levels:
			tmp_dict[x.id] = [x.g,x.energy]
		if cat.glow is None:
			cat.glow = np.empty_like(cat.frequency)	
		if cat.eup is None:
			cat.eup = np.empty_like(cat.frequency)			
		for x in range(len(cat.frequency)):
			cat.glow[x] = tmp_dict[cat.qnlow_str[x]][0]
			cat.eup[x] = tmp_dict[cat.qnup_str[x]][1]
	
	#now we have to load the transitions in	and make transition objects	
		
	#make the molecule
	mol = Molecule(levels=levels,catalog=cat,name=name,formula=formula, elements=elements,
	mass=mass, A=A, B=B, C=C, muA=muA, muB=muB, muC=muC, mu=mu)
	
	#make a partition function object and assign it to the molecule
	#if there's no other info, assume we're state counting
	if partition_dict is None:
		partition_dict = {}
		partition_dict['mol'] = mol
	#if there's a qpart file specified, read that in	
	if qpart_file is not None:
		partition_dict['qpart_file'] = qpart_file
	#make the partition function object and assign it	
	mol.qpart = PartitionFunction(	
				qpart_file = partition_dict['qpart_file'] if 'qpart_file' in partition_dict else None,
				form = partition_dict['form'] if 'form' in partition_dict else None,
				params = partition_dict['params'] if 'params' in partition_dict else None,
				temps = partition_dict['temps'] if 'temps' in partition_dict else None,
				vals = partition_dict['vals'] if 'vals' in partition_dict else None,
				mol = partition_dict['mol'] if 'mol' in partition_dict else None,
				gs = partition_dict['gs'] if 'gs' in partition_dict else None,
				energies = partition_dict['energies'] if 'energies' in partition_dict else None,
				sigma = partition_dict['sigma'] if 'sigma' in partition_dict else 1.,
				vib_states = partition_dict['vib_states'] if 'vib_states' in partition_dict else None,
				vib_is_K = partition_dict['vib_is_K'] if 'vib_is_K' in partition_dict else None,
				notes = partition_dict['notes'] if 'notes' in partition_dict else None,
							)
	
	#set sijmu and aij						
	mol.catalog._set_sijmu_aij(mol.qpart)						
	
	return	mol	

def _make_qnstr(qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8):
	qn_list = [qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8]
	tmp_list = [str(x).zfill(2) for x in qn_list if x != None]
	return ''.join(tmp_list)

def _make_level_dict(qn1low,qn2low,qn3low,qn4low,qn5low,qn6low,qn7low,qn8low,qn1up,qn2up,
					qn3up,qn4up,qn5up,qn6up,qn7up,qn8up,frequency,elow,gup,
					qn_list_low,qn_list_up,level_qns,level_dict,qnstrfmt=None):

	#a list to hold levels
	levels = []

	#we need to sort out unique levels from our catalog.  Those will have unique quantum
	#numbers. When we find a match to a lower level, add the info in.
	for x in range(len(frequency)):
		qnstr_low = qn_list_low[x]
		level_dict[qnstr_low] = {'energy'	:	elow[x],
								 'g'		:	None,
								 'g_flag'	:	False,
								 'qn1'		:	qn1low[x] if qn1low is not None else None,
								 'qn2'		:	qn2low[x] if qn2low is not None else None,
								 'qn3'		:	qn3low[x] if qn3low is not None else None,
								 'qn4'		:	qn4low[x] if qn4low is not None else None,
								 'qn5'		:	qn5low[x] if qn5low is not None else None,
								 'qn6'		:	qn6low[x] if qn6low is not None else None,
								 'qn7'		:	qn7low[x] if qn7low is not None else None,
								 'qn8'		:	qn8low[x] if qn8low is not None else None,
								 'id'		:	qn_list_low[x],
								 'qnstrfmt'	:	qnstrfmt,
								}

	#do it again to fill in energy levels that were upper states and didn't get hit
	for x in range(len(frequency)):
		qnstr_up = qn_list_up[x]
		if level_dict[qnstr_up] is None:
			#calculate the energy.  Move the transition from MHz -> cm-1 -> K
			freq_cm = (frequency[x]*1E6/ccm)
			freq_K = freq_cm / kcm
			level_dict[qnstr_up] = {'energy'	:	elow[x] + freq_K,
									 'g'		:	gup[x],
									 'g_flag'	:	False,
									 'qn1'		:	qn1up[x] if qn1up is not None else None,
									 'qn2'		:	qn2up[x] if qn2up is not None else None,
									 'qn3'		:	qn3up[x] if qn3up is not None else None,
									 'qn4'		:	qn4up[x] if qn4up is not None else None,
									 'qn5'		:	qn5up[x] if qn5up is not None else None,
									 'qn6'		:	qn6up[x] if qn6up is not None else None,
									 'qn7'		:	qn7up[x] if qn7up is not None else None,
									 'qn8'		:	qn8up[x] if qn8up is not None else None,
									 'id'		:	qn_list_up[x],
									 'qnstrfmt'	:	qnstrfmt,
									}

	#go grab the degeneracies
	for x in range(len(frequency)):
		qnstr_up = qn_list_up[x]
		if level_dict[qnstr_up]['g'] is None:
			level_dict[qnstr_up]['g'] = gup[x]

	#now go through and fill any degeneracies that didn't get hit (probably ground states)
	#assume it's just 2J+1.  Set the flag for a calculated degeneracy to True.
	for x in level_dict:
		if level_dict[x]['g'] is None:
			level_dict[x]['g'] = 2*level_dict[x]['qn1'] + 1
			level_dict[x]['g_flag'] = True

	return level_dict

def _load_catalog(filein,type='SPCAT',catdict=None):
	'''
	Reads in a catalog file of the specified type and returns a catalog object.  
	Optionally accepts a catdict dictionary to preload the catalog object with 
	additional information. Defaults to loading an spcat catalog.
	
	Anything in catdict will overwrite what's loaded in from the read catalog
	function, so use cautiously.
	'''
	from .molsim_classes import Catalog

	if type.lower() == 'molsim':
		npz_dict = np.load(filein,allow_pickle=True)	
		new_dict = {}
		for x in npz_dict:
			new_dict[x] = npz_dict[x]
		#sort some back to strings from numpy arrays
		entries = ['version','source','last_update','contributor_name','contributor_email','notes','refs','qnstr_fmt']
		for entry in entries:
			if entry in new_dict:
				new_dict[entry] = str(new_dict[entry])

	elif type.lower() == 'spcat':
		new_dict = _read_spcat(filein) #read in the catalog file and produce the
									   #dictionary
				
	elif type.lower() == 'freq_int':
		freq_tmp,int_tmp = 	_read_xy(filein) #read in a frequency intensity 
												   #delimited file
		new_dict = {}
		new_dict['frequency'] = freq_tmp
		new_dict['man_int'] = int_tmp
	
	if catdict is not None:
		for x in catdict:
			new_dict[x] = catdict[x] #either add it to the new_dict or overwrite it			
		
	cat = Catalog(catdict=new_dict) #make the catalog and return it
	
	return cat

def _read_spcat(filein):
	'''
	Reads an SPCAT catalog and returns spliced out numpy arrays.
	Catalog energy units for elow are converted to K from cm-1.	
	'''
	
	#read in the catalog
	raw_arr = _read_txt(filein)
	
	#set up some basic lists to populate
	frequency = []
	freq_err = []
	logint = []
	dof = []
	elow = []
	gup = []
	tag = []
	qnformat = []
	qn1 = []
	qn2 = []
	qn3 = []
	qn4 = []
	qn5 = []
	qn6 = []
	qn7 = []
	qn8 = []
	qn9 = []
	qn10 = []
	qn11 = []
	qn12 = []
	
	#split everything out	
	for x in raw_arr:
		#if there is a * in there, then the catalog was simulated too high for SPCAT format and we need to skip the line.
		if '*' in x:
			continue
		else:
			frequency.append(x[:13].strip())
			freq_err.append(x[13:21].strip())
			logint.append(x[21:29].strip())
			dof.append(x[29:31].strip())
			elow.append(x[31:41].strip())
			gup.append(x[41:44].strip())
			tag.append(x[44:51].strip())
			qnformat.append(x[51:55].strip())
			qn1.append(x[55:57].strip())
			qn2.append(x[57:59].strip())
			qn3.append(x[59:61].strip())
			qn4.append(x[61:63].strip())
			qn5.append(x[63:65].strip())
			qn6.append(x[65:67].strip())
			qn7.append(x[67:69].strip())
			qn8.append(x[69:71].strip())
			qn9.append(x[71:73].strip())
			qn10.append(x[73:75].strip())
			qn11.append(x[75:77].strip())
			qn12.append(x[77:].strip())
		
	#now go through and fix everything into the appropriate formats and make numpy arrays as needed
	
	#we start with the easy ones that don't have nonsense letters
	frequency = np.array(frequency)
	frequency = frequency.astype(float)
	freq_err = np.array(freq_err)
	freq_err = freq_err.astype(float)	
	logint = np.array(logint)
	logint = logint.astype(float)	
	dof = np.array(dof)
	dof = dof.astype(int)
	elow = np.array(elow)
	elow = elow.astype(float)
	tag = np.array(tag)
	tag = tag.astype(int)
	qnformat = np.array(qnformat)
	qnformat = qnformat.astype(int)	
	
	#convert elow to Kelvin
		
	elow /= kcm
	
	#now we use a sub-function to fix the letters and +/- that show up in gup, and the qns, and returns floats	
	def _fix_spcat(x):
		'''Fixes letters and +/- in something that's read in and returns floats'''
		
		#fix blanks - we just want them to be nice nones rather than empty strings
		
		if x == '':
			return None
		elif x == '+':
			return 1
		elif x == '-':
			return -1	
			
		sub_dict = {'a' : 100,
					'b' : 110,
					'c' : 120,
					'd' : 130,
					'e' : 140,
					'f' : 150,
					'g' : 160,
					'h' : 170,
					'i' : 180,
					'j' : 190,
					'k' : 200,
					'l' : 210,
					'm' : 220,
					'n' : 230,
					'o' : 240,
					'p' : 250,
					'q' : 260,
					'r' : 270,
					's' : 280,
					't' : 290,
					'u' : 300,
					'v' : 310,
					'w' : 320,
					'x' : 330,
					'y' : 340,
					'z' : 350,
					}
					
		alpha = re.sub('[^a-zA-Z]+','',x)
		
		if alpha == '':
			return int(x)
		else:		
			return sub_dict.get(alpha.lower(), 0) + int(x[1])
	
	#run the other arrays through the fixer, then convert them to what they need to be
	
	gup = [_fix_spcat(x) for x in gup]
	gup = np.array(gup)
	gup = gup.astype(int)
	qn1 = [_fix_spcat(x) for x in qn1]
	qn1 = np.array(qn1)
	qn1 = qn1.astype(int) if all(y is not None for y in qn1) is True else qn1
	qn2 = [_fix_spcat(x) for x in qn2]
	qn2 = np.array(qn2)
	qn2 = qn2.astype(int) if all(x is not None for x in qn2) is True else qn2			
	qn3 = [_fix_spcat(x) for x in qn3]
	qn3 = np.array(qn3)
	qn3 = qn3.astype(int) if all(x is not None for x in qn3) is True else qn3	
	qn4 = [_fix_spcat(x) for x in qn4]
	qn4 = np.array(qn4)
	qn4 = qn4.astype(int) if all(x is not None for x in qn4) is True else qn4	
	qn5 = [_fix_spcat(x) for x in qn5]
	qn5 = np.array(qn5)
	qn5 = qn5.astype(int) if all(x is not None for x in qn5) is True else qn5	
	qn6 = [_fix_spcat(x) for x in qn6]
	qn6 = np.array(qn6)
	qn6 = qn6.astype(int) if all(x is not None for x in qn6) is True else qn6	
	qn7 = [_fix_spcat(x) for x in qn7]
	qn7 = np.array(qn7)
	qn7 = qn7.astype(int) if all(x is not None for x in qn7) is True else qn7	
	qn8 = [_fix_spcat(x) for x in qn8]
	qn8 = np.array(qn8)
	qn8 = qn8.astype(int) if all(x is not None for x in qn8) is True else qn8	
	qn9 = [_fix_spcat(x) for x in qn9]
	qn9 = np.array(qn9)
	qn9 = qn9.astype(int) if all(x is not None for x in qn9) is True else qn9	
	qn10 = [_fix_spcat(x) for x in qn10]
	qn10 = np.array(qn10)
	qn10 = qn10.astype(int) if all(x is not None for x in qn10) is True else qn10	
	qn11 = [_fix_spcat(x) for x in qn11]
	qn11 = np.array(qn11)
	qn11 = qn11.astype(int) if all(x is not None for x in qn11) is True else qn11	
	qn12 = [_fix_spcat(x) for x in qn12]
	qn12 = np.array(qn12)
	qn12 = qn12.astype(int) if all(x is not None for x in qn12) is True else qn12	
	
	#make the qnstrings
	qn_list_up = [_make_qnstr(qn1,qn2,qn3,qn4,qn5,qn6,None,None) for qn1,qn2,qn3,qn4,qn5,qn6 in zip(qn1,qn2,qn3,qn4,qn5,qn6)]
	qn_list_low = [_make_qnstr(qn7,qn8,qn9,qn10,qn11,qn12,None,None) for qn7,qn8,qn9,qn10,qn11,qn12 in zip(qn7,qn8,qn9,qn10,qn11,qn12)]
	
	
	split_cat = {
					'frequency'	: 	frequency,
					'freq_err'	:	freq_err,
					'logint'	:	logint,
					'dof'		:	dof,
					'elow'		:	elow,
					'gup'		:	gup,
					'tag'		:	tag,
					'qnformat'	:	qnformat,
					'qn1up'		:	qn1,
					'qn2up'		:	qn2,
					'qn3up'		:	qn3,
					'qn4up'		:	qn4,
					'qn5up'		:	qn5,
					'qn6up'		:	qn6,
					'qn7up'		:	np.full(len(frequency),None),
					'qn8up'		:	np.full(len(frequency),None),
					'qnup_str'	:	np.array(qn_list_up),
					'qn1low'	:	qn7,
					'qn2low'	:	qn8,
					'qn3low'	:	qn9,
					'qn4low'	:	qn10,
					'qn5low'	:	qn11,
					'qn6low'	:	qn12,
					'qn7low'	:	np.full(len(frequency),None),
					'qn8low'	:	np.full(len(frequency),None),
					'qnlow_str'	:	np.array(qn_list_low),
					'notes'		:	'Loaded from file {}' .format(filein)
				}
	
	return split_cat

def _trim_arr(arr,lls,uls,key_arr=None,return_idxs=False,ll_idxs=None,ul_idxs=None,return_mask=False):
	'''
	Trims the input array to the limits specified.  Optionally, will get indices from
	the key_arr for trimming instead.
	'''

	if ll_idxs is not None:
		return np.concatenate([arr[ll_idx:ul_idx] for ll_idx,ul_idx in zip(ll_idxs,ul_idxs)])
	# modified to set as False to begin with, and working with
	# booleans instead of numbers
	mask_arr = np.zeros_like(arr, dtype=bool)
	if key_arr is None:
		for x,y in zip(lls,uls):
			mask_arr[(arr>x) & (arr<y)] = True
	else:
		for x,y in zip(lls,uls):
			mask_arr[(key_arr>x) & (key_arr<y)] = True
	if return_mask:
		return mask_arr
	if return_idxs is False:
		return arr[mask_arr]
	else:
		ll_idxs_out = _find_ones(mask_arr)[0]
		ul_idxs_out = _find_ones(mask_arr)[1]
		return arr[mask_arr],ll_idxs_out,ul_idxs_out