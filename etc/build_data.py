# Retrieval of species data from indices in binned coordinates
##### IMPORTS #####
import pygeode as pyg
import numpy as np
import rpnpy.librmn.all as rmn
import time, glob, math, operator, itertools
import matplotlib.pyplot as plt
import rpnpy.vgd.all as vgd
from mpl_toolkits.basemap import Basemap
from levels import *

##### CONSTANTS #####
pv_file_dir = '/home/ords/aq/alh002/pyscripts/workdir/pv_files'
dlatlon = 1.125

nk = len(const_pressure)
ni = 320
nj = 73

params0 = {
                'grtyp' : 'Z',
                'grref' : 'L',
                'nj'    : nj,
                'ni'    : ni,
                'lat0'  : 9,#lat[0],
                'lon0'  : 0,#lon[0],
                'dlat'  : dlatlon,
                'dlon'  : dlatlon
                }
tropo_ceiling = len(const_pressure) - const_pressure.index(100) - 1
strato_floor  = len(const_pressure) - const_pressure.index(610) - 1

##### FXNS #####
def build_data(species_data):
    '''
    builds data for the species identified based on binned coordinates

    ni: number of lon grid points
    nj: number of lat grid points
    species: species you wish to build the data for
    '''
    global pv_file_dir
    start_time = time.time()
    timesteps,nk,ni,nj = len(species_data), len(species_data[0]), len(species_data[0,0,0]), 73
    tropos = np.zeros((timesteps, nk, ni, nj))
    i = 0
    
    with open(pv_file_dir + '/tropo_coords.txt', 'r') as tropo_file:
        for line in tropo_file:
            # if the line read is a timestep barrier, retrieve the timestep for the following lines
            if line[:-1].split(',')[0] == 'TIMESTEP':
                timestep = int(line[:-1].split(',')[1])
                print "WORKING ON TIMESTEP: {0}".format(timestep)
                continue
            # otherwise, this will run
            coordinate = [int(x) for x in line[:-1].split(',')]  # creates a list with 3 ints [nk,ni,nj]
            for ind in coordinate:
                tropos[timestep, coordinate[0],coordinate[1],coordinate[2]] = species_data[timestep, coordinate[0],coordinate[2],coordinate[1]]
    print "That took {0} seconds.".format(str(time.time() - start_time))
    return tropos


def build_fst(params, array_tropo, array_strato, m_int):
    '''
    builds and fills an fst file given the proper params and the month integer
    '''
    global ni, nj, nk

    new_fst = "/space/hall1/sitestore/eccc/aq/r1/alh002/FST_STORAGE/tropo_strato_files/BINNED_{0}.fst".format(m_int)
    tmp = open(new_fst, "w"); tmp.close()

    array_tropo = array_tropo[::-1]
    array_strato = array_strato[::-1]

    MACC_grid = rmn.encodeGrid(params)
    file_id = rmn.fnom(new_fst)
    try:
        open_fst = rmn.fstouv(file_id)
        
        toc_record = vgd.vgd_new_pres(const_pressure, ip1=MACC_grid['ig1'], ip2=MACC_grid['ig2'])
        vgd.vgd_write(toc_record, file_id)
        rmn.writeGrid(file_id, MACC_grid)
        print "Grids created. Shape: " + str(MACC_grid['shape'])


        def get_grid_descriptors(file_id):
            '''
            gets grid descriptors from a fst file
            '''

            tic_record = rmn.FST_RDE_META_DEFAULT.copy()
            tac_record = rmn.FST_RDE_META_DEFAULT.copy()
            tac = rmn.fstinl(file_id, nomvar='>>')[0]
            tic = rmn.fstinl(file_id, nomvar='^^')[0]
            tic_record.update(rmn.fstprm(tic))
            tac_record.update(rmn.fstprm(tac))

            return tic_record, tac_record


        tic_record, tac_record = get_grid_descriptors(file_id)

        try:
            # copies the default record
            tropo_record = rmn.FST_RDE_META_DEFAULT.copy()
            strato_record = rmn.FST_RDE_META_DEFAULT.copy()
            for rp1 in xrange(len(const_pressure)):  # writes a record for every level (as a different ip1)
                # converts rp1 into a ip1 with pressure kind
                ip1 = rmn.convertIp(rmn.CONVIP_ENCODE, const_pressure[rp1], rmn.KIND_PRESSURE)
                tropo_record.update(MACC_grid)
                strato_record.update(MACC_grid)
                tropo_record.update({  # Update with specific meta
                    'nomvar': 'TGO3',
                    'typvar': 'C', 
                    'etiket': 'MACCRean',
                    'ni'    : MACC_grid['ni'],
                    'nj'    : MACC_grid['nj'],
                    'ig1'   : tic_record['ip1'],
                    'ig2'   : tic_record['ip2'],
                    'ig3'   : tic_record['ip3'],
                    'ig4'   : tic_record['ig4'],
                    'deet'  : int(86400/4),  # timestep in secs
                    'ip1'   : ip1
                    })
                
                strato_record.update({  # Update with specific meta
                    'nomvar': 'SGO3',
                    'typvar': 'C', 
                    'etiket': 'MACCRean',
                    'ni'    : MACC_grid['ni'],
                    'nj'    : MACC_grid['nj'],
                    'ig1'   : tic_record['ip1'],
                    'ig2'   : tic_record['ip2'],
                    'ig3'   : tic_record['ip3'],
                    'ig4'   : tic_record['ig4'],
                    'deet'  : int(86400/4),  # timestep in secs
                    'ip1'   : ip1
                    })

                tmp = array_tropo[rp1]
                tmp = np.asfortranarray(tmp)
                # data array is structured as tmp = monthly_mean[level] where monthly_mean is [level, lat, lon]
                # Updates with data array in the form (lon x lat)
                tropo_record.update({'d': tmp.astype(np.float32)}) 
                print "Defined a tropo record with dimensions ({0}, {1})".format(tropo_record['ni'], tropo_record['nj'])
                rmn.fstecr(file_id, tropo_record)  # write the dictionary record to the file as a new record

                tmp = array_strato[rp1]
                tmp = np.asfortranarray(tmp)
                # data array is structured as tmp = monthly_mean[level] where monthly_mean is [level, lat, lon]
                # Updates with data array in the form (lon x lat)
                strato_record.update({'d': tmp.astype(np.float32)}) 
                print "Defined a tropo record with dimensions ({0}, {1})".format(strato_record['ni'], strato_record['nj'])
                rmn.fstecr(file_id, strato_record)  # write the dictionary record to the file as a new record

            rmn.fstfrm(file_id)
            rmn.fclos(file_id)
        
        except:
            rmn.fstfrm(file_id)
            rmn.fclos(file_id)
            raise
    except:
        rmn.fstfrm(file_id)
        rmn.fclos(file_id)
        raise
    

def splice_month(open_var, m_int):
    '''
    takes an open var, splices out the timesteps that dont regard
    to that month (with m_int)
    '''
    print "Splicing..."
    if len(open_var.time) == 1464:
        leap_year = True
    elif len(open_var.time) == 1460:
        leap_year = False
    else:
        print("File must contain all 4 time values for each day of the year")
        exit()
    
    if leap_year:
        if m_int == 0:
            return open_var[:124]
        elif m_int == 1:
            return open_var[124:240]
        elif m_int == 2:
            return open_var[240:364]
        elif m_int == 3:
            return open_var[364:484]
        elif m_int == 4:
            return open_var[484:608]
        elif m_int == 5:
            return open_var[608:728]
        elif m_int == 6:
            return open_var[728:852]
        elif m_int == 7:
            return open_var[852:976]
        elif m_int == 8:
            return open_var[976:1096]
        elif m_int == 9:
            return open_var[1096:1220]
        elif m_int == 10:
            return open_var[1220:1340]
        elif m_int == 11:
            return open_var[1340:]

    else:
        if m_int == 0:
            return open_var[:124]
        elif m_int == 1:
            return open_var[124:236]
        elif m_int == 2:
            return open_var[236:360]
        elif m_int == 3:
            return open_var[360:480]
        elif m_int == 4:
            return open_var[480:604]
        elif m_int == 5:
            return open_var[604:724]
        elif m_int == 6:
            return open_var[724:848]
        elif m_int == 7:
            return open_var[848:972]
        elif m_int == 8:
            return open_var[972:1092]
        elif m_int == 9:
            return open_var[1092:1216]
        elif m_int == 10:
            return open_var[1216:1336]
        elif m_int == 11:
            return open_var[1336:]


def monthly_mean_data(array):
    '''
    gets monthly mean of the troposphere GO3
    levels with no tropospheric gridpoints are nan
    '''
    
    timesteps, nk, ni, nj = array.shape

    # these will hold number of indices of timesteps that are nonzero
    ind_t = np.zeros((nk,ni,nj))
    ind_s = np.zeros((nk,ni,nj))
    mm_tropo = np.zeros((nk,ni,nj))

    for i in xrange(ni):
        for j in xrange(nj):
            for k in xrange(nk):
                # nonzero returns a 2-tuple of (list of arrays,datyp) of all indices of values not 0
                ind_t[k,i,j]    = len(array[:,k,i,j].nonzero()[0])  # number of nonzero values
                mm_tropo[k,i,j] = array[:,k,i,j].sum() / ind_t[k,i,j]  # gets the mean of nonzero values

    return mm_tropo


def fill_strato_tropo(strato_go3, tropos_go3):
    '''
    the array has nans when monthly meaned, this fills the nans and replaces 0's with nearest level value
    '''

    strato_go3 = np.nan_to_num(strato_go3)
    tropos_go3 = np.nan_to_num(tropos_go3)

    start_time = time.time()
    print "FILLING ZEROS"
    for i in xrange(ni):
        for j in xrange(nj):
            for k in xrange(nk):
                if tropos_go3[k,i,j] == 0:
                    tropos_go3[k:,i,j] = tropos_go3[k-1, i, j]
                    break

    for i in xrange(ni):
        for j in xrange(nj):
            for k in xrange(nk - 1, -1, -1):  # goes through levels backwards
                if strato_go3[k,i,j] == 0:
                    strato_go3[:k+1,i,j] = strato_go3[k+1, i, j]
                    break

    print "That took {0} seconds.".format(str(time.time() - start_time))
    return tropos_go3, strato_go3


def get_from_binned(m_int, y_int, ts, species_data):
    '''
    gets the binned coordinates given the year int and month int, 
    then converts them to go3 datapoints based on binned indices
    '''
    global strato_floor, tropo_ceiling
    npy = '/space/hall1/sitestore/eccc/aq/r1/alh002/FST_STORAGE/tropo_strato_files/BINNED_1x/{0}-tropo_{1}-{2}.npy'.format(ts, y_int, m_int)
    binned = np.load(npy)
    binned = binned[:59]
    
    species_data = species_data[:,::-1]
    print "The binned shape is {0}".format(binned.shape)
    tropos_go3 = np.zeros(binned.shape)
    strato_go3 = np.zeros(binned.shape)

    for i in xrange(ni):
        for j in xrange(nj):
            for k in xrange(nk):
                if binned[k,i,j] == 1:
                    tropos_go3[k,i,j] = species_data[ts,k,j,i]
                else:
                    strato_go3[k,i,j] = species_data[ts,k,j,i]

    return tropos_go3, strato_go3


##### MAIN #####
years = ['2008', '2009', '2010', '2011', '2012']
month_list = ['01JAN','02FEB', '03MAR',  
              '04APR', '05MAY', '06JUN', 
              '07JLY', '08AUG', '09SEP', 
              '10OCT', '11NOV', '12DEC']

for m_int in xrange(7,8):
    if m_int >= 4:
        continue
    all_year_tropos_go3 = np.zeros((len(years), nk, ni, nj))
    all_year_strato_go3 = np.zeros((len(years), nk, ni, nj))

    for y_int, year in enumerate(years):
        # reads nc file to retrieve data.
        # you may need to call splice_month in pv_calc.py if you're not dealing with january.
        species_data = splice_month(pyg.open('/space/hall1/sitestore/eccc/aq/r1/alh002/NCDF/SPECIES/GO3/{0}.nc'.format(year)).go3, m_int)

        species_data = species_data[:,:,:nj]
        species_data = species_data[:,:,::-1]
    
        tropos_go3 = np.zeros((len(species_data), nk, ni, nj))
        strato_go3 = np.zeros(tropos_go3.shape)
        print "Set up tropo/strato arrays with size {0}".format(strato_go3.shape)

        # fills the tropo_go3 ts
        print "Filling up go3"
        for t in xrange(len(species_data)):
            tropos_go3[t], strato_go3[t] = get_from_binned(m_int, y_int, t, species_data)

        print "Acquiring monthly mean for year {0}".format(year)
        tropos_go3 = monthly_mean_data(tropos_go3)
        strato_go3 = monthly_mean_data(strato_go3)
    
        all_year_tropos_go3[y_int] = tropos_go3
        all_year_strato_go3[y_int] = strato_go3
    
    print "Acquiring monthly means over {0} years.".format(len(years))
    all_year_tropos_go3 = np.mean(all_year_tropos_go3, axis=0)
    all_year_strato_go3 = np.mean(all_year_strato_go3, axis=0)
    all_year_tropos_go3, all_year_strato_go3 = fill_strato_tropo(all_year_strato_go3, all_year_tropos_go3)

    print "Building FST file"
    build_fst(params0, all_year_tropos_go3, all_year_strato_go3, m_int)

