# Calculating potential vorticity from fst file
##### IMPORTS #####
import pygeode as pyg
import numpy as np
import rpnpy.librmn.all as rmn
import time, glob, math, operator, itertools
import matplotlib.pyplot as plt
import rpnpy.vgd.all as vgd
from mpl_toolkits.basemap import Basemap
from levels import *
#const_pressure = nointerp_pressure
##### CONSTANTS #####
a       = 6.371229e6        #earth radius in m
g0      = 9.81
kn2ms   = 0.51444444     #conversion from knots to m/s
kappa   = 0.287514
dlatlon = 1.125
l       = 0             # used to name files
#filename = '/home/ords/aq/alh002/pyscripts/workdir/momentum_test.fst'
filenames = glob.glob('/home/ords/aq/alh002/NCDF/MOMFILES/*.nc'); filenames.sort()
#filenames = ['/home/ords/aq/alh002/NCDF/MOMFILES/testmom.nc']
lnsp_files = ["/home/ords/aq/alh002/NCDF/LNSP/2008.nc", 
              "/home/ords/aq/alh002/NCDF/LNSP/2009.nc", 
              "/home/ords/aq/alh002/NCDF/LNSP/2010.nc", 
              "/home/ords/aq/alh002/NCDF/LNSP/2011.nc", 
              "/home/ords/aq/alh002/NCDF/LNSP/2012.nc"]  # in this list, place the directory of the year
#lnsp_files = [lnsp_files[0]]  # this is temporary, just to work with one file

##### FXN #####
def trim_excess(data, x, y):
    '''
    trims out the redundant portions of the partitioned array depending on its position in the partition
    this assumes a 2x2 partition
    '''
    trimmed_shape       = list(data.shape)
    trimmed_shape[1]    = trimmed_shape[1] - 2
    trimmed_shape[2]    = trimmed_shape[2] - 1

    trimmed = np.zeros(trimmed_shape)

    if x == 1:
        if y == 1:
            trimmed = data[:,1:,2:]
        else:
            trimmed = data[:,:-1,2:]
    else:
        if y == 1:
            trimmed = data[:,1:,:-2]
        else:
            trimmed = data[:,:-1,:-2]
    return trimmed


def build_data(data, coord_list):
    '''
    builds data for the species identified based on binned coordinates

    ni: number of lon grid points
    nj: number of lat grid points
    species: species you wish to build the data for
    '''
    global const_pressure, ni, nj

    print "GETTING GO3 FROM BINS"
    start_time = time.time()
    nk = len(const_pressure)
    tropos = np.zeros((nk, ni, nj))
    strato = np.zeros((nk, ni, nj))

#    for tuple_coord in coord_list[0]:
#        coordinate = [int(x) for x in tuple_coord]  # creates a list with 3 ints [nk,ni,nj]
#        for ind in coordinate:
#            tropos[coordinate[0], coordinate[1], coordinate[2]] = data[coordinate[0], coordinate[1], coordinate[2]]

    for i in xrange(ni):
        for j in xrange(nj):
            for k in xrange(nk):
                if coord_list[k,i,j] == 0:
                    strato[k,i,j] = data[k,i,j]
                elif coord_list[k,i,j] == 1:
                    tropos[k,i,j] = data[k,i,j]

    strato = np.nan_to_num(strato)
    tropos = np.nan_to_num(tropos)

    print "FILLING ZEROS"
    for i in xrange(ni):
        for j in xrange(nj):
            for k in xrange(nk):
                if tropos[k,i,j] == 0:
                    tropos[k:,i,j] = tropos[k-1, i, j]
                    break

    for i in xrange(ni):
        for j in xrange(nj):
            for k in xrange(nk - 1, -1, -1):  # goes through levels backwards
                if strato[k,i,j] == 0:
                    strato[:k+1,i,j] = strato[k+1, i, j]
                    break

#    for tuple_coord in coord_list[1]:
#        coordinate = [int(x) for x in tuple_coord]  # creates a list with 3 ints [nk,ni,nj]
#        for ind in coordinate:
#            strato[coordinate[0], coordinate[1], coordinate[2]] = data[coordinate[0], coordinate[1], coordinate[2]]
    print "That took {0} seconds.".format(str(time.time() - start_time))
    return tropos, strato


def monthly_mean_data(array):
    '''
    gets monthly mean of the troposphere GO3
    levels with no tropospheric gridpoints are nan
    '''
    
    print "GETTING MONTHLY MEAN"
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


def calc_qq(uu, vv, dx, dy, cosphi):
    '''
    calculates qq from u component of wind, v component of wind, dx, dy, and cosphi.

    uu/vv should be 4d arrays, dx/dy/cosphi should be 2d in the form nj, ni.

    returns 2d array in shape nj,ni
    '''

    # dvdx/dudy are calculated through leapfrogging technique
    dvdx = np.zeros((nj,ni))
    dudy = np.zeros((nj,ni))
    qq   = np.zeros((nj,ni))

    # calculation of qq from dvdx and dudy
    dvdx[:,0]  = (vv[t,k,:,1] - vv[t,k,:,0]) / dx[:,0]
    for i in xrange(1, ni-1):
        dvdx[:,i] = ((vv[t,k,:,i+1] - vv[t,k,:,i-1]) / dx[:,i]) / 2.
    dvdx[:,-1] = (vv[t,k,:,-1] - vv[t,k,:,-2]) / dx[:,-1]

    try:
        dudy[0,:]   = ((uu[t,k,1,:] * cosphi[1]) - (uu[t,k,0,:] * cosphi[0])) / dy[0,:]
        for j in xrange(1, nj-1):
            dudy[j,:] = (((uu[t,k,j+1,:] * cosphi[j+1]) - (uu[t,k,j-1,:] * cosphi[j-1])) / dy[j,:])/2.
        dudy[-1,:]  = ((uu[t,k,-2,:] * cosphi[-2]) - (uu[t,k,-1,:] * cosphi[-1])) / dy[-1,:]
    except:
        raise

    for i in xrange(ni):
        qq[:, i] = f0_p + dvdx[:,i] - dudy[:,i]/cosphi 

    return qq


def bin_coords(to_bin_array, timestep):
    '''
    bins coords based on if they are stratospheric or tropospheric relative to 
    if PV value is > 2

    writes the index coords in the form nk,ni,nj (comma seperated) in its respective file
    depending on if it's stratospheric or tropospheric
    '''
    global nk_mom, strato_file, tropo_file, const_pressure
    start_time = time.time()
    # reverses the levels so k will go from bottom to top (1000 - 0.1)
    PV_array = to_bin_array[::-1]
    nk = len(const_pressure)

    ni, nj = to_bin_array.shape[1:]

    # initializes lists
    strato_coords   = []
    tropo_coords    = []

    # if you wish to use a numpy solution (much faster), uncomment these two
    #strato_coords   = np.zeros((nk_mom,ni,nj))
    tropo_coords    = np.zeros((nk,ni,nj))

    # creates file
#    strato  = open(strato_file,'w')
#    tropo   = open(tropo_file, 'w')
#    strato.write("TIMESTEP,{0}\n".format(timestep))
#    tropo.write("TIMESTEP,{0}\n".format(timestep))

    for i in xrange(ni):
        for j in xrange(nj):
            # limits to 610 hPa in const_pressure
            #for k in xrange(nk - const_pressure.index(610), nk - const_pressure.index(100)):
            for k in xrange(nk):
                if PV_array[k,i,j] > 2.0:
                    # if you wish to use a numpy solution (much faster), uncomment these two
                    tropo_coords[:k,i,j] = 1
                    #strato_coords[k:,i,j] = 1
                    #tropo_coords    += [(z, i, j) for z in xrange(k)]
                    #strato_coords   += [(z, i, j) for z in xrange(k, nk)]
                    break
    # fills bottom tropo boundary (pressure > 610)
#    for i in xrange(ni):
#        for j in xrange(nj):
#            tropo_coords += [(z,i,j) for z in xrange(17)]
    tropo_coords[:17, :, :] = 1  # strato floor @ 610 hPa
    tropo_coords[35:, :, :] = 0  # tropo ceiling @ 100 hPa
#            tropo_coords[(nk - const_pressure.index(100)):] = 0
            #tropo_coords    += [(z, i, j) for z in xrange(nk - const_pressure.index(610))]
            #strato_coords   += [(z, i, j) for z in xrange(nk - const_pressure.index(100), nk)]
            
            # fills top (pressure < 100 hPa) strato boundary

    print "Binning time: {0}".format(str(time.time() - start_time))
    return tropo_coords#, strato_coords

    # none of this runs as the return statement stops it 
    # conducts writing to file
    print 'WRITING BINNED INDEX POSITIONS'
    for coord in tropo_coords:
        tropo.write('{0},{1},{2}\n'.format(coord[0], coord[1], coord[2]))
    for coord in strato_coords:
        strato.write('{0},{1},{2}\n'.format(coord[0], coord[1], coord[2]))
    print 'DONE WRITING'
    strato.close()
    tropo.close()

    print "Binning time: {0}".format(str(time.time() - start_time))


def build_fst(params, y_int, m_int):
    '''
    (dict) -> (int, dict)
    builds the file as per the parameters defined in the params dict
    
    returns the file_id
    '''

    # makes an empty .fst file
    day = time.gmtime()
    temp = '' 
    for x in day: temp += str(x)
    #new_nc = '/home/ords/aq/alh002/pyscripts/workdir/pv_files/TEST5.fst'
    new_nc = '/home/ords/aq/alh002/pyscripts/workdir/pv_files/4x_POTVOR_file_{0}_{1}.fst'.format(y_int + 2008, m_int+1)
    tmp = open(new_nc, 'w+'); tmp.close()
    output_file = new_nc

    try:
        file_id = rmn.fnom(output_file)
        open_fst = rmn.fstouv(file_id, rmn.FST_RW)
        print(file_id, open_fst)

        MACC_grid = rmn.encodeGrid(params)
        print("Grids created.")
        print 'Grid Shape:' + str(MACC_grid['shape'])

        rmn.writeGrid(file_id, MACC_grid)
        toc_record = vgd.vgd_new_pres(const_pressure, ip1=MACC_grid['ig1'], ip2=MACC_grid['ig2'])
        vgd.vgd_write(toc_record, file_id)

        return file_id, MACC_grid

    except:
        rmn.fstfrm(file_id)
        rmn.fclos(file_id)
        raise


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


def get_zonal_mean(array):
    '''
    array must be 3d of size nk, nlon, nlat
    '''
    
    zonal_mean = np.zeros(array.shape)
    for k in xrange(len(array)):
        for j in xrange(len(array[0,0])):
            zonal_mean[k,:,j] = array[k,:,j].mean()

    return zonal_mean


def shift_lon(array):
    '''
    shifts the longitude from -180 - 180 to 0 - 360
    '''
    new_array = np.zeros(array.shape)
    lonlen = len(new_array[0])
    new_array[:, :lonlen/2] = array[:, lonlen/2:]
    new_array[:, lonlen/2:] = array[:, :lonlen/2]
    return new_array


def vert_interp(pressure, org):
    '''
    org should be a 3d array, pressure should be a 3d array
    of pressures.
    '''
    global const_pressure    
    start_time = time.time()
    print "Conducting vertical interpolation..."

    lon         = len(org[0])
    lat         = len(org[0, 0])
    lev         = len(org)
    y_interp    = np.zeros((len(const_pressure), lon, lat))
    x_interp    = const_pressure

    for i in xrange(lon):
        for j in xrange(lat):
            try:
                x_initial = pressure[:, j, i]
                y_initial = org[:, i, j]
                y_interp[:, i, j] = np.interp(x_interp, x_initial, y_initial)
            except:
                print pressure.shape
                raise
                
    print "That took " + str(time.time() - start_time) + " seconds"
    return y_interp


def build_hhmmss(timestep):
    '''
    calculates seconds from timestep

    given seconds (0 being 00 hours 00 minutes 00 seconds in the day, 
    86399 being 23 hours 59 minutes 59 seconds in the day), returns an
    int in the form hhmmss00 for use in record definition with rmn.newdate 
    '''
    seconds = int((((timestep) % 4.)/4) * 86400.)

    hh   = int(math.floor(seconds/3600))
    mm   = int(math.floor((seconds - (hh * 3600)) / 60))
    ss   = int(math.floor((seconds - (hh * 3600) - (mm * 60)) / 60))

    time_int = int(hh * 1e6 + mm * 1e4 + ss * 1e2)
    
    print 'TIMEINT: {0}'.format(str(time_int))
    return time_int


def get_pressures(open_nc, m_int):
    '''
    (open .nc, int) -> np.ndarray
    gets pressures from open_file, based on m_int as an integer
    representing the month you wish to obtain.
    '''
    global Ak, Bk, const_pressure
                                 
    start_time = time.time()
    print "Getting pressures..."
    
    lnsp = open_nc.lnsp  # open file containing every lnsp for that year
    lnsp_array = splice_month(lnsp, m_int)
    print lnsp_array.shape

    shape = list(lnsp_array.shape); shape.insert(1, 60)  # creates a dimension for levels
    print "PRESSURE SHAPE IS {0}".format(str(shape))
    pressure = np.zeros(shape)
    full_pressure = np.zeros(shape)
    try:
        for lev in xrange(len(pressure[0])):
            pressure[:, lev] = ((Ak[lev] + Bk[lev]*np.exp(lnsp_array)) / 100.)

    except:
        debug_tuple = (lev,)
        print "LEV: {0}".format(debug_tuple)
        print pressure.shape, lnsp_array.shape
        raise

    for lev in xrange(1, len(pressure[0])):
        try:
            full_pressure[:, lev-1] = (pressure[:,lev] + pressure[:, lev-1]) / 2.
        except IndexError:
            if k+1 != 0:
                raise
            else:
                pass
    full_pressure[:,-1] = 1000.0

#    print pressure[5, :, 12, 33]
#    print pressure[15, :, 12, 23]
#    print pressure[45, :, 12, 53]
#    print pressure[75, :, 2, 39]
#    print full_pressure[23, :, 12, 23]
    print "That took " + str(time.time() - start_time) + " seconds"
    return full_pressure 


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


##### MAIN #####
month_list = ['01JAN','02FEB', '03MAR',  
              '04APR', '05MAY', '06JUN', 
              '07JLY', '08AUG', '09SEP', 
              '10OCT', '11NOV', '12DEC']
# this portion of the code handles parsing values from the nc file
for year_int, filename in enumerate(filenames):
    to3_list = []
    so3_list = []
    go3_list = []
    nc = pyg.open(filename)
    lnsp_file = pyg.open(lnsp_files[year_int])
    start_time = time.time()
    for m_int, month in enumerate(month_list):
        if m_int <= 8:# or m_int > 11:
            continue

        # breaking apart the file to conserve memory usage
        for x in xrange(2):
            for y in xrange(2):
                date_tuple = (year_int, month)
                strato_file = '/home/ords/aq/alh002/pyscripts/workdir/pv_files/strato_coords_{0}_{1}.txt'.format(year_int, month)
                tropo_file = '/home/ords/aq/alh002/pyscripts/workdir/pv_files/tropo_coords_{0}_{1}.txt'.format(year_int, month)
                uu = splice_month(nc.u, m_int)

                lon = nc.longitude.values
                lat = nc.latitude.values[:len(uu[0,0])]
                lat = lat[::-1]
                new_shape = list(uu.shape)
                new_shape[2] = new_shape[2] + 1
                new_shape[3] = new_shape[3] + 2
 
                ni = len(lon)
                nj = len(lat)
                timesteps = len(uu)

                # partitioning (s = start, f = final)
                if x == 0:
                    lonind_s = x * (ni / 2)
                    lonind_f = (ni / 2 + (x * (ni / 2))) + 1
                else:
                    lonind_s = (x * (ni / 2)) - 1
                    lonind_f = ni / 2 + (x * (ni / 2))
                if y == 0:
                    latind_s = y * (nj / 2)
                    latind_f = (nj / 2 + (y * (nj / 2))) + 1
                else:
                    latind_s = (y * (nj / 2)) - 1
                    latind_f = nj / 2 + (y * (nj / 2))

                pressures = get_pressures(lnsp_file, m_int)[:,:,latind_s:latind_f,lonind_s:lonind_f]
                pressures = pressures[:,:,::-1]
                nk_mom = len(pressures[0])#len(const_pressure)
                nk_thm = nk_mom
                qq2 = np.zeros((nk_mom, nj, ni))

                uu = splice_month(nc.u, m_int)[:,:,latind_s:latind_f,lonind_s:lonind_f]
                uu = uu[:,:,::-1]

                # reinits ni, nj
                nj = len(uu[0,0])
                ni = len(uu[0,0,0])

                grid_org    = (ni, nj, int(lat[0]), int(lon[0]), dlatlon, dlatlon)
                grid_new    = (ni * 2, nj * 2, int(lat[0]), int(lon[0]), dlatlon / 2, dlatlon / 2)
                outGrid     = rmn.defGrid_L(*grid_new)
                inGrid      = rmn.defGrid_L(*grid_org)
                print str(outGrid['ni']), str(outGrid['nj'])
                print str(inGrid['ni']), str(inGrid['nj'])

                new_pp = np.zeros(new_shape); new_pp = np.transpose(new_pp, (0,1,3,2))
                print "Horizontally interpolating..."
                for t in xrange(timesteps):
                    for k in xrange(nk_mom):
                        rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE, rmn.EZ_INTERP_LINEAR)
                        new_pp[t,k] = rmn.ezsint(outGrid['id'], inGrid['id'], np.transpose(pressures[t,k]))
                del pressures

                new_uu = np.zeros(new_shape); new_uu = np.transpose(new_uu, (0,1,3,2))
                print "Horizontally interpolating..."
                for t in xrange(timesteps):
                    for k in xrange(nk_mom):
                        rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE, rmn.EZ_INTERP_LINEAR)
                        new_uu[t,k] = rmn.ezsint(outGrid['id'], inGrid['id'], np.transpose(uu[t,k]))
                del uu
                
                vv = splice_month(nc.v, m_int)[:,:,latind_s:latind_f,lonind_s:lonind_f]
                vv = vv[:,:,::-1]
                new_vv = np.zeros(new_shape); new_vv = np.transpose(new_vv, (0,1,3,2))
                print "Horizontally interpolating..."
                for t in xrange(timesteps):
                    for k in xrange(nk_mom):
                        rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE, rmn.EZ_INTERP_LINEAR)
                        new_vv[t,k] = rmn.ezsint(outGrid['id'], inGrid['id'], np.transpose(vv[t,k]))
                del vv

                qq = splice_month(nc.vo, m_int)[:,:,latind_s:latind_f,lonind_s:lonind_f]
                qq = qq[:,:,::-1]
                new_qq = np.zeros(new_shape); new_qq = np.transpose(new_qq, (0,1,3,2))
                print "Horizontally interpolating..."
                for t in xrange(timesteps):
                    for k in xrange(nk_mom):
                        rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE, rmn.EZ_INTERP_LINEAR)
                        new_qq[t,k] = rmn.ezsint(outGrid['id'], inGrid['id'], np.transpose(qq[t,k]))
                del qq

                th = splice_month(nc.t, m_int)[:,:,latind_s:latind_f,lonind_s:lonind_f]
                th = th[:,:,::-1]
                new_th = np.zeros(new_shape); new_th = np.transpose(new_th, (0,1,3,2))
                print "Horizontally interpolating..."
                for t in xrange(timesteps):
                    for k in xrange(nk_mom):
                        rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE, rmn.EZ_INTERP_LINEAR)
                        new_th[t,k] = rmn.ezsint(outGrid['id'], inGrid['id'], np.transpose(th[t,k]))
                del th

                qq = np.transpose(new_qq, (0,1,3,2))
                del new_qq
                vv = np.transpose(new_vv, (0,1,3,2))
                del new_vv
                uu = np.transpose(new_uu, (0,1,3,2))
                del new_uu
                th = np.transpose(new_th, (0,1,3,2))
                del new_th
                pressures = np.transpose(new_pp, (0,1,3,2))
                del new_pp

                # reinits ni, nj
                latmin = lat[0]
                latmax = lat[nj-1]
                lonmin = lon[0]
                lonmax = lon[ni-1]
                nj = len(uu[0,0])
                ni = len(uu[0,0,0])
                lat = np.linspace(latmin, latmax, nj)
                lon = np.linspace(lonmin, lonmax, ni)

                # stuff on level k - 1
                uulast = np.zeros([nj, ni])
                vvlast = np.zeros([nj, ni])

                # stuff on level k + 0.5 (th will be on this level)
                dthdx = np.zeros([nj, ni])
                dthdy = np.zeros([nj, ni])

                # stuff on level k - 0.5
                thlast = np.zeros([nj, ni])

                # calculation must be done with actual lon-lat, not rotated
                cosphi = np.cos(lat * (np.pi / 180.))
                cosphi[-1] = cosphi[-2]
                f0_p   = 2 * np.pi/86400. * np.sin(lat * np.pi/180.)
                for t in xrange(timesteps):
                    for i in xrange(ni):
                        for k in xrange(nk_mom):
                            qq[t,k,:,i] = f0_p + qq[t,k,:,i]
                dx     = np.zeros([nj, ni])  # shape is lon, lat
                dy     = np.zeros([nj, ni])

                # setting dx and dy partial derivative arrays through leapfrogging
                try:
                    print "Setting dx..."
                    if x == 1:
                        for i in xrange(1,ni-1):
                            # in absolute vorticity, this is part of du*cosphi / dx
                            dx[:,i] = a * cosphi * ((lon[i+1]-lon[i-1]) * np.pi/360.)
                            # top level init. 
                        dx[:,-1] = a * cosphi * (lon[-1] - lon[-2]) * np.pi/180.  # top level init
                    else:
                        for i in xrange(1, ni-1):
                            # in absolute vorticity, this is part of du*cosphi / dx
                            dx[:,i] = a * cosphi * ((lon[i+1]-lon[i-1]) * np.pi/360.)
                            # bottom level init. 
                        dx[:,0] = a * cosphi * (lon[1] - lon[0]) * np.pi/180. 
                except:
                    print i 
                    raise
                try:
                    print "Setting dy..."
                    if y == 1:
                        for j in xrange(nj-1):
                            dy[j,:] = a * (lat[j+1] - lat[j-1]) * np.pi/360.
                        dy[-1,:] = a * (lat[-1] - lat[-2]) * np.pi/180.  # top level init

                    else:
                        for j in xrange(1, nj-1):
                            dy[j,:] = a * (lat[j+1] - lat[j-1]) * np.pi/360.
                        dy[0,:] = a * (lat[1] - lat[0]) * np.pi/180.  # bottom level init. 
                except:
                    print j
                    raise

                # params for partial grid descriptors and grid definition for fst files
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

                tempq = np.zeros((nk_mom, nj, ni))
                tempv = np.zeros((nj, ni))
                tempu = np.zeros((nj, ni))

                # creates surf/pressure array and turns temp into theta
                for o in xrange(len(uu)):
                    for k in xrange(len(uu[0])):
                        thet1 = pressures[o, -1] / pressures[o, k]
                        th[o,k] = th[o,k] * thet1 ** kappa

                # holds all the bin lists for each timestep
                strato_timed_bin_list   = np.zeros((nk_mom, ni, nj))
                tropo_timed_bin_list    = np.zeros((nk_mom, ni, nj))
                start_time = time.time()
                for t in xrange(len(uu)):
                #for t in xrange(1):
                    # Potential vorticity field
                    PV          = np.zeros([nk_mom, nj, ni])
                    pres_levs   = np.zeros(PV.shape)
                    dp          = np.zeros(PV.shape)

                    pres_levs   = pressures[t]
                    surf_pres   = pres_levs[-1]
                    tempq       = qq[t]

                    try:
                        dp[-1] = 100 * (pres_levs[-1] - pres_levs[-2])
                        dp[0] = 100 * pres_levs[0]
                        for k in xrange(1, nk_mom-1):
                            dp[k] = 100 * (pres_levs[k] - pres_levs[k-1])
                    except:
                        print k
                        raise

                    # calculation of term1/term2 (level by level)
                    # omits the final level (i = nk_mom)
                    for k in xrange(nk_mom):
                        print "Record Level: " + str(k+1)
                        if k == 0:
                            term1   = np.zeros((nk_mom, nj, ni))
                            term2   = np.zeros((nk_mom, nj, ni))
                            dt      = np.zeros((nk_mom, nj, ni))

                        elif k != nk_mom - 1:
                            # define the dthdy and dthdx based of thermo level
                            dthdy[0,:] = (th[t,k,1,:] - th[t,k,0,:]) / dy[0,:]  # first dthdy level
                            for j in xrange(1, nj-1):
                                dthdy[j,:] = ((th[t,k,j+1,:] - th[t,k,j-1,:]) / dy[j,:]) / 2.
                            dthdy[-1,:] = (th[t,k,-1,:] - th[t,k,-2,:]) / dy[-1,:]  # last dthdy level
                            dthdx[:,0] = (th[t,k,:,1] - th[t,k,:,0]) / dx[:,0]  # first dthdy level
                            for i in xrange(1, ni-1):
                                dthdx[:,i] = ((th[t,k,:,i+1] - th[t,k,:,i-1])/ dx[:,i]) / 2.
                            dthdx[:,-1] = (th[t,k,:,-1] - th[t,k,:,-2]) / dx[:,-1] # last dthdy level
                            
                        else:
                            print "FINAL LEVEL REACHED"
                            # final level (k = nk)
                            dthdy[0,:]  = (th[t,k,1,:] - th[t,k,0,:]) / dy[0,:]
                            dthdy[-1,:] = (th[t,k,-1,:] - th[t,k,-2,:]) / dy[-1,:]
                            for j in xrange(1, nj-1):
                                dthdy[j,:] = ((th[t,k,j+1,:] - th[t,k,j-1,:]) / dy[j,:])/2

                            dthdx[:,0]  = (th[t,k,:,1] - th[t,k,:,0]) / dx[:,0]
                            dthdx[:,-1] = (th[t,k,:,-1] - th[t,k,:,-2]) / dx[:,-1]
                            for i in xrange(1, ni-1):
                                dthdx[:,i] = ((th[t,k,:,i+1] - th[t,k,:,i-1]) / dx[:,i])/2

                        '''
                        from this point onwards, horizontal grid leapfrogging is complete.
                        the next steps performed are going to be vertical integration of the terms
                        onto pressure levels. these pressure levels will be integrated at "half" levels,
                        not full levels. in the horizontal grid, leapfrogging was done with 
                        level k-1 and level k+1 to give level k. this will be done on level k and k+1 to
                        give level k + 0.5. this level doesnt really exist, it's just the inbetween of the
                        two levels. 
                        '''
                        
                        # PV values are on fields k-1/2 respective to the values obtained
                        if k != 0:
                            term1[k]    = g0 * ((vv[t,k] - vv[t,k-1])/dp[k]) * dthdx
                            term2[k]    = g0 * ((uu[t,k] - uu[t,k-1])/dp[k]) * dthdy
                            dt[k]       = th[t,k] - th[t,k-1]
                    
                    # these exist on half levels -0.5 of the original level
                    PV      = ((-g0 * (dt/dp) * tempq) + term1 - term2) * 1e6
                     
                    PV  = trim_excess(PV, x, y)
                    print PV.shape
                    PV  = np.transpose(PV, (0,2,1))
                    PV  = shift_lon(PV)
                    PV  = vert_interp(pressures[t], PV)
                    #strato_timed_bin_list, tropo_timed_bin_list = bin_coords(PV, t)
                    tropo_timed_bin_list = bin_coords(PV, t)
                    #strato_tropo = [tropo_timed_bin_list, strato_timed_bin_list]

                    # saves np array
                    np.save('/space/hall1/sitestore/eccc/aq/r1/alh002/FST_STORAGE/tropo_strato_files/4x_bins/{3}{4}/{2}-tropo_{0}-{1}'.format(year_int, m_int, t, x, y), tropo_timed_bin_list)

