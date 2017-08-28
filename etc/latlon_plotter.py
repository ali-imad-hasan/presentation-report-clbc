# Plotting of fst climatologies previously made by monthly_mean
##### IMPORTS #####
import numpy as np
import rpnpy.librmn.all as rmn
import rpnpy.vgd.all as vgd
import time, glob, math
import matplotlib as mpl
# required for saving
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from levels import *

##### CONSTANTS #####
fsts = glob.glob("/space/hall1/sitestore/eccc/aq/r1/alh002/FST_STORAGE/output_*.fst"); fsts.sort()
ppb_conv = 2.897e10
resolution = 1.125
lev, ni, nj = 59, 320, 161
params = {'go3':None,
          'co':None,
          'nox':None,
          'ch4':None,
          'dst1':None,
          'dst2':None,
          'so2' :None,
          'dst3':None,
          'samr':None,
          'hcho':None}
#params = {'aermr04':None, 
#          'aermr05':None, 
#          'aermr06':None, 
#          'aermr11':None, 
#          'go3':None, 
#          'hcho':None, 
#          'ch4':None, 
#          'co':None, 
#          'nox':None, 
#          'q':None}


##### FXNS #####
def shift_lon(array):
    '''
    shifts the longitude from -180 - 180 to 0 - 360
    '''
    new_array = np.zeros(array.shape)
    lonlen = len(new_array[0])
    new_array[:, :lonlen/2] = array[:, lonlen/2:]
    new_array[:, lonlen/2:] = array[:, :lonlen/2]
    return new_array


def get_zonal_mean(array):
    '''
    array must be 3d of size nk, nlon, nlat
    '''
    zonal_mean = np.zeros(array.shape)
    for k in xrange(len(array)):
        for j in xrange(len(array[0,0])):
            zonal_mean[k,:,j] = array[k,:,j].mean()

    return zonal_mean


def conv_param(param, data):
    '''
    converts the parameter into data that's to the plot standards
    '''
    global params

    units = plprms[param]['units']
    if units == 'ppbv':
        params[param] = grid_data * (ppb_conv / plprms[param]['mw'])
    elif units == 'ug/kg':  # original is kg/kg
        params[param] = grid_data * 1e9


##### MAIN #####
for m_int, fst in enumerate(fsts):
    #if m_int >= 1: break
    if m_int % 3 == 0: continue
    print "NOW PROCESSING {0}".format(fst)

    # extraction of grid data from fst files
    try:
        file_id = rmn.fstopenall(fst)
        for param in params.keys():
            if param in special_params.keys():
                var_name = special_params[param]
            else:
                var_name = param.upper()

            grid_data = np.zeros((lev, ni, nj))
            v_grid = vgd.vgd_read(file_id)  # gets the vgd object
            ip1s = vgd.vgd_get(v_grid, 'VIPM')
            
            # builds pressure levels
            pres_levs = [float("%6f" % rmn.convertIp(rmn.CONVIP_DECODE, ip1,rmn.KIND_PRESSURE)[0]) for ip1 in ip1s]

            for k, ip in enumerate(ip1s):
                rec_key = rmn.fstinf(file_id, ip1=ip, nomvar=var_name)  # returns key based on params
                grid_data[k] = rmn.fstluk(rec_key)['d']  # reads grid data

                conv_param(param, grid_data)

        latkey = rmn.fstinf(file_id, nomvar='^^')
        lonkey = rmn.fstinf(file_id, nomvar='>>')
        lats = rmn.fstluk(latkey)['d'][0]
        lats = lats[::-1]
        lons = rmn.fstluk(lonkey)['d']
        rmn.fstcloseall(file_id)

    except:
        print "Error opening/operating on file"
        rmn.fstcloseall(file_id)
        exit()

    ni, nj, nk = len(lons), len(lats), len(pres_levs)
   
    # PLOTTING OF VERTICAL CROSS SECTION #
    x_axis = np.zeros((ni,nj))
    y_axis = np.zeros((ni,nj))
    for i in xrange(ni):
        y_axis[i] = lats[::-1]
    for j in xrange(nj):
        x_axis[:,j] = lons
    for param in params.keys():
        for k in xrange(nk):
            if k % 5 != 0 and k != nk - 1: continue
            
            # set up parallels and meridans (spaced every 10 deg)
            parallels = np.arange(-90,90,20)
            meridians = np.arange(-180,180,20)

            # grid data init
            data = np.transpose(params[param][k])
            
            # set up contours
            c_values = plprms[param]['c_values']
            c_map = plt.get_cmap(plprms[param]['cmap_name'])
            param_title = plprms[param]['named_var']
            norm = mpl.colors.BoundaryNorm(c_values, ncolors=c_map.N, clip=True)
            
            # set up fonts+titles
            axis_font = {'size' : '11'}
            fig = plt.figure(figsize=(16,9))
            fig.suptitle(param_title + ' [' +  str(pres_levs[k])+ ' hPa]', fontsize=20)
            fig.canvas.set_window_title(fst[-9:-4])

            # plotting
            ax1 = fig.add_axes([0.05, 0.05, 0.90, 0.90], label='ax1')
            varplt1 = ax1.imshow(data, interpolation='nearest', cmap=c_map)

            # building colorbar axes with mpl axes toolkit
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="2%", pad=0.1)
            cax.set_xlabel(plprms[param]['units'], labelpad=20)

            m = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,
                        llcrnrlon=-180,urcrnrlon=180,resolution='c',ax=ax1)
            m.drawcoastlines()
            #m.fillcontinents(color='coral',lake_color='aqua')

            # basemap building proper grid and showing image
            #topodat = m.transform_scalar(data, lons, lats[::-1], ni, nj)
            im = m.imshow(shift_lon(data), c_map)

            # plotting meridans and parallels
            m.drawparallels(parallels, labels=[1,0,0,1])
            m.drawmeridians(meridians, labels=[1,0,0,1])

            # colorbar
            plt.colorbar(varplt1, cax=cax)

            # save/show the file
            print "Saving..."
            plt.savefig('/home/ords/aq/alh002/latlon/{1}/{0}_{2}.png'.format(param.upper(), str(m_int), pres_levs[k]))
            #plt.show()
            plt.close()

