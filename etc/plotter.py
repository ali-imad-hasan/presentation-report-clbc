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
zonals = params
for m_int, fst in enumerate(fsts):
    # ZONAL MEAN CALCULATION #
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
            pres_levs = [rmn.convertIp(rmn.CONVIP_DECODE, ip1, rmn.KIND_PRESSURE)[0] for ip1 in ip1s]
            
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
        exit()

    ni, nj, nk = len(lons), len(lats), len(pres_levs)
    # builds zonal mean
    print "GETTING ZONAL MEANS" 
    start_time = time.time()
    for param in params.keys():
        zonals[param] = get_zonal_mean(params[param])

    print "That took {0} seconds.".format(time.time() - start_time)
   
    # PLOTTING OF VERTICAL CROSS SECTION #
    x_axis = np.zeros((nk,nj))
    y_axis = np.zeros((nk,nj))
    for k in xrange(nk):
        x_axis[k] = lats[::-1]
    for j in xrange(nj):
        y_axis[:,j] = pres_levs
    for param in params.keys():
        # set up contours
        c_values = plprms[param]['c_values']
        c_map = plt.get_cmap(plprms[param]['cmap_name'])
        param_title = plprms[param]['named_var']
        norm = mpl.colors.BoundaryNorm(c_values, ncolors=c_map.N, clip=True)
        
        # set up fonts+titles
        axis_font = {'size' : '11'}
        fig = plt.figure(figsize=(8,6))
        fig.suptitle(param_title, fontsize=20)
        fig.canvas.set_window_title(fst[-9:-4])

        # plotting
        ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8], label='ax1', xscale='linear', yscale='log', 
                           xlim=[lats[-1],lats[0]], ylim=[1000,0.1], ylabel='Pressure [hPa]', xlabel='Latitude')
        # the 20 doesnt matter, it can be any gun as it's a zonal mean
        varplt1 = ax1.contourf(x_axis, y_axis, zonals[param][:,20,:],
                               cmap=c_map, norm=norm, levels=c_values, fontsize=11, extend='both')

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cax.set_xlabel(plprms[param]['units'])
        
        # colorbar
        plt.colorbar(varplt1, cax=cax)
        # save/show the file
        print "Saving..."
        plt.savefig('/home/ords/aq/alh002/vcross/{0}_{1}.png'.format(param.upper(), str(m_int)))
        plt.close()
        #plt.show()

