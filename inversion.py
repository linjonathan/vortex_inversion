# %%
import vortex_lib
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

# Read in sample data
fn = 'era5_sample_wind.nc'
ds = xr.open_dataset(fn, engine = 'netcdf4')

# Spherical harmonic expansion needs to remove 90S and 360E
lat = ds['latitude'].sel(latitude = slice(90.0, -89.99)).data
lon = ds['longitude'].sel(longitude = slice(0, 359.99)).data

# Create vortex library; xres uses "super-resolution" to increase the
# resolution of the field. This is optional and helps for inverting 
# vorticity in low-resolution data.
# For xres = 1, no super-resolution will occur.
# For xres = 2, super-resolution will be used prior to inverting the
# vortex (at the cost of some smoothing of the field) and can lead to
# higher accuracy finite-differencing as a result of finer grid spacings.
xres = 1
v_lib = vortex_lib.VORTEX_LIB(lon, lat, xres = xres)

u_level = ds['u'].sel(level = 850, latitude = slice(90.0, -89.99),
                      longitude = slice(0, 359.99))
v_level = ds['v'].sel(level = 850, latitude = slice(90.0, -89.99),
                      longitude = slice(0, 359.99))

# TC_lon_position and TC_lat_position are guesses of where the
# storm center is for the vortex you want to remove.
TC_lon_position = 137
TC_lat_position = 26

# TC_lon_pos_grid/TC_lat_pos_grid are the grid-centers of the vortex.
# u_filt/v_filt are the wind fields with the vortex removed.
# u_filt_HD/v_filt_HD are the super-resolution wind fields with the vortex removed.
TC_lon_pos_grid, TC_lat_pos_grid, u_filt, v_filt, u_filt_HD, v_filt_HD = \
    v_lib.vortex_surgery(u_level.data, v_level.data, TC_lon_position, TC_lat_position)

u_filt = xr.DataArray(data = u_filt, dims = u_level.dims, 
                      coords = u_level.coords, attrs = u_level.attrs)
v_filt = xr.DataArray(data = v_filt, dims = v_level.dims, 
                      coords = v_level.coords, attrs = v_level.attrs)
u_filt_HD = xr.DataArray(data = u_filt_HD, dims = u_level.dims, 
                         coords = {'longitude': np.linspace(lon[0], 360, len(lon)*xres+1)[:-1],
                                   'latitude': np.linspace(90, -90, len(lat)*xres+1)[:-1]}, 
                         attrs = u_level.attrs)
v_filt_HD = xr.DataArray(data = v_filt_HD, dims = v_level.dims, 
                         coords = {'longitude': np.linspace(lon[0], 360, len(lon)*xres+1)[:-1],
                                   'latitude': np.linspace(90, -90, len(lat)*xres+1)[:-1]},
                         attrs = v_level.attrs)
# %% Plot the zonal winds
u_level.plot.pcolormesh(vmin = -30, vmax = 30, cmap = 'RdBu_r')
plt.xlim([100, 170]); plt.ylim([0, 45])

plt.figure()
u_filt.plot.pcolormesh(vmin = -30, vmax = 30, cmap = 'RdBu_r')
plt.xlim([100, 170]); plt.ylim([0, 45])

if xres > 1:
    plt.figure()
    u_filt_HD.plot.pcolormesh(vmin = -30, vmax = 30, cmap = 'RdBu_r')
    plt.xlim([100, 170]); plt.ylim([0, 45])

# %% Plot the meridional winds
plt.figure()
v_level.plot.pcolormesh(vmin = -30, vmax = 30, cmap = 'RdBu_r')
plt.xlim([100, 170]); plt.ylim([0, 45])

plt.figure()
v_filt.plot.pcolormesh(vmin = -30, vmax = 30, cmap = 'RdBu_r')
plt.xlim([100, 170]); plt.ylim([0, 45])

if xres > 1:
    plt.figure()
    v_filt_HD.plot.pcolormesh(vmin = -30, vmax = 30, cmap = 'RdBu_r')
    plt.xlim([100, 170]); plt.ylim([0, 45])
