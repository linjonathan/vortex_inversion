#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Jonathan Lin
Utility library that implements common functions on the sphere.
"""

import numpy as np
import constants

"""
Implements the Haversine algorithm to calculate the great circle distance
between two points. Returns distance in kilometers.
"""
def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)

    # Use the Haversine formula.
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.square(np.sin(dlat/2)) + np.cos(lat1) *
         np.cos(lat2) * np.square(np.sin(dlon/2)))
    c = 2 * np.arcsin(np.sqrt(a))

    km = (constants.R / 1000.) * c
    return(km)

"""
Returns the cartesian angles from a center (lonc, latc)
"""
def sphere_theta(lonc, latc, lon_grid, lat_grid):
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    lon_dist = haversine(LON, latc, lonc, latc)
    lat_dist = haversine(lonc, LAT, lonc, latc)

    with np.errstate(invalid='ignore'):
        lon_dist[LON < lonc] *= -1
        lat_dist[LAT < latc] *= -1
    theta = np.arctan2(lat_dist, lon_dist)

    return(theta)

# Calculate relative vorticity using second order, central differencing.
# TODO: Use forwards and backwards differencing at the poles.
# uwnd and vwnd are grids with dimensions [lat, lon].
def calc_rvor_sphere(lon, lat, uwnd, vwnd):
    # Pad the left and right edges for the finite differences..
    lon_wrap = np.hstack((lon[-1], lon, lon[0]))
    u_wrap = np.hstack((np.expand_dims(uwnd[:, -1], 1), uwnd, np.expand_dims(uwnd[:, 0], 1)))
    v_wrap = np.hstack((np.expand_dims(vwnd[:, -1], 1), vwnd, np.expand_dims(vwnd[:, 0], 1)))
    LON, LAT = np.meshgrid(np.deg2rad(lon_wrap), np.deg2rad(lat))
    dlon = np.deg2rad(lon[1] - lon[0])
    dlat = np.deg2rad(lat[1] - lat[0])

    t1 = (u_wrap[2:, 1:-1] - u_wrap[0:-2, 1:-1]) / (2 * dlon * constants.R)
    t2 = np.multiply(u_wrap[1:-1, 1:-1], np.tan(LAT[1:-1, 1:-1])) / constants.R
    t3 = -np.divide(v_wrap[1:-1, 2:] - v_wrap[1:-1, 0:-2],
                    2 * dlat * constants.R * np.cos(LAT[1:-1, 1:-1]))

    rv = np.zeros([len(lat), len(lon)])
    rv[1:-1, :] = t1 + t2 + t3
    return(rv)

# Calculate divergence using second order, central differencing.
# TODO: Use forwards and backwards differencing at the poles.
# uwnd and vwnd are grids with dimensions [lat, lon].
def calc_div_sphere(lon, lat, uwnd, vwnd):
    # Pad the left and right.
    lon_wrap = np.hstack((lon[-1], lon, lat[0]))
    u_wrap = np.hstack((np.expand_dims(uwnd[:, -1], 1), uwnd, np.expand_dims(uwnd[:, 0], 1)))
    v_wrap = np.hstack((np.expand_dims(vwnd[:, -1], 1), vwnd, np.expand_dims(vwnd[:, 0], 1)))
    LON, LAT = np.meshgrid(np.deg2rad(lon_wrap), np.deg2rad(lat))
    dlon = np.deg2rad(lon[1] - lon[0])
    dlat = np.deg2rad(lat[1] - lat[0])

    t1 = (v_wrap[2:, 1:-1] - v_wrap[0:-2, 1:-1]) / (2 * dlat * constants.R)
    t2 = -np.multiply(v_wrap[1:-1, 1:-1], np.tan(LAT[1:-1, 1:-1])) / constants.R
    t3 = np.divide(u_wrap[1:-1, 2:] - u_wrap[1:-1, 0:-2],
                   2 * dlon * constants.R * np.cos(LAT[1:-1, 1:-1]))

    div = np.zeros([len(lat), len(lon)])
    div[1:-1, :] = t1 + t2 + t3
    return (div)

# Calculate the zonal and meridional wind from the streamfunction.
def sf_wnd(lon, lat, psi):
    u = np.zeros(psi.shape)
    v = np.zeros(psi.shape)
    LON, LAT = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    dlon = np.deg2rad(lon[1] - lon[0])
    dlat = np.deg2rad(lat[1] - lat[0])
    u[1:-1, 1:-1] = np.divide(psi[2:, 1:-1] - psi[0:-2, 1:-1], 2 * dlat * constants.R)
    v[1:-1, 1:-1] = -np.divide(psi[1:-1, 2:] - psi[1:-1, 0:-2],
                               2 * dlon * constants.R * np.cos(LAT[1:-1, 1:-1]))
    return(u, v)

# Calculate the zonal and meridional wind from the velocity potential.
def pot_wnd(lon, lat, phi):
    u = np.zeros(phi.shape)
    v = np.zeros(phi.shape)
    LON, LAT = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    dlon = np.deg2rad(lon[1] - lon[0])
    dlat = np.deg2rad(lat[1] - lat[0])
    u[1:-1, 1:-1] = -np.divide(phi[1:-1, 2:] - phi[1:-1, 0:-2],
                              2 * dlon * constants.R * np.cos(LAT[1:-1, 1:-1]))
    v[1:-1, 1:-1] = -np.divide(phi[2:, 1:-1] - phi[0:-2, 1:-1], 2 * dlat * constants.R)
    return(u, v)

""" Transforms x-y distances to lon-lat distances (Cartesian approximation). """
def to_sphere_dist(clon, clat, dx, dy):
    p_lat = clat + (dy / constants.R) * (180. / np.pi)
    p_lon = clon + (dx / constants.R) * (180. / np.pi) / np.cos(clat * np.pi / 180.)
    return(p_lon, p_lat)

"""
Calculates the translational speed given a vector of lon/lat, and dt_s which is
the discrete time displacement in the lon/lat vectors. Returns in m/s.
dt_s can be either a constant or a vector.
"""
def calc_translational_speed(lon, lat, dt_s):
    if len(lon) <= 1:
        return(np.full(1, np.nan), np.full(1, np.nan))
    elif len(lon.shape) == 1:
        lon = np.expand_dims(lon, 0)
        lat = np.expand_dims(lat, 0)

    fa = lambda x, idx: np.expand_dims(x[:, idx], 1)
    e_lon = np.hstack((2*fa(lon,0) - fa(lon, 1), lon[:],
                       2*fa(lon,-1) - fa(lon,-2)))
    e_lat = np.hstack((2*fa(lat,0) - fa(lat, 1), lat[:],
                       2*fa(lat,-1) - fa(lat,-2)))

    dlon = 0.5 * (np.sign(e_lon[:, 2:] - e_lon[:, 0:-2]) *
                   haversine(e_lon[:,2:], e_lat[:,1:-1],
                             e_lon[:,0:-2], e_lat[:,1:-1]))
    dlat = 0.5 * (np.sign(e_lat[:,2:] - e_lat[:,0:-2]) *
                   haversine(e_lon[:,1:-1], e_lat[:,2:],
                             e_lon[:,1:-1], e_lat[:,0:-2]))

    ut = dlon * 1000. / dt_s
    vt = dlat * 1000. / dt_s
    if len(lon.shape) == 1:
        ut = ut.flatten(); vt = vt.flatten();

    return(ut, vt)
