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

def calc_rvor_sphere(lon, lat, uwnd, vwnd):
    """
    Relative vorticity on a sphere:
        zeta = (1/(a cos(phi))) * [ d(v)/d(lambda) - d(u cos(phi))/d(phi) ]
    uwnd, vwnd: shape [lat, lon].
    """
    R = constants.R
    dlon = np.deg2rad(lon[1] - lon[0])
    dlat = np.deg2rad(lat[1] - lat[0])
    coslat = np.cos(np.deg2rad(lat))[:, None]

    dv_dlambda   = _d_dx4(vwnd, dlon, axis=1)          # axis=1 is periodic: roll is correct
    ducos_dphi   = _d_dx4(uwnd * coslat, dlat, axis=0) # axis=0 is NOT periodic

    rv = (dv_dlambda - ducos_dphi) / (R * coslat)

    # np.roll wraps in latitude, which is wrong; invalidate the two edge rings.
    rv[:2, :]  = 0.0
    rv[-2:, :] = 0.0
    return rv

def calc_div_sphere(lon, lat, uwnd, vwnd):
    """
    Divergence on a sphere:
        div = (1/(a cos(phi))) * [ d(u)/d(lambda) + d(v cos(phi))/d(phi) ]
    uwnd, vwnd: shape [lat, lon].
    """
    R = constants.R
    dlon = np.deg2rad(lon[1] - lon[0])
    dlat = np.deg2rad(lat[1] - lat[0])
    coslat = np.cos(np.deg2rad(lat))[:, None]

    du_dlambda  = _d_dx4(uwnd, dlon, axis=1)
    dvcos_dphi  = _d_dx4(vwnd * coslat, dlat, axis=0)

    div = (du_dlambda + dvcos_dphi) / (R * coslat)

    div[:2, :]  = 0.0
    div[-2:, :] = 0.0
    return div

def _d_dx4(f, h, axis):
    """4th-order centered derivative; uses i+-1 and i+-2 so the odd/even
    sublattices are coupled (no checkerboard decoupling)."""
    d = np.zeros_like(f)
    fm2 = np.roll(f, 2, axis=axis)
    fm1 = np.roll(f, 1, axis=axis)
    fp1 = np.roll(f, -1, axis=axis)
    fp2 = np.roll(f, -2, axis=axis)
    d = (fm2 - 8*fm1 + 8*fp1 - fp2) / (12.0 * h)
    return d

def sf_wnd(lon, lat, psi):
    LON, LAT = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    dlon = np.deg2rad(lon[1] - lon[0])
    dlat = np.deg2rad(lat[1] - lat[0])

    dpsi_dlon = _d_dx4(psi, dlon, axis=1)
    dpsi_dlat = _d_dx4(psi, dlat, axis=0)

    v =  dpsi_dlon / (constants.R * np.cos(LAT))
    u = -dpsi_dlat / constants.R

    # edges are garbage from the roll; zero or one-side them
    u[:2, :] = u[-2:, :] = 0.0
    u[:, :2] = u[:, -2:] = 0.0
    v[:2, :] = v[-2:, :] = 0.0
    v[:, :2] = v[:, -2:] = 0.0
    return u, v

# Calculate the zonal and meridional wind from the velocity potential.
def pot_wnd(lon, lat, phi):
    LON, LAT = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    dlon = np.deg2rad(lon[1] - lon[0])
    dlat = np.deg2rad(lat[1] - lat[0])

    dphi_dlon = _d_dx4(phi, dlon, axis=1)
    dphi_dlat = _d_dx4(phi, dlat, axis=0)

    # u = (1/(a cosphi)) dphi/dlambda
    u = dphi_dlon / (constants.R * np.cos(LAT))

    # v = (1/a) dphi/dphi
    v = dphi_dlat / constants.R

    # np.roll wraps, so the two outermost rings are invalid.
    for f in (u, v):
        f[:2, :] = 0.0
        f[-2:, :] = 0.0
        f[:, :2] = 0.0
        f[:, -2:] = 0.0

    return u, v

# # Calculate the zonal and meridional wind from the streamfunction.
# def sf_wnd(lon, lat, psi):
#     u = np.zeros_like(psi)
#     v = np.zeros_like(psi)

#     LON, LAT = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
#     dlon = np.deg2rad(lon[1] - lon[0])
#     dlat = np.deg2rad(lat[1] - lat[0])

#     # v = (1/a) dψ/dλ
#     v[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dlon * constants.R)

#     # u = -(1/(a cosφ)) dψ/dφ
#     u[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (
#         2 * dlat * constants.R * np.cos(LAT[1:-1, 1:-1])
#     )

#     return u, v

# # Calculate the zonal and meridional wind from the velocity potential.
# def pot_wnd(lon, lat, phi):
#     u = np.zeros_like(phi)
#     v = np.zeros_like(phi)

#     LON, LAT = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
#     dlon = np.deg2rad(lon[1] - lon[0])
#     dlat = np.deg2rad(lat[1] - lat[0])

#     # u = (1/(a cosφ)) dφ/dλ
#     u[1:-1, 1:-1] = (phi[1:-1, 2:] - phi[1:-1, :-2]) / (
#         2 * dlon * constants.R * np.cos(LAT[1:-1, 1:-1])
#     )

#     # v = (1/a) dφ/dφ
#     v[1:-1, 1:-1] = (phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2 * dlat * constants.R)

#     return u, v


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
