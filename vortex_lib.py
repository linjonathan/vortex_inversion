#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Jonathan Lin
Vortex surgery library.
"""

import numpy as np
import scipy.sparse as sp
import warnings
from scipy.sparse.linalg import spsolve
from pyshtools import SHCoeffs
from pyshtools.expand import SHExpandDH
from pyshtools.expand import MakeGridDH

import constants
import sphere

"""
Returns the longitude and latitude vector given lonn/latn spacings in
the below range.
Longitude range is from [0, 360), and latitude range is from [90, -90).
"""
def lon_lat(lonn, latn):
    lon = np.linspace(0, 360, lonn+1)[0:-1]
    lat = np.linspace(90, -90, latn+1)[0:-1]
    return(lon, lat)

class VORTEX_LIB:
    """
    This is a class which implements the main methods for inverting a patch of
    vorticity on a sphere.
    """

    """
    lon and lat are the grid for inversion, should be np.arrays and in degrees.
    dlon and dlat are the grid spacings defined by lon and lat.
    num_inv describes the number of iterative inversions
    """
    def __init__(self, lon, lat, num_inv=3, xres=1):
        # 9/10/2020: Change xres=1 (instead of 2) since ensemble resolutions
        # are becoming better (0.5 degrees), and can perform the inversion
        # with high accuracy.

        self.lon = lon
        self.lat = lat
        self.dlon = np.abs(lon[1] - lon[0])
        self.dlat = np.abs(lat[1] - lat[0])

        self.lon_rad = np.deg2rad(self.lon)
        self.lat_rad = np.deg2rad(self.lat)

        self.LON, self.LAT = np.meshgrid(self.lon_rad, self.lat_rad)
        self.num_inv = num_inv
        self.xres = xres

        self.d_crit = 400
        self.n_deg = 25

        # How to fill the vortex patch:
        #   'harmonic' : harmonic extension of the surrounding field
        #                (smooth, seamless, matches the environment)
        #   'zero'     : remove the full vorticity/divergence inside the
        #                patch, leaving zero
        self.env_mode = 'zero'

    def func_powerlaw(self, x, m, c, c0):
        return c0 + x**m * c

    def wrap_lon_indexing(self, f, latidxs, lonidxs):
        return f[latidxs, :].take(lonidxs, axis=1, mode='wrap')

    # Returns the center of vortex calculated using gradient descent.
    def vortex_center(self, rv, slon, slat):
        lon, lat = lon_lat(rv.shape[1], rv.shape[0])
        LON, LAT = np.meshgrid(lon, lat)
        dists = sphere.haversine(slon, slat, LON, LAT)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mask = dists < 400
        idx = np.argmax(rv[mask])
        return(LON[mask][idx], LAT[mask][idx])

    def border_mean(self, f, x, y):
        lon, lat = lon_lat(f.shape[1], f.shape[0])
        LON, LAT = np.meshgrid(lon, lat)
        dists = sphere.haversine(x, y, LON, LAT)
        dists[np.isnan(dists)] = 1e10

        delta = 0.25 * self.d_crit
        border = (dists >= (self.d_crit - delta)) & (dists <= self.d_crit)
        cos_lat = np.cos(np.deg2rad(LAT))
        return (np.nansum(f[border] * cos_lat[border]) /
                np.nansum(cos_lat[border]))

    def border_mean_theta(self, f, x, y, n_bins=8):
        """
        Environmental background inside the patch, allowing azimuthal variation.

        Average the border ring in `n_bins` coarse theta sectors, then
        interpolate smoothly in theta (periodic) across the whole patch.
        Constant in radius, so values inside stay small and bounded by the
        surrounding ring.
        """
        lon, lat = lon_lat(f.shape[1], f.shape[0])
        LON, LAT = np.meshgrid(lon, lat)
        dists = sphere.haversine(x, y, LON, LAT)
        dists[np.isnan(dists)] = 1e10

        # Azimuth about the storm centre (local tangent plane).
        coslat0 = np.cos(np.deg2rad(y))
        dx = ((LON - x + 180.0) % 360.0 - 180.0) * coslat0
        dy = (LAT - y)
        theta = np.arctan2(dy, dx)          # [-pi, pi]

        delta = 0.25 * self.d_crit
        border = (dists >= (self.d_crit - delta)) & (dists <= self.d_crit)

        cos_lat = np.cos(np.deg2rad(LAT))

        # Sector means around the ring.
        edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        sector = np.zeros(n_bins)
        for b in range(n_bins):
            m = border & (theta >= edges[b]) & (theta < edges[b + 1])
            if m.sum() == 0:
                sector[b] = np.nan
            else:
                sector[b] = (np.nansum(f[m] * cos_lat[m]) /
                             np.nansum(cos_lat[m]))

        # Fill any empty sector with the overall ring mean.
        if np.any(np.isnan(sector)):
            ring_mean = (np.nansum(f[border] * cos_lat[border]) /
                         np.nansum(cos_lat[border]))
            sector[np.isnan(sector)] = ring_mean

        # Smooth, periodic interpolation in theta: keep only the lowest
        # few azimuthal harmonics (mean + wavenumber 1 and 2). This avoids
        # bin-edge discontinuities and prevents the background from
        # chasing noise in any single sector.
        c = np.fft.rfft(sector)
        n_keep = 3                       # k = 0, 1, 2
        c[n_keep:] = 0.0
        theta_fit = np.fft.irfft(c, n=n_bins)

        # Evaluate the smooth theta function on the full grid.
        # Wrap centers so np.interp sees a monotonic, periodic axis.
        th_x = np.concatenate(([centers[-1] - 2*np.pi], centers,
                               [centers[0] + 2*np.pi]))
        th_y = np.concatenate(([theta_fit[-1]], theta_fit, [theta_fit[0]]))

        f_env = np.copy(f)
        inside = dists <= self.d_crit
        f_env[inside] = np.interp(theta[inside], th_x, th_y)
        return f_env

    def env_background(self, f, x, y):
        """
        Background field inside the vortex patch, per self.env_mode.
        Returns a full-grid array; the anomaly removed is w * (f - f_env).
        """
        if self.env_mode == 'zero':
            return np.zeros_like(f)
        elif self.env_mode == 'harmonic':
            return self.env_interp(f, x, y)
        else:
            raise ValueError("env_mode must be 'harmonic' or 'zero'.")
        
    def env_interp(self, f, x, y, d_crit=None):
        """
        Environmental background inside the vortex patch: the harmonic
        extension of the surrounding field.

        Solves Laplace's equation inside the patch with Dirichlet boundary
        conditions taken from the field just outside it. The result is the
        smoothest field consistent with the surroundings, and by the maximum
        principle it cannot exceed the range of the boundary values -- so the
        interior stays small and bounded, with no seam at the patch edge.
        Reduces exactly to the ring mean when the environment is uniform.
        """
        if d_crit is None:
            d_crit = self.d_crit

        Ny, Nx = f.shape
        lon, lat = lon_lat(Nx, Ny)
        LON, LAT = np.meshgrid(lon, lat)
        dists = sphere.haversine(x, y, LON, LAT)
        dists[np.isnan(dists)] = 1e10

        hole = dists < d_crit
        if not np.any(hole):
            return np.copy(f)

        # Local box around the hole, with a pad of Dirichlet cells.
        pad = 4
        jj, ii = np.where(hole)
        j0 = max(int(jj.min()) - pad, 0)
        j1 = min(int(jj.max()) + pad + 1, Ny)
        i0 = int(ii.min()) - pad
        i1 = int(ii.max()) + pad + 1

        # Longitude is periodic; latitude is not.
        i_idx = np.arange(i0, i1) % Nx
        fb = f[j0:j1][:, i_idx]
        hb = hole[j0:j1][:, i_idx]

        Ny_b, Nx_b = fb.shape
        N = Ny_b * Nx_b
        kk = np.arange(N).reshape(Ny_b, Nx_b)

        A = sp.lil_matrix((N, N))
        b = np.zeros(N)

        for j in range(Ny_b):
            for i in range(Nx_b):
                k = kk[j, i]
                if not hb[j, i]:
                    # Dirichlet: pinned to the observed environmental value.
                    A[k, k] = 1.0
                    b[k] = fb[j, i]
                else:
                    # Laplace. Uniform weights: the hole is small enough that
                    # the spherical metric is negligible over it, and this is
                    # a smoothness prior, not a physical operator.
                    A[k, k] = -4.0
                    for (dj, di) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        jn, iN = j + dj, i + di
                        if 0 <= jn < Ny_b and 0 <= iN < Nx_b:
                            A[k, kk[jn, iN]] = 1.0
                        else:
                            A[k, k] += 1.0        # no-flux; should not occur
                    b[k] = 0.0

        f_box = spsolve(A.tocsr(), b).reshape(Ny_b, Nx_b)

        f_env = np.copy(f)
        f_env[j0:j1, i_idx] = np.where(hb, f_box, fb)
        return f_env

    def vortex_search_asym(self, rv, x, y, f_env=None):
        lon, lat = lon_lat(rv.shape[1], rv.shape[0])
        LON, LAT = np.meshgrid(lon, lat)

        dists = sphere.haversine(x, y, LON, LAT)
        dists[np.isnan(dists)] = 1e10

        delta = 0.25 * self.d_crit
        w = np.ones_like(rv, dtype=float)

        ring = (dists >= (self.d_crit - delta)) & (dists <= self.d_crit)
        r = (dists[ring] - (self.d_crit - delta)) / delta
        w[ring] = 0.5 * (1 + np.cos(np.pi * r))
        w[dists > self.d_crit] = 0.0

        if f_env is None:
            f_env = self.env_interp(rv, x, y)

        # Anomaly is now relative to the interpolated environmental field,
        # not a single scalar, so the background gradient is preserved.
        rv_vort = w * (rv - f_env)
        rv_env  = rv - rv_vort

        return rv_env, rv_vort

    def cut_vortex(self, rv, x, y, f_env=None):
        return self.vortex_search_asym(rv, x, y, f_env=f_env)

    # Increases the resolution of vorticity by expanding in spherical harmonics
    def super_resolution(self, f, xres):
        if xres > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                # Expand into spherical harmonics
                f_coeffs = SHExpandDH(f, norm=4, sampling=2)
                f_coeffs = SHCoeffs.from_array(f_coeffs, normalization='ortho')

                ns = f_coeffs.coeffs.shape[1]   # native size (lmax + 1)
                ns_hd = ns * xres               # target size

                # Zero-pad the coefficients: band-limited interpolation.
                # No power is invented at high degrees, so the result is
                # deterministic and free of injected grid-scale noise.
                hres_coef = np.zeros((2, ns_hd, ns_hd), dtype=f_coeffs.coeffs.dtype)
                hres_coef[:, 0:ns, 0:ns] = f_coeffs.coeffs

                f_hd = MakeGridDH(hres_coef, sampling=2, norm=4)

                return f_hd
        else:
            return f
        
    # Recast a higher resolution wind field [lat, lon] back to lower resolution.
    # As in above, the vortex library must be consistent with the wnd array.
    def lower_res_f(self, f, xres):
        if xres == 1:
            return f

        lon, lat = lon_lat(f.shape[1], f.shape[0])

        dlon = np.round(np.abs(lon[1] - lon[0]), 4)
        dlat = np.round(np.abs(lat[1] - lat[0]), 4)
        X, Y = np.meshgrid(lon, lat)
        lon_diff = np.abs(np.mod(np.round(X, 4), dlon*xres))
        lat_diff = np.abs(np.mod(np.round(Y, 4), dlat*xres))
        lonidx = np.logical_or(lon_diff <= 1e-5, np.abs(lon_diff - (dlon*xres)) < 1e-5)
        latidx = np.logical_or(lat_diff <= 1e-5, np.abs(lat_diff - (dlat*xres)) < 1e-5)
        lowres_idx = np.logical_and(lonidx, latidx)

        return(np.reshape(f[lowres_idx],
                          (int(len(lat) / xres),
                           int(len(lon) / xres))))

    def build_laplacian_4th_order(self, Ny, Nx, R, COS, dlat_ravel, dlon_ravel):
        """
        Build a 4th-order accurate spherical Laplacian with Neumann BC.
        Returns A in LIL format.
        """

        N = Ny * Nx
        A = sp.lil_matrix((N, N))

        # Convenience: index(k,j) → flat index
        def idx(j,i): 
            return j*Nx + i

        # ---- 4th order coefficients ----
        # 1D second derivative stencil: [ -1, 16, -30, 16, -1 ] / (12*h^2)

        # Loop through interior points
        for j in range(Ny):
            for i in range(Nx):
                k = idx(j,i)
                cosphi = COS[k]
                dlat = dlat_ravel[k]
                dlon = dlon_ravel[k]

                # Longitudinal multiplier: (1 / (R^2 * cos^2 φ))
                lon_fac = 1.0 / (R*R * cosphi*cosphi)

                # Latitudinal metric terms:
                # 4th-order stencil for φ–derivative
                # Using uniform lat grid assumption:
                lat_fac = 1.0 / (R*R)

                # i-2
                if i-2 >= 0:
                    A[k, idx(j, i-2)] += -1.0 * lon_fac / (12*dlon*dlon)
                else:
                    # Neumann: mirror stencil
                    A[k, idx(j, i)] += -1.0 * lon_fac / (12*dlon*dlon)

                # i-1
                if i-1 >= 0:
                    A[k, idx(j, i-1)] += 16.0 * lon_fac / (12*dlon*dlon)
                else:
                    A[k, idx(j, i)] += 16.0 * lon_fac / (12*dlon*dlon)

                # center
                A[k, k] += -30.0 * lon_fac / (12*dlon*dlon)

                # i+1
                if i+1 < Nx:
                    A[k, idx(j, i+1)] += 16.0 * lon_fac / (12*dlon*dlon)
                else:
                    A[k, idx(j, i)] += 16.0 * lon_fac / (12*dlon*dlon)

                # i+2
                if i+2 < Nx:
                    A[k, idx(j, i+2)] += -1.0 * lon_fac / (12*dlon*dlon)
                else:
                    A[k, idx(j, i)] += -1.0 * lon_fac / (12*dlon*dlon)

                # j-2
                if j-2 >= 0:
                    A[k, idx(j-2, i)] += -1.0 * lat_fac / (12*dlat*dlat)
                else:
                    A[k, k] += -1.0 * lat_fac / (12*dlat*dlat)

                # j-1
                if j-1 >= 0:
                    A[k, idx(j-1, i)] += 16.0 * lat_fac / (12*dlat*dlat)
                else:
                    A[k, k] += 16.0 * lat_fac / (12*dlat*dlat)

                # center (lat)
                A[k, k] += -30.0 * lat_fac / (12*dlat*dlat)

                # j+1
                if j+1 < Ny:
                    A[k, idx(j+1, i)] += 16.0 * lat_fac / (12*dlat*dlat)
                else:
                    A[k, k] += 16.0 * lat_fac / (12*dlat*dlat)

                # j+2
                if j+2 < Ny:
                    A[k, idx(j+2, i)] += -1.0 * lat_fac / (12*dlat*dlat)
                else:
                    A[k, k] += -1.0 * lat_fac / (12*dlat*dlat)

        return A

    def spherical_laplacian_inversion(self, rv_box, lon_box, lat_box, order=2, bc='neumann'):
            """
            Numerical inversion of the Laplacian on a spherical box:
            Solve ∇²ψ = ζ (or ∇²φ = δ), matching sphere.sf_wnd and sphere.pot_wnd.

            Parameters
            ----------
            rv_box : 2D array (Ny, Nx)
                Vorticity or divergence field.
            lon_box : 1D array (Nx,)
                Longitudes in degrees.
            lat_box : 1D array (Ny,)
                Latitudes in degrees.
            order : int
                Finite-difference order (2 or 4).
            bc : str
                Boundary condition: 'neumann' (zero normal derivative) or
                'dirichlet' (psi = 0 on the box edge).

            Returns
            -------
            psi : 2D array (Ny, Nx)
                Streamfunction or velocity potential. Zero mean under Neumann;
                zero on the boundary under Dirichlet.
            """
            R = constants.R
            Ny, Nx = rv_box.shape
            N = Ny * Nx

            # Convert to radians
            lat_rad = np.deg2rad(lat_box)
            lon_rad = np.deg2rad(lon_box)
            dlat = np.gradient(lat_rad)
            dlon = np.gradient(lon_rad)

            # Half-grid cosines for metric
            cos_lat = np.cos(lat_rad)
            cos_lat_p = np.zeros(Ny)
            cos_lat_m = np.zeros(Ny)
            cos_lat_p[:-1] = (cos_lat[:-1] + cos_lat[1:]) / 2.0
            cos_lat_p[-1] = cos_lat[-1]
            cos_lat_m[1:] = (cos_lat[:-1] + cos_lat[1:]) / 2.0
            cos_lat_m[0] = cos_lat[0]

            # Flatten metrics for vectorized sparse matrix
            COS  = np.repeat(cos_lat[:, None], Nx, axis=1).ravel()
            COSP = np.repeat(cos_lat_p[:, None], Nx, axis=1).ravel()
            COSM = np.repeat(cos_lat_m[:, None], Nx, axis=1).ravel()
            dlat_ravel = np.repeat(dlat[:, None], Nx, axis=1).ravel()
            dlon_ravel = np.repeat(dlon[None, :], Ny, axis=0).ravel()

            # Finite difference coefficients
            diag         = -(1/(R**2 * dlat_ravel**2) * (COSP + COSM)/COS + 2/(R**2 * dlon_ravel**2 * COS**2))
            off_lat_up   = COSP / (R**2 * dlat_ravel**2 * COS)
            off_lat_down = COSM / (R**2 * dlat_ravel**2 * COS)
            off_lon      = 1 / (R**2 * dlon_ravel**2 * COS**2)

            # Build sparse matrix
            if order == 2:
                data = [diag.copy(),
                        off_lon[:-1].copy(),       # +1 lon
                        off_lon[:-1].copy(),       # -1 lon
                        off_lat_up[:-Nx].copy(),   # +Nx lat
                        off_lat_down[:-Nx].copy()] # -Nx lat
                offsets = [0, 1, -1, Nx, -Nx]
                A = sp.diags(data, offsets, shape=(N, N), format='lil')

                # sp.diags' ±1 bands couple the east edge of row j to the west edge of
                # row j+1. Remove those spurious wrap-around connections.
                for j in range(Ny - 1):
                    k = j*Nx + (Nx - 1)
                    A[k, k+1] = 0.0
                    A[k+1, k] = 0.0
            elif order == 4:
                A = self.build_laplacian_4th_order(
                    rv_box.shape[0], Nx, constants.R,
                    COS,
                    dlat_ravel, dlon_ravel
                )
            else:
                raise ValueError("Order must be 2 or 4.")

            b = rv_box.flatten().astype(float)

            if bc == 'neumann':
                # --- Neumann boundary conditions (zero normal derivative) ---
                # Left boundary (i=0)
                for j in range(Ny):
                    k = j*Nx
                    A[k, k] += off_lon[k]
                    A[k, k+1] -= off_lon[k]

                # Right boundary (i=Nx-1)
                for j in range(Ny):
                    k = j*Nx + (Nx-1)
                    A[k, k] += off_lon[k-1]
                    A[k, k-1] -= off_lon[k-1]

                # Bottom boundary (j=0)
                for i in range(Nx):
                    k = i
                    A[k, k] += off_lat_down[k]
                    A[k, k+Nx] -= off_lat_down[k]

                # Top boundary (j=Ny-1)
                for i in range(Nx):
                    k = (Ny-1)*Nx + i
                    A[k, k] += off_lat_up[k-Nx]
                    A[k, k-Nx] -= off_lat_up[k-Nx]

            elif bc == 'dirichlet':
                # --- Dirichlet boundary conditions (psi = 0 on the box edge) ---
                # Overwrite each boundary row with the identity so psi is pinned to 0.
                def is_edge(j, i):
                    return (j == 0) or (j == Ny - 1) or (i == 0) or (i == Nx - 1)

                for j in range(Ny):
                    for i in range(Nx):
                        if is_edge(j, i):
                            k = j*Nx + i
                            A.rows[k] = [k]      # clear off-diagonals in this row
                            A.data[k] = [1.0]
                            b[k] = 0.0

            else:
                raise ValueError("bc must be 'neumann' or 'dirichlet'.")

            # Solve
            A = A.tocsr()
            x = spsolve(A, b)

            # Reshape
            psi = x.reshape((Ny, Nx))

            # Under Neumann the solution is defined only up to a constant, so remove
            # the nullspace by enforcing zero mean. Under Dirichlet the boundary
            # pins the constant already, so leave psi as-is.
            if bc == 'neumann':
                psi -= np.mean(psi)

            return psi

    """
    Extracts a box of 20 by 20 degrees from the vortex center x, y.
    """
    def get_box(self, lon, lat, x, y, F):
        # Wrap longitude to ensure the box is not limited by the grid coordinates.
        lon_wrap = np.concatenate((lon - 360, lon, lon + 360))
        lon_idx = np.argmin(np.abs(lon_wrap - x))
        lat_idx = np.argmin(np.abs(lat - y))
        n_lon_deg = self.n_deg
        n_lat_deg = self.n_deg
        dlon = np.round(np.abs(lon_wrap[1] - lon_wrap[0]), 4)
        dlat = np.round(np.abs(lat[1] - lat[0]), 4)
        nx = int(n_lon_deg / dlon)
        ny = int(n_lat_deg / dlat)
        lon_box = lon_wrap[(lon_idx - nx):(lon_idx + nx)]
        lat_box = lat[(lat_idx - ny):(lat_idx + ny)]

        F_wrap = np.hstack((F, F, F))
        f_box = F_wrap[(lat_idx - ny):(lat_idx + ny), (lon_idx - nx):(lon_idx + nx)]
        return (lon_box, lat_box, f_box)

    """
    Sets the values of F_box.
    """
    def set_box(self, lon, lat, x, y, F, F_box):
        # Wrap longitude to ensure the box is not limited by the grid coordinates.
        lon_wrap = np.concatenate((lon - 360, lon, lon + 360))
        lon_idx = np.argmin(np.abs(lon_wrap - x))
        lat_idx = np.argmin(np.abs(lat - y))
        n_lon_deg = self.n_deg
        n_lat_deg = self.n_deg
        dlon = np.round(np.abs(lon[1] - lon[0]), 4)
        dlat = np.round(np.abs(lat[1] - lat[0]), 4)
        nx = int(n_lon_deg / dlon)
        ny = int(n_lat_deg / dlat)
        F_wrap = np.hstack((F, F, F))
        Fs_wrap = np.copy(F_wrap)
        Fs_wrap[(lat_idx - ny):(lat_idx + ny), (lon_idx - nx):(lon_idx + nx)] -= F_box

        # Return in the original coordinates.
        Fs = np.copy(Fs_wrap[:, lon.size:(2 * lon.size)])
        if (lon_idx - nx) < lon.size:
            # Take the western points and move it to the right side.
            Fs[:, ((lon_idx - nx) - lon.size):] = Fs_wrap[:, (lon_idx - nx):lon.size]
        elif (lon_idx + nx) >= (2 * lon.size):
            # Take the eastern points and move it to the left side.
            Fs[:, :(lon_idx + nx - (2 * lon.size))] = Fs_wrap[:, (2 * lon.size):(lon_idx + nx)]
        return (Fs)

    # Vortex surgery on a wind field uwnd, vwnd.
    # slon and slat represent the guess point for the vortex that will be removed.
    # Returns the wind fields (uFilt, vFilt) of the filtered environment,
    # as well as the wind fields of the extracted vortex, (uVort, vVort).
    # Uses multi-grid methods.
    # xres defaults to the value given to the constructor.
    # Returns (x, y, u_filt, v_filt, u_filt_HD, v_filt_HD).
    def vortex_surgery(self, u, v, slon, slat, xres=None):
        if xres is None:
            xres = self.xres

        u_hd = self.super_resolution(u, xres)
        v_hd = self.super_resolution(v, xres)
        lon_hd, lat_hd = lon_lat(u_hd.shape[1], u_hd.shape[0])

        rv0  = sphere.calc_rvor_sphere(lon_hd, lat_hd, u_hd, v_hd)
        div0 = sphere.calc_div_sphere(lon_hd, lat_hd, u_hd, v_hd)
        x, y = self.vortex_center(rv0, slon, slat)

        # Harmonic extension of the environment into the patch, computed
        # ONCE so the target does not drift with iteration.
        rv_env_bg  = self.env_background(rv0,  x, y)
        div_env_bg = self.env_background(div0, x, y)
        
        for i in range(self.num_inv):
            rv = sphere.calc_rvor_sphere(lon_hd, lat_hd, u_hd, v_hd)
            div = sphere.calc_div_sphere(lon_hd, lat_hd, u_hd, v_hd)
            _, rv_vort  = self.cut_vortex(rv,  x, y, f_env=rv_env_bg)
            _, div_vort = self.cut_vortex(div, x, y, f_env=div_env_bg)

            # Get relative voriticty and divergence over a box centered on vortex.
            lon_box, lat_box, rv_box = self.get_box(lon_hd, lat_hd, x, y, rv_vort)
            lon_box, lat_box, div_box = self.get_box(lon_hd, lat_hd, x, y, div_vort)

            # Invert the laplacian to calculate the stream function and velocity potential.
            psi = self.spherical_laplacian_inversion(rv_box, lon_box, lat_box)
            phi = self.spherical_laplacian_inversion(div_box, lon_box, lat_box)

            # Calculate velocities associated with Helmholtz decomposition.
            u_vort_rv, v_vort_rv = sphere.sf_wnd(lon_box, lat_box, psi)
            u_vort_div, v_vort_div = sphere.pot_wnd(lon_box, lat_box, phi)

            u_hd = self.set_box(lon_hd, lat_hd, x, y, u_hd, u_vort_rv + u_vort_div)
            v_hd = self.set_box(lon_hd, lat_hd, x, y, v_hd, v_vort_rv + v_vort_div)

        # Recast all the return fields into the original resolution.
        uf_s = self.lower_res_f(u_hd, xres)
        vf_s = self.lower_res_f(v_hd, xres)

        return(x, y, uf_s, vf_s, u_hd, v_hd)
