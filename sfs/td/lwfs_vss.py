from collections import namedtuple as _namedtuple
from copy import deepcopy
import numpy as _np
import numpy as np
from sfs import util as _util
from apicultor.constraints.dynamic_range import *
from apicultor.utils.algorithms import *
from apicultor.machine_learning.lstm_synth_w import *
from apicultor.sonification.Sonification import normalize, write_file
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import lfilter, fftconvolve, firwin, medfilt
from soundfile import read
import os
import sys
#import matplotlib.pyplot as plt
import numpy as np
import sfs
from sfs.td.source import point
from sfs.util import source_selection_focused,XyzComponents
from scipy.signal import unit_impulse
from apicultor.utils.algorithms import *
from pathos.pools import ProcessPool    
from glob import glob
from math import *
from random import *
import sys
import logging

#you should comment what you've already processed (avoid over-processing) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=SyntaxWarning) 
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
logging.basicConfig(level=logging.DEBUG)
logging.disable(logging.WARNING)
logger = logging.getLogger(__name__)

#c of throat
#We use it because we want to understand
#how the difracted wave field propagates as a thermal
#function, to understand how wave field is absorbed mechanically
#in the throat considering the anatomical features during perception
#that affect absorption mechanisms
ADIABATIC_CONSTANT = 1.4 #O2
#La presión que se refleja durante la fluctuación sonora
#se mide en decipascales
STATIC_PRESSURE_CGS = 1.013e6;   #deci-Pa = ubar
#Masa molecular de la tráquea
MOLECULAR_MASS = 28.94;		  #g/mol
GAS_CONSTANT = 8.314e7;		  #erg/K/mol por O2
KELVIN_SHIFT = 273.15;		#K	
#kpx0
NUM_CENTERLINE_POINTS_EXPONENT = 7;
NUM_CENTERLINE_POINTS = (1 << NUM_CENTERLINE_POINTS_EXPONENT) + 1;
#Considering sound propagation in throat
AIR_VISCOSITY_CGS = 1.86e-4;     #dyne-s/cm^2
HEAT_CONDUCTION_CGS = 0.055e-3;  #cal/cm-s-K
SPECIFIC_HEAT_CGS = 0.24;        #cal/g-K
#ecological interpretation in terms of temperature
#temperature = 28
temperature = 31.4266
volumic_mass = STATIC_PRESSURE_CGS * MOLECULAR_MASS / (GAS_CONSTANT * (temperature + KELVIN_SHIFT))
snd_speed = np.sqrt(ADIABATIC_CONSTANT * STATIC_PRESSURE_CGS / volumic_mass)
#viscous boundary length
lv = AIR_VISCOSITY_CGS / volumic_mass / snd_speed
viscous_bnd_spec_adm = np.complex128([1., 1.]) * np.sqrt(np.pi * lv / snd_speed) 
#thermal boundary size
lt = HEAT_CONDUCTION_CGS * MOLECULAR_MASS / volumic_mass / snd_speed / SPECIFIC_HEAT_CGS;
thermal_bnd_spec_adm = np.complex128([1., 1.]) * np.sqrt(np.pi * lt / snd_speed) * (ADIABATIC_CONSTANT - 1.);
area = lambda r: r**2 * np.pi

def strict_arange(start, stop, step=1, *, endpoint=False, dtype=None,
                  **kwargs):
    """Like :func:`numpy.arange`, but compensating numeric errors.

    Unlike :func:`numpy.arange`, but similar to :func:`numpy.linspace`,
    providing ``endpoint=True`` includes both endpoints.

    Parameters
    ----------
    start, stop, step, dtype
        See :func:`numpy.arange`.
    endpoint
        See :func:`numpy.linspace`.

        .. note:: With ``endpoint=True``, the difference between *start*
           and *end* value must be an integer multiple of the
           corresponding *spacing* value!
    **kwargs
        All further arguments are forwarded to :func:`numpy.isclose`.

    Returns
    -------
    `numpy.ndarray`
        Array of evenly spaced values.  See :func:`numpy.arange`.

    """
    remainder = (stop - start) % step
    if np.any(np.isclose(remainder, (0.0, step), **kwargs)):
        if endpoint:
            stop += step * 0.5
        else:
            stop -= step * 0.5
    elif endpoint:
        raise ValueError("Invalid stop value for endpoint=True")
    return np.arange(start, stop, step, dtype)

def xyz_grid(x, y, z, *, spacing, endpoint=True, N=3, **kwargs):
    """Create an 3xND grid with given range and spacing.

    Parameters
    ----------
    x, y, z : float or pair of float
        Inclusive range of the respective coordinate or a single value
        if only a slice along this dimension is needed.
    spacing : float or triple of float
        Grid spacing.  If a single value is specified, it is used for
        all dimensions, if multiple values are given, one value is used
        per dimension.  If a dimension (*x*, *y* or *z*) has only a
        single value, the corresponding spacing is ignored.
    endpoint : bool, optional
        If ``True`` (the default), the endpoint of each range is
        included in the grid.  Use ``False`` to get a result similar to
        :func:`numpy.arange`.  See `strict_arange()`.
    **kwargs
        All further arguments are forwarded to `strict_arange()`.

    Returns
    -------
    `XyzComponents`
        A grid that can be used for sound field calculations.

    See Also
    --------
    strict_arange, numpy.meshgrid

    """
    if np.isscalar(spacing):
        spacing = [spacing] * N
    ranges = []
    scalars = []
    for i, coord in enumerate([x, y, z]):
        if np.isscalar(coord):
            scalars.append((i, coord))
        else:
            #print("INDEX AT GRID", i)
            #print("COORD AT GRID", coord)
            start, stop = coord
            ranges.append(strict_arange(start, stop, spacing[i],
                                        endpoint=endpoint, **kwargs))
    grid = np.meshgrid(*ranges, sparse=True, copy=False)
    #print("MESHGRID", grid)
    for i, s in scalars:
        grid.insert(i, s)
    #print("GRID WITH SCALARS", grid)
    grid = XyzComponents(grid)
    # Obtener las coordenadas de los vértices
    vertices = np.column_stack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()])
    #print("VERTICES NUMBER", len(vertices[0]), len(vertices[1]), len(vertices[2]))
    
    # Obtener las caras de la malla
    nx, ny, nz = 3,3,3
    faces = []
    for i in range(3 - 1):
        for j in range(3 - 1):
            for k in range(3 - 1):
                # Cara 1
                faces.append([
                    i * ny * nz + j * nz + k,
                    (i + 1) * ny * nz + j * nz + k,
                    (i + 1) * ny * nz + (j + 1) * nz + k,
                    i * ny * nz + (j + 1) * nz + k
                ])
                # Cara 2
                faces.append([
                    i * ny * nz + j * nz + k + 1,
                    (i + 1) * ny * nz + j * nz + k + 1,
                    (i + 1) * ny * nz + (j + 1) * nz + k + 1,
                    i * ny * nz + (j + 1) * nz + k + 1
                ])
                # Cara 3
                faces.append([
                    i * ny * nz + j * nz + k,
                    (i + 1) * ny * nz + j * nz + k,
                    (i + 1) * ny * nz + j * nz + k + 1,
                    i * ny * nz + j * nz + k + 1
                ])
                # Cara 4
                faces.append([
                    (i + 1) * ny * nz + j * nz + k,
                    (i + 1) * ny * nz + (j + 1) * nz + k,
                    (i + 1) * ny * nz + (j + 1) * nz + k + 1,
                    (i + 1) * ny * nz + j * nz + k + 1
                ])
                # Cara 5
                faces.append([
                    i * ny * nz + (j + 1) * nz + k,
                    (i + 1) * ny * nz + (j + 1) * nz + k,
                    (i + 1) * ny * nz + (j + 1) * nz + k + 1,
                    i * ny * nz + (j + 1) * nz + k + 1
                ])
                # Cara 6
                faces.append([
                    i * ny * nz + j * nz + k,
                    i * ny * nz + (j + 1) * nz + k,
                    i * ny * nz + (j + 1) * nz + k + 1,
                    i * ny * nz + j * nz + k + 1
                ])    
    return XyzComponents(grid), vertices, faces

def bioestimulated_signal(f,xs=None):
        audio = mono_stereo(read(f)[0])
        impulse = unit_impulse(4096*2)
        impulse[1:] += 10e-5
        #Take all of xs greater than 30° back and forth
        #Take all of xcs greater than 30° back and forth    
        b = firwin(2, [0.05 / (2 ** (1 / (2 * -12))), 0.95 / (2 ** (1 / (2 * -12)))], width=0.05, pass_zero=False)
        #Torso throat, mouth and lips
        w_binaural_field, lr_binaural_field, height_binaural_field, deep_binaural_field = wfs_from_audio(impulse,xsi=xs)
        w_convolved = fftconvolve(w_binaural_field, b, mode='valid')
        w_convolved = np.vstack((np.nan_to_num(w_convolved, nan=10e-5, posinf=10e-5, neginf=10e-5), np.nan_to_num(w_convolved, nan=10e-5, posinf=10e-5, neginf=10e-5)))
        w_left = w_convolved[:int(w_convolved.shape[0]/2), :]
        w_right = w_convolved[int(w_convolved.shape[0]/2):, :]      
        w_h_sig_L = lfilter(w_left.flatten(), 1., audio)
        w_h_sig_R = lfilter(w_right.flatten(), 1., audio)
        w_result = np.float32([w_h_sig_L, w_h_sig_R]).T
        w_neg_angle = w_result[:,(1,0)]
        w_panned = w_result + w_neg_angle
        w_normalized = normalize(w_panned)
        #Omnidirectional torso, throat, lips radiation lobes filtering
        #b = firwin(2, [0.05, 7], width=0.05, pass_zero=False)
        #w_convolved = fftconvolve(w_binaural_field, b, mode='valid')   
        #w_panned = lfilter(w_convolved.flatten(), 1., lr_normalized)
        #w_normalized = normalize(w_panned)
        w_db_mag = 20 * np.log10(abs(w_normalized))
        #print(("Rewriting without silence in %s" % f))
        w_silence_threshold = -130  # complete silence
        w_loud_audio = np.delete(w_normalized, np.where(
                    w_db_mag < w_silence_threshold))  # remove it
        #LR Binaural filtering
        lr_convolved = fftconvolve(lr_binaural_field, b, mode='valid')
        lr_convolved = np.vstack((np.nan_to_num(lr_convolved, nan=10e-5, posinf=10e-5, neginf=10e-5), np.nan_to_num(lr_convolved, nan=10e-5, posinf=10e-5, neginf=10e-5)))
        lr_left = lr_convolved[:int(lr_convolved.shape[0]/2), :]
        lr_right = lr_convolved[int(lr_convolved.shape[0]/2):, :]      
        lr_h_sig_L = lfilter(lr_left.flatten()/np.max(lr_left), 1., audio)
        lr_h_sig_R = lfilter(lr_right.flatten()/np.max(lr_right), 1., audio)
        lr_result = np.float32([lr_h_sig_L, lr_h_sig_R]).T
        lr_neg_angle = lr_result[:,(1,0)]
        lr_panned = lr_result + lr_neg_angle
        lr_normalized = normalize(lr_panned)
        #LR Torso lobes filtering
        #b = firwin(2, [0.05, 7], width=0.05, pass_zero=False)
        #lr_convolved = fftconvolve(lr_binaural_field, b, mode='valid')   
        #lr_panned = lfilter(lr_convolved.flatten(), 1., lr_normalized)
        #lr_normalized = normalize(lr_panned)
        lr_db_mag = 20 * np.log10(abs(lr_normalized))
        #print(("Rewriting without silence in %s" % f))
        lr_silence_threshold = -130  # complete silence
        lr_loud_audio = lr_normalized
        #Height binaural filtering            
        height_convolved = fftconvolve(height_binaural_field, b, mode='valid')
        height_convolved = np.vstack((height_convolved,height_convolved))
        height_left = height_convolved[:int(height_convolved.shape[0]/2), :]
        height_right = height_convolved[int(height_convolved.shape[0]/2):, :]
        height_h_sig_L = np.nan_to_num(lfilter(height_left.flatten()/np.max(height_left), 1., audio), nan=10e-5, posinf=10e-5, neginf=10e-5)
        height_h_sig_R = np.nan_to_num(lfilter(height_right.flatten()/np.max(height_right), 1., audio), nan=10e-5, posinf=10e-5, neginf=10e-5)
        height_result = np.float32([height_h_sig_L, height_h_sig_R]).T
        height_neg_angle = height_result[:,(1,0)]
        height_panned = height_result + height_neg_angle
        height_normalized = normalize(height_panned)
        height_db_mag = 20 * np.log10(abs(height_normalized))
        #print(("Rewriting without silence in %s" % f))
        height_silence_threshold = -130  # complete silence
        height_loud_audio = height_normalized
        #Depth binaural filtering 
        deep_convolved = fftconvolve(deep_binaural_field, b, mode='valid')
        deep_convolved = np.vstack((deep_convolved,deep_convolved))
        deep_left = deep_convolved[:int(deep_convolved.shape[0]/2), :]
        deep_right = deep_convolved[int(deep_convolved.shape[0]/2):, :]
        deep_h_sig_L = np.nan_to_num(lfilter(deep_left.flatten()/np.max(deep_left), 1., audio), nan=10e-5, posinf=10e-5, neginf=10e-5)
        deep_h_sig_R = np.nan_to_num(lfilter(deep_right.flatten()/np.max(deep_right), 1., audio), nan=10e-5, posinf=10e-5, neginf=10e-5)
        deep_result = np.float32([deep_h_sig_L, deep_h_sig_R]).T
        deep_neg_angle = deep_result[:,(1,0)]
        deep_panned = deep_result + deep_neg_angle
        deep_normalized = normalize(deep_panned)
        deep_db_mag = 20 * np.log10(abs(deep_normalized))
        #print(("Rewriting without silence in %s" % f))
        deep_silence_threshold = -130  # complete silence
        deep_loud_audio = deep_normalized
        #print("W LEFT", w_h_sig_L)
        #print("LR LEFT", lr_h_sig_L)
        #print("HEIGHT LEFT", height_h_sig_L)
        #print("DEEP LEFT", deep_h_sig_L)    
        #print("W RIGHT", w_h_sig_R)
        #print("LR RIGHT", lr_h_sig_R)
        #print("HEIGHT RIGHT", height_h_sig_R)
        #print("DEEP RIGHT", deep_h_sig_R)    
        #print("W NEG ANGLE", w_neg_angle)
        #print("LR NEG ANGLE", lr_neg_angle)
        #print("HEIGHT NEG ANGLE", height_neg_angle)
        #print("DEEP NEG ANGLE", deep_neg_angle)        
        return np.nan_to_num(w_loud_audio, nan=10e-5, posinf=10e-5, neginf=10e-5),  np.nan_to_num(lr_loud_audio, nan=10e-5, posinf=10e-5, neginf=10e-5), np.nan_to_num(height_loud_audio, nan=10e-5, posinf=10e-5, neginf=10e-5), np.nan_to_num(deep_loud_audio, nan=10e-5, posinf=10e-5, neginf=10e-5)

def circular(N, R, center):
    """Return circular secondary source distribution parallel to the xy-plane.

    Parameters
    ----------
    N : int
        Number of secondary sources.
    R : float
        Radius in metres.
    center
        See `linear()`.

    Returns
    -------
    `SecondarySourceDistribution`
        Positions, orientations and weights of secondary sources.

    Examples
    --------
    .. plot::
        :context: close-figs

        x0, n0, a0 = sfs.array.circular(16, 1)
        sfs.plot2d.loudspeakers(x0, n0, a0, size=0.2, show_numbers=True)
        plt.axis('equal')
        plt.xlabel('x / m')
        plt.ylabel('y / m')

    """
    center = _util.asarray_1d(center)
    #alpha != s/R
    alpha = _np.linspace(0, 2 * _np.pi, N, endpoint=False)
    positions = _np.zeros((N, len(center)))
    positions[:, 0] = R * _np.cos(alpha/R)
    positions[:, 1] = R * _np.sin(alpha/R)
    positions[:, 2] = 0;
    positions[:, 3] = -np.cos(alpha/R);
    positions[:, 4] = -np.sin(alpha/R);
    positions[:, 5] = 0;
    normals = _np.zeros_like(positions)
    omega = 2 * np.pi * (alpha/R)
    normals[:, 0] = np.cos(omega)
    normals[:, 1] = np.sin(omega)
    #print("XL", normals)
    weights = _np.ones(N) * 2 * _np.pi * R / N
    #for throats
    contours = R * normals[:, 0]
    #print("CONTOURS", contours)    
    return positions, alpha, normals+0.2, weights

def correct_synthesis_frequency(x,xs,Rl,Rc,xl, xc, M, R, size,N,grid, vertices, faces):
    """%*****************************************************************************
    % Copyright (c) 2019      Fiete Winter                                       *
    %                         Institut fuer Nachrichtentechnik                   *
    %                         Universitaet Rostock                               *
    %                         Richard-Wagner-Strasse 31, 18119 Rostock, Germany  *
    %                                                                            *
    % This file is part of the supplementary material for Fiete Winter's         *
    % PhD thesis                                                                 *
    %                                                                            *
    % You can redistribute the material and/or modify it  under the terms of the *
    % GNU  General  Public  License as published by the Free Software Foundation *
    % , either version 3 of the License,  or (at your option) any later version. *
    %                                                                            *
    % This Material is distributed in the hope that it will be useful, but       *
    % WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY *
    % or FITNESS FOR A PARTICULAR PURPOSE.                                       *
    % See the GNU General Public License for more details.                       *
    %                                                                            *
    % You should  have received a copy of the GNU General Public License along   *
    % with this program. If not, see <http://www.gnu.org/licenses/>.             *
    %                                                                            *
    % http://github.com/fietew/phd-thesis                 fiete.winter@gmail.com *
    %*****************************************************************************"""

    if (not np.any(Rc)) and (not np.any(M)) and np.isnan(Rc) and np.isnan(M):
        return ValueError('rc and M cannot be defined at the same time!')

    #if geometry in ['circle', 'circular']:
    delta = len(xs)*np.pi/N;
    #elif geometry in ['line', 'linear']:
    #        delta = len(secondary_sources)/(N-1);
    #ideal conf
    #tmpconf = 2048; #% densely sampled SSD for calculations
    tmpconf = 307; #adjust tmp N to frame size
    x0full, alpha, xl, weights = circular(tmpconf, R, xc) #3,2
    #print("CIRCULAR X0FULL", x0full)  
    c = 340        
    #print("XS", np.any(xs < 0) )   
    xsR = np.resize(xs, (int(R/2),1,2))
    xs = xs.T[:3].T
    #Compute x0 from focused point sources
    x0full = point(xs, (x, 48000), 0.083, grid,c) 
    #print("GRID", grid, np.shape(grid) ) 
    select = source_selection_focused(np.mean(xl,axis=0)[:3], x0full[:3] , xs).T
    #There has to be one of x0s interacting with the harmonics radiation
    try:
        #print("X0FULL", np.shape(x0full[0, 0, :, 0, 0, 0:3]) ) 
        #print("SELECT", np.shape(select) ) 
        x0 = x0full[0, 0, :, 0, 0, 0:3].reshape(2, 3)[select].reshape(2,3)
    except Exception as e:
        #print("X0FULL", np.shape(x0full[0, 0, :, 0, 0, 0:3][select]) ) 
        x0 = x0full[0, 0, :, 0, 0, 0:3][select]
    #print("X0", np.any(x0 < 0), np.shape(x0))
    #print("XS", xs, np.shape(xs))     
    kPx0 = local_wavenumber_vector(x0,xs,'ps');  #% unit vectors        
    #print("kPx0", np.shape(kPx0)) 
    #print("X0FULL", x0full.T[:,3:6].T, np.shape(x0full.T[:,3:6].T))
    x0fulll = x0full[:,:3]
    xll = np.resize(xl[:,:3], np.shape(x0fulll) )
    landmark_difference = xll - x0fulll
    #print("X0FULLl", np.min(x0fulll), np.max(x0fulll))
    #print("XLL", np.shape(xl))
    delayed_sources = x0full.T[:,3:6].T
    delayed_sources_mat = np.resize(delayed_sources, np.shape(landmark_difference) )
    #print("LANDMARK DIFFERENCES", np.max(landmark_difference), np.min(landmark_difference))
    #print("DELAYED SOURCES", np.max(delayed_sources), np.min(delayed_sources))
    aliasing_region_error = np.sum( np.array([landmark_difference*delayed_sources_mat]) ,axis=2)
    #print("ALIASING REGION ERROR", np.max(aliasing_region_error), np.min(aliasing_region_error), np.any(aliasing_region_error < 0))


    #% is xl inside array?
    if np.any(aliasing_region_error < 0):
        fcs = np.NaN;
    else:    
        #2norm distance between xl and the line x0 + w*k_P(x0)     
        dlx0 = np.linalg.norm(np.cross((xl[:2,:3] - x0[:3]),kPx0[:3]), None);
        #% remove all x0, which do not contribute to the listening area
        #print("DLX0", dlx0)
        #print("RL", Rl)  
        select = dlx0 <= Rl;
        #print("IS LISTENING AREA ESTIMATE INSIDE LISTENING RADIUS?", np.shape(select))
        #print("KPX0", np.shape(kPx0))
        #print("KPX0 SHAPE", kPx0[:46][select])
        #print("XC", xc)
        #print("X0", x0[:46][select,:2])     
        #2norm distance between xc and the contour line x0 + w*k_P(x0)
        x0 = x0full[select, :3]
        #print("X0", np.shape(x0) )
        source_center_dist = np.expand_dims(xc, axis=1) - np.resize(x0,(46,1,2))
        select = np.resize(select,(2,3))
        expanded_kPx0 = np.reshape(kPx0[select], (3,2))
        #print("SOURCE CENTER_DIST", source_center_dist[:3][:, 0, :], np.shape(source_center_dist[:3][:, 0, :]))   
        #print("EXPANDED AREA kPx0", expanded_kPx0, np.shape(expanded_kPx0))   
        cx0 = np.cross( source_center_dist[:3][:, 0, :].T, expanded_kPx0.T)
        #print("CX0", cx0)
        dcx0 = np.linalg.norm(cx0,None);
        #print("DCX0", dcx0 )     
        #This creates a magnus effect of counterclockwise 
        #listening area expansion
        if (not np.any(Rc)) and np.isnan(Rc):        
            if any(dcx0 >= Rc):
                fcs = 0;
                velocity = 0
                wall_admittance = 0
                impedance = 0
            else:
                fcs = np.inf;  
                velocity = np.inf;  
                wall_admittance = np.inf; 
                impedance = np.inf; 
        else:
            fcs = np.min( M*c/(2.*np.pi*dcx0) );
            #print("F", fcs)
            fcs_idx = np.argmin( M*c/(2.*np.pi*dcx0) );
            #print("F INDEX", fcs_idx)
            #Omega it takes for the frequency to reach the vocal tract
            velocity = -1j * 2.0 * np.pi * fcs * volumic_mass
            print("VELOCITY", velocity)
            wall_admittance = 1j * 2. * np.pi * fcs * thermal_bnd_spec_adm / snd_speed
            #print("WALL ADMITTANCE", wall_admittance)
            field_f, modes, C, DN, E = compute_modes(grid, x0, xl, fcs, vertices, faces)
            impedance, junctions = propagate_imped_admit(fcs_idx, fcs_idx, fcs, 1,len(select),np.sign(dcx0), cx0, dcx0, R, E, Rl, C, DN, vertices)
            #print("IMPEDANCE", impedance)
            impedance = abs(1j * 2. * np.pi * fcs * volumic_mass * impedance[fcs_idx, 0])            
            inductance = 1 / (velocity**2 * wall_admittance)
            capacitance = 1 / (velocity**2 * impedance)
            #Coefficients for 2nd Order Butterworth Filter
            a = np.array([1,(-2*impedance*capacitance)[0][0] ,-1])
            b = np.array([(1 / -2*impedance*capacitance+1)[0][0],(1/(2*impedance*capacitance+1))[0][0],0])
            print("IMPEDANCE", impedance)
            
    #if exist(conf.fcsfile, 'file'):
    #  fcs = [gp_load(conf.fcsfile), fcs];
    #gp_save( conf.fcsfile, fcs );
    return min(fcs, 20000), cx0, velocity, wall_admittance, float(impedance[-1][-1]), a, b, x0, source_center_dist[:3][:, 0, :].T, dcx0, M*c/(2.*np.pi*dcx0), modes, C, DN, E, junctions, field_f

from scipy.linalg import lu, solve

def get_specific_bnd_adm(freq, m_eigen_freqs):
    """
    Calculate boundary admittance for a given simulation parameters and frequency.

    Parameters:
        simu_params (dict): Dictionary containing simulation parameters.
        freq (float): Frequency.
        m_eigen_freqs (numpy.ndarray): Eigen frequencies.

    Returns:
        numpy.ndarray: Boundary admittance.
    """
    bnd_spec_adm = []
    k = 2.0 * np.pi * freq / snd_speed
    percentage_loss = 1 #%
    for m in range(len(m_eigen_freqs)):
        bndi = percentage_loss * (
                ((1.0 - (2.0 * np.pi * m_eigen_freqs[m] / snd_speed)**2 / k**2) *
                 viscous_bnd_spec_adm + thermal_bnd_spec_adm) * np.sqrt(freq))
        #print("BAND SPECIFIC ADMITTANCE",bndi)         
        bnd_spec_adm.append(bndi)
    return np.array(bnd_spec_adm)
    

def propagate_imped_admit(start_imped, start_admit, 
                           f, start_section, end_section, 
                           direction, cx0, dcx0, R, E, Rl, C, DN, grid):
    #idx2time                       	
    prev_imped = np.zeros_like(start_imped)
    prev_admit = np.zeros_like(start_admit)
    #wall admittance
    wall_interface_admit = 1j * 2. * np.pi * f * thermal_bnd_spec_adm / snd_speed

    input_imped = []

    # Set initial impedance and admittance matrices

    # Set propagation direction of the first section
    dcx0z = np.sign(dcx0)
    dcx0y = np.sign(dcx0)
    
    junctions = compute_junction_matrices(3, end_section, 0, cx0, Rl)

    # Loop over sections
    input_imped = []
    for i in range(int(np.sign(dcx0)), len(cx0)):
        prev_sec = int(i - np.sign(dcx0))
        Z = np.sign(dcx0)
        Y = np.sign(dcx0)
        #print("cx0i",cx0[i])
        n_i = len(cx0[i])
        #print("PREV SEC",prev_sec)
        n_ps = len(cx0[prev_sec])

        # Extract scattering matrix and its complementary
        F = np.resize(junctions, (n_ps,n_ps)) + 0.083
        #print("NPS", n_ps)
        #print("NI", n_i)
        #print("F", F)
        g = np.eye(n_i) - np.dot(F.T, F) if direction == -1 else np.eye(n_ps) - np.dot(F, F.T)

        # Propagate admittance in the section
        # Case of MAGNUS propagation method
        prev_admit = propagate_magnus(cx0[i], f, direction, 'ADMITTANCE', prev_admit, C=C, E=E, DN=DN, R=R, Rl=Rl, field_f = [280,f] )
        #try:
        #    print("ADMITTANCE SHAPES", [len(i) for i in prev_admit] )
        #except Exception as e:
        #    pass  
        min_admittance_size = min([1 if isinstance(i, np.complex128) else len(i) for i in prev_admit])
        try:
            prev_admit =  np.array([i[:min_admittance_size] for i in prev_admit]).T   
        except Exception as e:
            pass        
        try:
            #print("ADMITTANCE SHAPES", [len(i) for i in prev_admit] )
            prev_admit =  np.array([i[:min_admittance_size] for i in prev_admit]).T   
        except Exception as e:
            pass  
        try:
            min_size = len(input_imped[0])     
        except Exception as e:
            min_size = 1          
        for it in prev_admit:                     
            try: 
                min_size = min(min_size, len(it))
                #print("IT", np.shape(it) ) 
            except Exception as e:
                min_size = min(min_size, 1)
            try: 
                input_imped.append(solve(lu(np.array([it])),np.eye(1))) #
            except Exception as e:
                try:
                    #logger.exception(e)
                    input_imped.append(lu(np.array(([it]))*np.eye(1)))
                except Exception as e:
                    min_it_size = min([len(i) for i in it])
                    input_imped.append(lu(np.array(([it[min_it_size]]))*np.eye(1)))   
        min_impedance_size = min([len(i) for i in input_imped])
        #print("MIN IMPEDANCE", min_impedance_size)
        try:
            #print("IMPEDANCE SHAPES", [len(i) for i in input_imped] )
            input_imped =  np.array([i[:min_impedance_size] for i in input_imped]).T   
        except Exception as e:
            #logger.exception(e)
            #print("IMPEDANCE SHAPES", [len(i) for i in input_imped] )
            input_imped =  np.array([i[:min_size] for i in input_imped]).T   
            pass   
        #print("IMPEDANCE", input_imped )
    return np.array(input_imped), F
    
def aliasing(x0, kSx0, x, conf):
    """%ALIASING aliasing frequency at position x
    %
    %   Usage: f = aliasing(x0, kSx0, x, conf)
    %
    %   Input options:
    %       x0              - position, direction, and sampling distance of 
    %                         secondary sources [N0x7] / m
    %       kSx0            - normalised local wavenumber vector kS(x0) 
    %                         of virtual sound field at x0 [N0x3]
    %       x               - position for which aliasing frequency is calculated
    %                         [Nx3]
    %       conf            - configuration struct (see SFS_config)
    %
    %   Output parameters:
    %       f   - aliasing frequency [Nx1]
    %

    %*****************************************************************************
    % Copyright (c) 2019      Fiete Winter                                       *
    %                         Institut fuer Nachrichtentechnik                   *
    %                         Universitaet Rostock                               *
    %                         Richard-Wagner-Strasse 31, 18119 Rostock, Germany  *
    %                                                                            *
    % This file is part of the supplementary material for Fiete Winter's         *
    % PhD thesis                                                                 *
    %                                                                            *
    % You can redistribute the material and/or modify it  under the terms of the *
    % GNU  General  Public  License as published by the Free Software Foundation *
    % , either version 3 of the License,  or (at your option) any later version. *
    %                                                                            *
    % This Material is distributed in the hope that it will be useful, but       *
    % WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY *
    % or FITNESS FOR A PARTICULAR PURPOSE.                                       *
    % See the GNU General Public License for more details.                       *
    %                                                                            *
    % You should  have received a copy of the GNU General Public License along   *
    % with this program. If not, see <http://www.gnu.org/licenses/>.             *
    %                                                                            *
    % http://github.com/fietew/phd-thesis                 fiete.winter@gmail.com *
    %*****************************************************************************"""

    #% the position x is a circular area with radius 0
    minmax_kGt_fun = minmax_kt_circle(x0p,xp,0);

    #% no restriction with respect to kSt(x0) (tangential component of kS(x0) )
    minmax_kSt_fun = minmax_kt_circle(x0p,[0,0,0],Inf);

    f, fx0 = aliasing_extended_control(x0, kSx0, x, minmax_kGt_fun, minmax_kSt_fun, conf);
    return f, fx0    
 
def aliasing_extended_control(x0, kSx0, x, minmax_kGt_fun, minmax_kSt_fun, conf):
    """%ALIASING_EXTENDED_CONTROL aliasing frequency for an extended listening area 
    %with an defined control area where the sound field synthesis is prioritized
    %
    %   Usage: f = aliasing_extended_control(x0, kSx0, x, minmax_kGt_fun, minmax_kSt_fun, conf)
    %
    %   Input options:
    %       x0              - position, direction, and sampling distance of 
    %                         secondary sources [N0x7] / m
    %       kSx0            - normalised local wavenumber vector kS(x0) 
    %                         of virtual sound field at x0 [N0x3]
    %       x               - position for which aliasing frequency is calculated
    %                         [Nx3]
    %       minmax_kGt_fun  - function handle to determine the extremal value of 
    %                         the tangential component of k_G(x-x0)
    %                         [kGtmin, kGtmax] = minmax_kGt_fun(x0,x)
    %       minmax_kSt_fun  - function handle to determine the extremal value of 
    %                         the tangential component of k_S(x0)
    %                         [kStmin, kStmax] = minmax_kSt_fun(x0)
    %       conf            - configuration struct (see SFS_config)
    %
    %   Output parameters:
    %       f   - aliasing frequency [Nx1]
    %

    %*****************************************************************************
    % Copyright (c) 2019      Fiete Winter                                       *
    %                         Institut fuer Nachrichtentechnik                   *
    %                         Universitaet Rostock                               *
    %                         Richard-Wagner-Strasse 31, 18119 Rostock, Germany  *
    %                                                                            *
    % This file is part of the supplementary material for Fiete Winter's         *
    % PhD thesis                                                                 *
    %                                                                            *
    % You can redistribute the material and/or modify it  under the terms of the *
    % GNU  General  Public  License as published by the Free Software Foundation *
    % , either version 3 of the License,  or (at your option) any later version. *
    %                                                                            *
    % This Material is distributed in the hope that it will be useful, but       *
    % WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY *
    % or FITNESS FOR A PARTICULAR PURPOSE.                                       *
    % See the GNU General Public License for more details.                       *
    %                                                                            *
    % You should  have received a copy of the GNU General Public License along   *
    % with this program. If not, see <http://www.gnu.org/licenses/>.             *
    %                                                                            *
    % http://github.com/fietew/phd-thesis                 fiete.winter@gmail.com *
    %*****************************************************************************"""


    phik = cart2pol(kSx0[:,1],kSx0[:,2]);  #% azimuth angle of kS(x0)
    phin0 = cart2pol(x0[:,4],x0[:,5]);  #% azimuth angle of normal vector n0(x0)

    #% secondary source selection
    select = np.cos(phin0 - phik) >= 0;  
    x0 = x0[select,:];
    #% kSt(x0) (tangential component of kS(x0) )
    kSt = np.sin(phin0[select] - phik[select]);  

    #% mininum and maximum values of kSt(x_0)
    [kStmin, kStmax] = minmax_kSt_fun(x0);
    select = kSt >= kStmin & kSt <= kStmax;
    x0 = x0[select,:];
    kSt = kSt[select];

    #% sampling distance
    deltax0 = abs(x0[:,7]);

    fx0 = inf(size(x,1), size(x0,1));
    for xdx in range(x[1].size):
        #% mininum and maximum values of k_Gt(x - x_0) 
        #% (tangential component of k_G(x-x0))
        [kGtmin, kGtmax] = minmax_kGt_fun(x0,x[xdx,:]);
        #% aliasing frequency for x
        fx0[xdx,:] = c/(deltax0*max(np.abs(kSt-kGtmin),np.abs(kSt-kGtmax)));

    return min(fx0, [], 2); 

def aliasing_extended_modal(x0, kSx0, x, minmax_kGt_fun, xc, M, Rl):
    """%ALIASING_EXTENDED_MODAL aliasing frequency for an extended listening area for 
    %an circular control area at xc with R=M/k where synthesis is focused on.
    %
    %   Usage: f = aliasing_extended_modal(x0, kSx0, x, minmax_kGt_fun, xc, M, conf)
    %
    %   Input options:
    %       x0              - position, direction, and sampling distance of 
    %                         secondary sources [N0x7] / m
    %       kSx0            - normalised local wavenumber vector of virtual sound 
    %                         field [N0x3]
    %       x               - position for which aliasing frequency is calculated
    %                         [Nx3]
    %       minmax_kGt_fun  - function handle to determine the extremal value of 
    %                         the tangential component of k_G(x-x0)
    %                         [kGtmin, kGtmax] = minmax_kGt_fun(x0,x)
    %       xc              - center of circular control area
    %       M               - modal order which defines the radius R=M/k
    %       conf            - configuration struct (see SFS_config)
    %
    %   Output parameters:
    %       f   - aliasing frequency [Nx1]
    %

    %*****************************************************************************
    % Copyright (c) 2019      Fiete Winter                                       *
    %                         Institut fuer Nachrichtentechnik                   *
    %                         Universitaet Rostock                               *
    %                         Richard-Wagner-Strasse 31, 18119 Rostock, Germany  *
    %                                                                            *
    % This file is part of the supplementary material for Fiete Winter's         *
    % PhD thesis                                                                 *
    %                                                                            *
    % You can redistribute the material and/or modify it  under the terms of the *
    % GNU  General  Public  License as published by the Free Software Foundation *
    % , either version 3 of the License,  or (at your option) any later version. *
    %                                                                            *
    % This Material is distributed in the hope that it will be useful, but       *
    % WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY *
    % or FITNESS FOR A PARTICULAR PURPOSE.                                       *
    % See the GNU General Public License for more details.                       *
    %                                                                            *
    % You should  have received a copy of the GNU General Public License along   *
    % with this program. If not, see <http://www.gnu.org/licenses/>.             *
    %                                                                            *
    % http://github.com/fietew/phd-thesis                 fiete.winter@gmail.com *
    %*****************************************************************************"""

    phik = cartesian_2_polar(kSx0[:,1],kSx0[:,2]);  #% azimuth angle of kS(x0)
    phin0 = cartesian_2_polar(x0[:,4],x0[:,5]);  #% azimuth angle of normal vector n0(x0)

    #% secondary source selection
    select = np.cos(phin0 - phik) >= 0;  
    x0 = x0[select,:];
    #% k_St(x0) (tangential component of k_S(x0) )
    kSt = np.sin(phin0[select] - phik[select]);  

    #% sampling distance
    deltax0 = abs(x0[:,7]);
    #print("x1", len(x))
    f = np.zeros(46)+np.inf;
    for xdx in range(1,1):
        #% mininum and maximum values of k_Gt(x - x_0) 
        #% (tangential component of k_G(x-x0))
        [kGtmin, kGtmax] = minmax_kGt_fun(x0,x[xdx,:],Rl);
        #% aliasing frequency for x0
        f0 = c/(deltax0*np.max(abs(kSt-kGtmin),abs(kSt-kGtmax)));
        #% radius of local region at f0
        rM = M*c/(2.*np.pi*f0);
        #% mininum and maximum values of kSt(x_0) at f0
        [kStmin, kStmax] = minmax_kt_circle(x0, xc, rM);
        select = kSt > kStmin & kSt < kStmax;
        if np.sum(select) == 0:
            f[xdx]  = np.min(f0[select] );
    return f        
    
def aliasing_extended(x0p,x0, kSx0, x, minmax_kGt_fun, minmax_kSt_fun, conf):
    #no restriction with respect to kSt(x0) (tangential component of kS(x0) )
    minmax_kSt_fun = minmax_kt_circle(x0p,[0,0,0],np.inf);

    f, fx0 = aliasing_extended_control(x0, kSx0, x, minmax_kGt_fun, minmax_kSt_fun, conf);
    return f,fx0    

def aliasing_extended_arbitrary_soundfield(x0p,x0, kSx0, x, minmax_kGt_fun, minmax_kSt_fun, c):
    """
    This methods applies aliasing for any arbitray soundfield
    """
    #sampling distance
    deltax0 = np.abs(x0[:,7]);

    fx0 = np.inf(np.size(x,1), x0[1].size);
    for xdx in range(1,x[1].size ):
        #mininum and maximum values of k_Gt(x - x_0) 
        #(tangential component of k_G(x-x0))
        [kGtmin, kGtmax] = minmax_kGt_fun(x0,x[xdx,:] );
        #aliasing frequency for x
        fx0 [xdx,:] = c/(deltax0*(1 + max(abs(kGtmin),abs(kGtmax))));
    
    f = min(fx0, [], 2);
    return f

def cartesian_2_polar(x,y):
    # Calculating the Euclidean distance (r) using the formula sqrt(x^2 + y^2)
    r = np.sqrt(x**2 + y**2)

    # Calculating the angles (t) using arctan2() function, which returns the arctangent of y/x in radians
    return  np.arctan2(y, x)

def local_aliasing_vector(x0p,x0, kSx0, x, minmax_kGt_fun, minmax_kSt_fun, conf):
    """
    This methods applies aliasing for any arbitray soundfield
    """
    # ===== Main ============================================================

    phik = cartesian_2_polar(kvec[:,1],kvec[:,2]);
    phin0 = cartesian_2_polar(n0[:,1],n0[:,2]);

    kSt0 = np.sin(phin0-phik);  #tangential component of virtual sound field

    phial = phin0 - np.asin(kSt0 + 2.*np.pi*eta/(k*delta));

    kvec = np.NaN(np.size(phial,1),2);
    select = np.imag(phial)==0;
    phial = phial(select);

    if (not np.any(phial)):
        kvec[select,:] = [np.cos(phial), np.sin(phial)];
    return kvec

def minmax_kt_circle(x0, xc, Rc):
    """
    This method optimizes the search for the aliasing free 
    region
    """
    phin0 = cartesian_2_polar(x0[:,4],x0[:,5] );

    # range for k_Gt(x-x_0) (tangential component of k_G)
    xc = xc - x0[:,1:3];
    phil, rc = cartesian_2_polar(xc[:,1],xc[:,2]);
    klt = np.sin(phin0 - phil);
    kln = np.sqrt(1 - klt^2);
    #Update rho from tangential component of kG
    rho = Rc/rc;

    kGtmin = np.zeros(size(xc,1),1);
    select = (rho > 1) or (-np.sqrt(1-rho^2) > klt);
    #select = np.where(rho > 1) || np.where(-np.sqrt(1-rho^2) > klt);
    kGtmin[select] = -1;
    kGtmin[~select] = klt[select]*np.sqrt(1-rho[select]^2) - kln[select]*rho[select];

    kGtmax = np.zeros(np.size(xc,1),1);
    select = rho > 1 | np.sqrt(1-rho^2) < klt;
    kGtmax[select] = +1;
    kGtmax[select] = klt[select]*np.sqrt(1-rho[select]^2) + kln[select]*rho[select];
    return kGtmin, kGtmax

def ear_filtering(fs,sig,f, song):
    """
    Get Composite Loudness Level after Pulkki 1999
    """
    # middle ear filtering

    bl, al = gammatone(f,48000)
    br, ar = gammatone(f,48000)
    
    sos = np.zeros((2, 6))
    sos[:2, :3] = butter(order, frequency / (fs / 2), btype='low', output='sos')
    sos[1:, 3:] = butter(order, frequency / (fs / 2), btype='low', output='sos')
    lr = sosfilt(sos, sig)
    return lr, lr

def perception_CLL(fs,sig, f):
    """
    Get Composite Loudness Level after Pulkki 1999
    """
    song = MIR(sig,48000)
    sig = song.IIR(sig,np.array([500, 2000])/(fs/2),'bandpass',-12);
    erb_bands = song.ERBBands()

    gammatone_left, gammatone_right = ear_filtering(fs,sig,erb_bands, song)
    
    # Cascade the filters to create a 4th order filter

    # left/right loudness in sones (Pulkki 1999, eq. 2)
    LL_sones = np.sqrt(rms(gammatone_left));
    RL_sones = np.sqrt(rms(gammatone_right));

    # composite loudness in sones (Pulkki 1999, eq. 2)
    CL_sones = LL_sones + RL_sones;

    # composite loudness level in phones (Pulkki 1999, eq. 3)
    CLL_phones = 40 + 10*np.log2(CL_sones);
    return CLL_phones

def perception_ITDILDIC(sig, fs, rf, f):
    """
    Get ITD, ILD, IC
    """
    # middle ear filtering
    song = MIR(sig,48000)
    sig = song.IIR(sig,np.array([500, 2000])/(fs/2),'bandpass',-12);
    erb_bands = song.ERBBands()

    gammatone = ear_filtering(fs,sig,erb_bands, song)
    
    rectified = np.maximum(gammatone_left, 0), np.maximum(gammatone_right, 0)

    ciliad = song.IIR(rectified,song.nyquist(1000),'lowpass',6);    
    # output of auditory frontend
    ild = ild(sig)  # ild (linear)
    itd = itd_ild(gammatone_left, gammatone_right, rf,fs)  # itd in seconds
    ic = ic(sig)  # interaural coherence [0..1]
    return itd, ild, ic, fc

def wrapToInterval(x, a, b):
    """wrap value to interval"""
    return np.mod(x - a, b - a) + a;

def local_wavenumber_vector(x, xs, src):
    if src == 'ps': #point source
        kvec = x - xs;
    elif src == 'fs': #focused source
        kvec = xs[:2] - x;
    elif src == 'pw': #plane wave
        kvec = np.matlib.repmat(xs,[np.size(x,0),0]);
    #Normalize based on maximum sample    
    return np.true_divide(kvec, np.linalg.norm(kvec,None)); 
    
def allcombs(A,A2,An):
    """% generate all combinations of cell arrays' elements (similar to ndgrid)
    %
    % usage:
    %   combMat = allcombs(A1,A2,...,An)
    %   combMat = allcombs(A1,A2,...,An,groups)
    %
    % inputs:
    %   A1,A2,...,An - cell arrays
    %   groups       - vector of indexes indicating which cell arrays are merged
    %                  into one group (and are therefore not combined).
    %
    % By default, any combination of the elements of A1,A2,...,An is generated. 
    % However, giving a vector (groups) as the last arguments allows to group 
    % cell arrays. The vector has to have the same length as he number of cell 
    % arrays. The i-th element of the vector belongs to the i-th cell array. Cell
    % arrays with the same index belong to one group. Cell arrays of the same
    % group have to have the same size.
    % A group is basically treated as one single cell array input. Hence, the 
    % elements of the cell arrays in one group are not combined with each other.
    %
    % outputs:
    %   combs - cell array where each column contains elements of the respective
    %           cell array A1,A2,...,An. Each row represents one combination.
    %
    % example:
    %   A1 = {'blub', 'test'}
    %   A2 = {2, 'hello world', 'hihi', pi}
    %   A3 = {1, inf }
    %
    %   combs = allcombs(A1, A2, ..., An, [1,2,1])  % A1 and A3 in one group
    """
    #%% ===== Computation =========================================================

    #% remove duplicates from sizeVec, i.e. group dimensions
    ia, ic = np.unique( groups, 'stable');
    sizeVec = sizeVec( ia );

    #% generate a cell array where each element i contains the indices starting
    #% from 1 to sizeVec(i)
    indices = np.fliplr([ n[:N] for n in sizeVec] );
    #% make a grid of all combinations of those indices
    [indices[:]] = np.ndgrid(indices[:]);
    indices = np.fliplr(indices);
    #% vectorize each indices matrix and repeat grouped indices
    indices = np.array([idx[:] for idx in indices[ic] ])
    #% finally create cell array of all combinations
    combs = np.array([np.resize( c[idx[:]], [], 1) for idx,c in zip(An, indices) ])
    combs = np.array([combs]);
    return combs

def spatial_aliasing_frequency(x, xs, Rl, Rc, xl, xc, M, N, grid):
    """
    Compute spatial aliasing frequency given specific geometry
    """

    if (not np.any(Rc)) and (not np.any(M)) and np.isnan(Rc) and np.isnan(M):
        return ValueError('rc and M cannot be defined at the same time!')

    #if geometry in ('circle', 'circular'):
    deltax0 = N*np.pi/307;
    #elif geometry in ('line', 'linear'):
    #    deltax0 = N/(secondary_sources.number-1);

    tmpconf = 2**10; #% densely sampled SSD for calculations
    x0full, alpha, xl, weights = circular(tmpconf, 46*2, xc) #3,2
    c = 340          
    x0t = point(xs, (x, 48000), 0.083, grid,c) 
    select = source_selection_focused(np.mean(xl,axis=0), x0full , xs).T
    #print("SELECT", np.shape(select))
    #print("x0T", np.shape(x0t))
    x0 = np.resize(x0t[0, 0, :, 0, 0, :][np.resize(select, (2,35))],(2,35))
    #print("FOCUSED X0", x0, np.shape(x0))
    x0[:,7] = deltax0;

    kSx0 = local_wavenumber_vector(x0[:,:3],xs[:,:3], 'ps');  #point source unit vectors
    #print("kSx0", kSx0, np.shape(kSx0))
    if (not np.any(Rc)) and np.isnan(Rc):
        #next bound at x0 middlepoint boundary
        circle_x0t = minmax_kt_circle(left_x0t, xc, Rc);
        #compute aliasing frequency from circle boundary
        fS  = aliasing_extended_control(x0, kSx0, xl, (left_x0t,xt), circle_x0t, conf);
    else:
        #optimize to compute aliasing frequency
        fS = aliasing_extended_modal(x0, kSx0, xl, minmax_kt_circle, xc, M, Rl);
    #Omega it takes for the frequency to reach the vocal tract
    #if os.path.exists(falfile):
    #  fS = [gp_load(conf.falfile); fS];

    #gp_save( conf.falfile, fS );
    #print("fS", fS)
    return fS, x0

green_field = lambda r, omega, c: np.nan_to_num(1/(4*np.pi) * (1e0*omega/c + 1/r) * np.exp(-1e0*omega/c*r) /r**2, nan=10e-5, posinf=10e-5, neginf=10e-5);

def greens_gradient_mono(x,y,z,xs,src,f, phase, c ):
    """%GREENS_FUNCTION_MONO returns a Green's function in the frequency domain
    %
    %   Usage: G = greens_function_mono(x,y,z,xs,src,f,conf)
    %
    %   Input options:
    %       x,y,z   - x,y,z points for which the Green's function should be
    %                 calculated / m
    %       xs      - position of the source
    %       src     - source model of the Green's function. Valid models are:
    %                   'ps'  - point source
    %                   'ls'  - line source
    %                   'pw'  - plane wave
    %                   'dps' - dipole point source
    %       f       - frequency of the source / Hz
    %       conf    - configuration struct (see SFS_config)
    %
    %   Output parameters:
    %       G       - Green's function of the size of sources evaluated at the points x,y,z
    %
    %   GREENS_FUNCTION_MONO(x,y,z,xs,src,f,conf) calculates the Green's function
    %   for the given source model located at xs for the given points x,y and the
    %   frequency f.
    %
    %   See also: sound_field_mono
    
    %*****************************************************************************
    % The MIT License (MIT)                                                      *
    %                                                                            *
    % Copyright (c) 2010-2017 SFS Toolbox Developers                             *
    %                                                                            *
    % Permission is hereby granted,  free of charge,  to any person  obtaining a *
    % copy of this software and associated documentation files (the "Software"), *
    % to deal in the Software without  restriction, including without limitation *
    % the rights  to use, copy, modify, merge,  publish, distribute, sublicense, *
    % and/or  sell copies of  the Software,  and to permit  persons to whom  the *
    % Software is furnished to do so, subject to the following conditions:       *
    %                                                                            *
    % The above copyright notice and this permission notice shall be included in *
    % all copies or substantial portions of the Software.                        *
    %                                                                            *
    % THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
    % IMPLIED, INCLUDING BUT  NOT LIMITED TO THE  WARRANTIES OF MERCHANTABILITY, *
    % FITNESS  FOR A PARTICULAR  PURPOSE AND  NONINFRINGEMENT. IN NO EVENT SHALL *
    % THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
    % LIABILITY, WHETHER  IN AN  ACTION OF CONTRACT, TORT  OR OTHERWISE, ARISING *
    % FROM,  OUT OF  OR IN  CONNECTION  WITH THE  SOFTWARE OR  THE USE  OR OTHER *
    % DEALINGS IN THE SOFTWARE.                                                  *
    %                                                                            *
    % The SFS Toolbox  allows to simulate and  investigate sound field synthesis *
    % methods like wave field synthesis or higher order ambisonics.              *
    %                                                                            *
    % http://sfstoolbox.org                                 sfstoolbox@gmail.com *
    %*****************************************************************************
    
    
    %% ===== Checking of input  parameters ==================================
    % Disabled checking for performance reasons
    
    
    %% ===== Configuration =================================================="""
    W = np.zeros(46)
    Y = np.zeros(46)
    Z = np.zeros(46)
    Y[0] = y[0]
    Z[0] = z[0]
    Y[1] = y[1]
    Z[1] = z[1]
    w = x[1]
    y = Y
    z = Z
    x = x[:46]
    #%% ===== Computation =====================================================
    #% Frequency
    omega = 2*np.pi*f;
    #% Calculate Green's function for the given source model
    if 'ps' == src:
        """% Source model for a point source: 3D Green's function.
        %
        %                   1   / iw       1    \    (xs-x) 
        % grad G(x-xs,w) = --- | ----- + ------- | ----------- e^(-i w/c |x-xs|)
        %                  4pi  \  c     |x-xs| /    |x-xs|^2
        %"""
        #print("XS", xs)
        #print("X", x)
        #print("XS1", xs[1])
        #print("XS1", xs[2])
        #print("X GREEN", x)
        #print("Y GREEN", y)
        #print("Z GREEN", z)
        #print("PHASE", phase)
        r = np.nan_to_num(np.sqrt((x[0] -xs[0]**2+(y[0]-xs[1]**2+(z[0]-xs[2] )**2))), nan=10e-5, posinf=10e-5, neginf=10e-5);
        rw = np.nan_to_num(np.sqrt((x[1] -xs[0]**2+(y[1]-xs[1]**2+(z[1]-xs[2] )**2))), nan=10e-5, posinf=10e-5, neginf=10e-5);        
        G = green_field(r, omega, snd_speed)        
        Gw = green_field(rw, omega, snd_speed)      
        #print("G", G) 
        #print("R", r)         
        #print("OMEGA", omega)        
        Gw = (xs[0]-w)*Gw;
        Gx = (xs[0]-x[0])*G;
        Gy = (xs[1]-y[0])*G;
        Gz = (xs[2]-z[0])*G;
    elif 'pw' == src:
        """% Source model for a plane wave:
        %
        %            -iw
        % grad Ppw = --- n e^(-i w/c n x)
        %             c
        %    
        % Direction of plane wave"""
        nxs = xs[:,:2] / np.linalg.norm(xs[:,:2])  ;
        #% Calculate sound field
        G = -1e0*omega/c*np.exp(-1e0*omega/c*(nxs[0]*x+nxs[1]*y+nxs[2]*z));        
        Gw = (xs[:,0]-w)*G;
        Gx = nxs[:,0]*G;
        Gy = nxs[:,1]*G;
        Gz = nxs[:,2]*G;
    else:
        return None

    #% Add phase to be able to simulate different time steps
    #print("PHASE ", phase)
    #print("EXP ", np.exp(-1e0*phase))
    Gw = Gw * np.exp(-1e0*phase);
    Gx = Gx * np.exp(-1e0*phase);
    Gy = Gy * np.exp(-1e0*phase);
    Gz = Gz * np.exp(-1e0*phase);
    return Gw, Gx, Gy, Gz

def sound_field_gradient_mono(X,Y,Z,x0,src,D,f, field_f, theta):
    """SOUND_FIELD_MONO simulates a monofrequent sound field for the given driving
    signals and secondary sources

    Usage: [P,x,y,z] = sound_field_mono(X,Y,Z,x0,src,D,f,conf)

    Input parameters:
        X           - x-axis / m; single value or [xmin,xmax] or nD-array
        Y           - y-axis / m; single value or [ymin,ymax] or nD-array
        Z           - z-axis / m; single value or [zmin,zmax] or nD-array
        x0          - secondary sources / m [nx7]
        src         - source model for the secondary sources. This describes the
                    Green's function, that is used for the modeling of the
                    sound propagation. Valid models are:
                       'ps' - point source
                       'ls' - line source
                       'pw' - plane wave
       D           - driving signals for the secondary sources [mxn]
       f           - monochromatic frequency / Hz
       conf        - configuration struct (see SFS_config)
   Output parameters:
      P           - Simulated sound field
       x           - corresponding x values / m
       y           - corresponding y values / m
       z           - corresponding z values / m

   SOUND_FIELD_MONO(X,Y,Z,x0,src,D,f,conf) simulates a monochromatic sound
   field for the given secondary sources, driven by the corresponding driving
   signals. The given source model src is applied by the corresponding Green's
   function for the secondary sources. The simulation is done for one
   frequency in the frequency domain, by calculating the integral for P with a
   summation.
   For the input of X,Y,Z (DIM as a wildcard) :
     * if DIM is given as single value, the respective dimension is
     squeezed, so that dimensionality of the simulated sound field P is
     decreased by one.
     * if DIM is given as [dimmin, dimmax], a linear grid for the
     respective dimension with a resolution defined in conf.resolution is
     established
     * if DIM is given as n-dimensional array, the other dimensions have
     to be given as n-dimensional arrays of the same size or as a single value.
     Each triple of X,Y,Z is interpreted as an evaluation point in an
     customized grid.

   To plot the result use:
   plot_sound_field(P,X,Y,Z,conf);
   or simple call the function without output argument:
   sound_field_mono(X,Y,Z,x0,src,D,f,conf)

   See also: plot_sound_field, sound_field_imp

    """
    #% ===== Computation ====================================================
    #% Create a x-y-z-grid
    R = 46*2
    N = 46
    x1  = N # get first non-singleton axis 
    #x1 = 4101   
    # Initialize empty sound field
    Pw = deepcopy(np.zeros(x1));
    Px = deepcopy(np.zeros(x1));
    Py = deepcopy(np.zeros(x1));
    Pz = deepcopy(np.zeros(x1));
    field_f = np.append(field_f, f)
    #print("GREEN FREQUENCIES", field_f)

    #Integration over secondary source positions
    song = MIR(x0,48000)
    #print("D", np.shape(D[0]))
    omega = 2*np.pi*f;     
    #x0 less or equal than the wavelength of the signal 
    for ii in range(1,len(D[0])):
        """ ====================================================================
        % Secondary source model G(x-x0,omega)
        % This is the model for the secondary sources we apply.
        % The exact function is given by the dimensionality of the problem, e.g. a
        % point source for 3D
        [Gx,Gy,Gz] = greens_gradient_mono(xx,yy,zz,x0(ii,1:6),src,f,conf);

        % ====================================================================
        % Integration
        %              /
        % P(x,omega) = | D(x0,omega) grad G(x-x0,omega) dx0
        %              /
        %
        % See http://sfstoolbox.org/#equation-single-layer
        %"""
        try:
            #x0[ii,7] is a weight for the single secondary sources which includes for
            #example a tapering window for WFS or a weighting of the sources for
            #integration on a sphere
            #print("X0 SOUND FIELD GRADIENT", x0)
            song.frame = x0[:5,ii]    
            #print("X0", x0[:5,ii])
            song.M = len(song.frame)
            song.H = int(song.M/2)
            song.window()
            #print("WINDOWED X", song.windowed_x, np.shape(song.windowed_x))            
            song.Phase(np.nan_to_num(np.fft.fft(np.nan_to_num(song.windowed_x, nan=10e-5, posinf=10e-5, neginf=10e-5), n=len(D[0])*4))) 
            #print("WINDOWED X", np.nan_to_num(song.windowed_x))
            #print("FFT", np.nan_to_num(np.fft.fft(song.windowed_x, n=len(D[ii])*4)))
            #print("PHASE", song.phase)
            green_points = np.array([greens_gradient_mono( X, Y, Z,song.frame, src, f, song.phase[:len(D[0])], snd_speed) for f in field_f])
            #print("GREEN POINTS", np.shape(green_points))
            Gw = green_points
            #print("GW", np.shape(Gw.T))
            Gx = green_points[-1][1]
            #print("GX", np.shape(Gx))
            Gy = green_points[-1][2]
            #print("GY", np.shape(Gy))
            Gz = green_points[-1][-1]
            #print("GZ", np.shape(Gz))
            weight = x0[6,ii]
            #print("Gx", Gx)
            #print("WEIGHT", weight)       
            #Axis 2 == axis of alpha (the radius of the optimal area)
            #Axis 1 == axis of phi (the amplitude inside the listening and breathing area)
            spacing = 4
            Pw += 1/omega * np.sum(np.sum((np.cos(theta-spacing/2)-np.cos(theta+spacing/2)*spacing)*Gw.T,axis=2),axis=1)* weight * D[ii]   
            #print("PW", np.shape(Pw))      
            Px += np.nan_to_num(D[ii] * Gx * weight, nan=10e-5, posinf=10e-5, neginf=10e-5)
            Py += np.nan_to_num(D[ii] * Gy * weight, nan=10e-5, posinf=10e-5, neginf=10e-5)
            Pz += np.nan_to_num(D[ii] * Gz * weight, nan=10e-5, posinf=10e-5, neginf=10e-5)         
            #print("PX",  np.shape(Px))   
            #print("PZ",  np.shape(Py))   
            #print("PZ",  np.shape(Pz)) 
        except Exception as e:
            #logger.exception(e)
            break
    # return parameter
    return Pw, Px, Py, Pz, X, Y, Z
    
def itd_ild(chan_a, chan_b, rf,fs):
    fs = 48000 # sampling frequency
    T = 0.02 #integration constant in seconds, same as grid
    T = T * fs # convert to samples
    mic_dist = rf
    soundspeed = 340
    max_ITD = mic_dist / soundspeed # max interaural time difference in seconds
    max_ITD = np.floor( max_ITD * fs ) # max in samples

    corr_arr = np.zeros(int(2 * max_ITD + 1))
    #print("Max ITD", max_ITD)
    #print("CORRELATION ARRAY", corr_arr)

    for sample in range(int(1 + max_ITD), int(1000 - max_ITD)):
        for itd in range(int(-max_ITD), int(max_ITD)):
            corr_arr[int(itd + max_ITD + 1)] = np.exp(-1/T) * corr_arr[int(itd + max_ITD + 1)] + chan_a[sample] + chan_b[sample - itd]
            # corr_arr[0][itd + max_ITD + 1] = math.exp(-1/integration_constant) * corr_arr[0][itd + max_ITD + 1] + l[sample] + r[sample - itd]
    #print("ITD CORRELATIONS", corr_arr)
    x = np.arange(int(-max_ITD),int(max_ITD+1))
    maximum = np.argmax(corr_arr)  # find the index where the correlation is maximum value
    corr_max = corr_arr[maximum] - max_ITD - 1 # get the corresponding ITD (in samples)
    #print("MAX CORR", corr_max)    
    return max_ITD, 20*np.log10(corr_max)


def calc_pan(index):
    return cos(radians(index))

#playlist_songs = [AudioSegment.from_mp3(mp3_file) for mp3_file in glob("mp3/*.mp3")]

#first_song = playlist_songs.pop(0)

def delay_focused_25d(x0, n0, xs, ns, xref=[0, 0, 0], c=None):
    r"""Point source by 2.5-dimensional WFS.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of secondary source orientations.
    xs : (3,) array_like
        Virtual source position.
    ns : (3,) array_like
        Normal vector (propagation direction) of focused source.
        This is used for secondary source selection,
        see `sfs.util.source_selection_focused()`.
    xref : (3,) array_like, optional
        Reference position
    c : float, optional
        Speed of sound

    Returns
    -------
    delays : (N,) numpy.ndarray
        Delays of secondary sources in seconds.
    weights: (N,) numpy.ndarray
        Weights of secondary sources.
    selection : (N,) numpy.ndarray
        Boolean array containing ``True`` or ``False`` depending on
        whether the corresponding secondary source is "active" or not.
    secondary_source_function : callable
        A function that can be used to create the sound field of a
        single secondary source.  See `sfs.td.synthesize()`.

    Notes
    -----
    2.5D correction factor

    .. math::

         g_0 = \sqrt{\frac{|x_\mathrm{ref} - x_0|}
         {|x_0-x_s| + |x_\mathrm{ref}-x_0|}}


    d using a point source as source model

    .. math::

         d_{2.5D}(x_0,t) =
         \frac{g_0  \scalarprod{(x_0 - x_s)}{n_0}}
         {|x_0 - x_s|^{3/2}}
         \dirac{t + \frac{|x_0 - x_s|}{c}}  \ast_t h(t)

    with wfs(2.5D) prefilter h(t), which is not implemented yet.

    See :sfs:`d_wfs/#equation-td-wfs-focused-25d`

    Examples
    --------
    .. plot::
        :context: close-figs

        delays, weights, selection, secondary_source = \
            sfs.td.wfs.focused_25d(array.x, array.n, xf, nf)
        d = sfs.td.wfs.driving_signals(delays, weights, signal)
        plot(d, selection, secondary_source, t=tf)

    """
    if c is None:
        c = snd_speed
    x0 = _util.asarray_of_rows(x0)
    n0 = _util.asarray_of_rows(n0)
    #xs = _util.asarray_1d(xs)
    xref = _util.asarray_1d(xref)
    ds = _np.array([x0 - xsi for xsi in xs])
    r = _np.linalg.norm(ds, axis=1)
    delays = -r/c
    #print("DELAYS", delays)    
    return delays

def synthesize_frames(D, delays, x, ts, p, N, rf, y, z, x0, f,  Rl, Rc, xl, xc, xs, M, i, field_f, theta):
    if np.shape(D)[0] == 0:
        #print("EMPTY D")
        return 0
    #compute signal values at every point in the grid
    #output value is of the shape of the grid
    try:
        #print("SIGNAL", D, "OF SIZE", np.array(D).shape)
        #Compute DelayedSignal for the frame of the signal 
        #print("DELAYS", delays, "AND WEIGHTS", x0[:,6])
        driven_signal = sfs.td.wfs.driving_signals(delays, x0[:,6] , D[:,0] )     
        #print("DRIVEN SIGNAL", driven_signal.data)
        #print("ALL ZERO", np.any(driven_signal.data) != 0)
        #print("SYNTHESIZING FRAME", i)
        if (np.any(driven_signal.data) != 0):
            pass
        else:
            return 0
        #compute soundfield using the driven signal, x0, the correct aliasing frequency and natural exponent 
        #for gradients of point sources
        Pw, Px,Py,Pz, xx,yy,zz = sound_field_gradient_mono(x,y,z,x0,'ps', driven_signal, f, field_f, theta)
        #print("Pw", Pw)
        #print("Px", Px)
        #print("Py", Py)
        #print("Pz", Pz)
        #print("xx", xx)
        #print("yy", yy)
        #print("zz", zz)                                
        #spatial aliasing frequency must be obtained from driven_signal with virtual sources 
        #to reduce virtual soundfield and not soundfield
        #fx = spatial_aliasing_frequency(Px, xs, Rl, Rc, xl, xc, M, N, grid)
        #fy = spatial_aliasing_frequency(Py, xs, Rl, Rc, xl, xc, M, N, grid)
        #fy = spatial_aliasing_frequency(Pz, xs, Rl, Rc, xl, xc, M, N, grid)
        #print("Fx", Px)
        #print("Fy", Py)
        #print("Fz", Pz)
        #Constrain difference between frequencies to have a difference of less than 4400 Hz
        #assert (abs(new_f - f) <= 4400)
        #cllX = perception_CLL(48000, Px, f)  
        fs = 48000
        #itdX, ildX, icX = perception_ITDILDIC(Px, fs, rf, f)     
        #cllY = perception_CLL(48000, Py, f)  
        #itdY, ildY, icY = perception_ITDILDIC(Py, fs, rf, f)    
        #cllZ = perception_CLL(48000, Pz, f)  
        #itdZ, ildZ, icZ = perception_ITDILDIC(Pz, fs, rf, f)    
        #print("ITD X", itdX, "ILD X", ildX, "IC X", icX)
        #print("ITD Y", itdY, "ILD Y", ildY, "IC Y", icY)
        #print("ITD Z", itdZ, "ILD Z", ildZ, "IC Z", icZ)   
        #print("CLL X", cllX)
        #print("CLL Y", cllY)
        #print("CLL Z", cllZ)        
        return np.nan_to_num(Pw, nan=10e-5, posinf=10e-5, neginf=10e-5), np.nan_to_num(Px, nan=10e-5, posinf=10e-5, neginf=10e-5), np.nan_to_num(Py, nan=10e-5, posinf=10e-5, neginf=10e-5), np.nan_to_num(Pz, nan=10e-5, posinf=10e-5, neginf=10e-5), np.nan_to_num(xx, nan=10e-5, posinf=10e-5, neginf=10e-5), np.nan_to_num(yy, nan=10e-5, posinf=10e-5, neginf=10e-5), np.nan_to_num(zz, nan=10e-5, posinf=10e-5, neginf=10e-5)           
    except Exception as e:
        #o[pin:pin+2001]+= np.hstack(np.median(p[:2000],axis=1))[:2000]
        logger.exception(e)
        print("ERROR IN RUN, RETURNING ZERO")
        return 0

def magnus_effect_scaling(m_scalingFactors, tau):
  return (m_scalingFactors[1] - m_scalingFactors[0]) * tau+ m_scalingFactors[0];

def time_of_arrival(volume, total_surface_area, absorption_coef):
  return 0.161*(volume/(-total_surface_area*np.log10*(1-absorption_coef)))

def schroeder_frequency(volume, total_surface_area, absorption_coef):
  k = 2000
  return k * (np.sqrt(time_of_arrival/volume))

def BandPass(sig, fc, fb):
    """
    This function computes b and a coeffiecents from the TML (torso, throat, mouth, lips)
    values
    fc: frequency cutoff in Hz
    fb: frequency bandwidth in Hz
    """
    fs = 48000
    octave_rollof = -12  
    # Adjusting center frequency and bandwidth for octave rolloff
    fc_adjusted = fc / (2 ** (1 / (2 * octave_rolloff)))
    fb_adjusted = fb / (2 ** (1 / (2 * octave_rolloff)))
    c_adjusted = np.tan(np.pi * fb_adjusted / fs) - 1 / np.tan(np.pi * fb_adjusted / fs)
    b = np.array([1 + c_adjusted / 2, 0, (1 + c_adjusted) / 2])
    a = np.array([1, d * (1 - c_adjusted), -c_adjusted])
    return lfilter(b,a,sig) * -1

def compute_ba_TML_filter(impedance, velocity):
    """
    This function computes b and a coeffiecents from the TML (torso, throat, mouth, lips)
    values
    """
    capacitance = 1 / (velocity**2 * impedance)
    a = np.array([1,(-2*impedance*capacitance),-1])
    b = np.array([(1 / -2*impedance*capacitance+1),(1/(2*impedance*capacitance+1)),0])
    return b, a

def wfs_from_audio(filename, xsi=None,xc=None):
    interval = int(0.1 * 1000) # sec
    #s = 3
    #Window size == Number of speakers
    #N = 3900 what humans would need to localize everything correctly
    N = 46  # number of loudspeakers 575 N = 46 #Greater radius, lower spacing
    grid_size = 307
    endpoint = True
    nframes = 10000
    R = N*2  # keeping radius not so big not so small as possible to avoid stochastic resampling
    x = [np.max(xsi) +11.5, (R *2)+.5]
    y = [np.max(xsi) +11.5, (R *2)+.5]
    yc = np.max(y)/2
    z = [np.max(xsi) +11.5, (R *2)+.5]
    #grid shape == N
    grid, vertices, faces = xyz_grid(x, y, z, N=3, spacing = 4, endpoint = False);  #spacing = .09
    ideal_size = grid[0].shape[1]
    frames = int(nframes * ideal_size/N)
    d = filename[:N*frames]    
    d = np.vstack(list(d for i in range(N))).T
    free_rs = np.linalg.norm((0,0,0))   # distance from origin
    # Focused source
    #Focused source must be near to HRTF function to produce good MAA
    xs = np.zeros((2,grid_size))
    xs[0] += xsi[0]
    xs[1] += xsi[1]
    #xs[2] += 0
    #Set a sound source that represents torso 
    rf = np.linalg.norm(xs)  # distance from origin, ideally less than 0.00004
    #print("DISTANCE FROM ORIGIN", rf)
    #angle in rad, ideally azimuth from microphone
    #degA = correlation(d[:,0],d[:,1],rf,48000)
    #degB = correlation(d[:,1],d[:,2],rf,48000)
    #print("IPD A",degA, "IPD B", degB)
    # Plane wave
    degA = -14
    npw = sfs.util.direction_vector(np.radians(degA))
    nf = sfs.util.direction_vector(np.radians(degA))  # normal vector
    #Free direction
    free_nf = sfs.util.direction_vector(np.radians(0))  # normal vector
    tf = rf / sfs.default.c  # time-of-arrival (direct sound reverb) at origin

    # Impulsive excitation
    fs = 48000
    # Circular loudspeaker array
    M = int(N/2);
    # Point source
    rs = np.linalg.norm(xs)  # distance from origin
    ts = rs / sfs.default.c  # time-of-arrival (direct sound reverb) at origin of the point source
    free_ts = rs / sfs.default.c  # time-of-arrival (direct sound reverb) at origin of the point source without position and orientation
    #get azimuth from direction    
    #keeeping good centering to create perceivable colouration
    #define center position from halfwave-length constrained M
    xc = np.zeros(grid_size)
    #xc[0] = M
    xc += 0.04791666666666667 #m as M/constant
    #Compute an array object with positions and orientations of the sources
    #x0 is weighted in the end
    #x0 has a dependent virtual source
    x0, alpha, xl, weights = circular(N, R, xc) #3,2
    x0[:,6] += weights
    x0_size = x0.size
    Rc = R+xc
    Rl = ((R + xl)*0.08)[:N,3] #.08
    theta = 2 * np.pi * R
    #print("GRID SHAPE", np.shape(grid[0]) )      
    #print("XL", xl)      
    #print("D",d)
    f, cx0, velocity, wall_admittance, impedance, a, b, n_extended_area, source_center_dist, dcx0, field_f, modes, C, DN, E, junctions, field_f = correct_synthesis_frequency(d,xs,Rl,Rc,xl, xc, M, R, x0_size,N, grid, vertices, faces)
    try:
        new_f, n_aliasing = spatial_aliasing_frequency(d, xs, Rl, Rc, xl, xc, M, N, grid)    
        #print("ALIASING FREQUENCY", new_f)
    except Exception as e:
        new_f, n_aliasing = np.inf, np.inf
    #The other frequencies can be brain activity related or other mechanical resonances
    field_f = np.array(field_f)[np.array(field_f) > 20]
    print("VOCAL TRACT FREQUENCY LOBES", field_f)
    freq_steps = int(4400 / 48000 * 2048) #optimal_frequencies_difference / sample_rate * N
    #Method to compute full signal filtering by torso and lips
    #of the acoustic field
    tff = compute_transfer_function(n_aliasing, cx0, [f,new_f] , impedance, 2, freq_steps, b, a, Rc, yc, source_center_dist, velocity, wall_admittance, impedance, (R + xl[0])*0.08, dcx0, field_f, grid, n_extended_area, xl, vertices, faces, modes, C, DN, E, junctions) #naliasing, cx0, tf_f, m_plane_mode_input_impedance, nf, freq_steps
    b, a = tff[0], tff[1]
    print("B", b, "A", a)
    b[0] = b[0] / np.max(np.abs(b))
    b[-1] = b[-1] / np.max(np.abs(b))
    a[0] = a[0] / np.max(np.abs(a))
    a[-1] = a[-1] / np.max(np.abs(a))
    #Create a representation of the free field
    #Spacing must be equal to synthesis tapering
    #print("Fl Frequency", f)
    #print("Spatial aliasing Frequency", new_f) 
    spacing=0.02
    #print(grid[0].shape)
    #Compute the virtual source selection, the secondary source points, delays and weights 
    #for the free point source with targets 
    #print("NF DIRECTION VECTOR", nf)
    NF = np.zeros(grid_size)
    xref = np.zeros(grid_size)
    xref[0] += 180
    NF[0] = nf[0]
    NF[1] = nf[1]
    NF[2] = nf[2]
    nf = NF
    delays = delay_focused_25d(x0, xl , xs, nf,xref=xref)
    #for lfws-vss wegiths == x[:,6]
    pin = 0
    parallel_Ds = []
    for i in range(frames):
        if pin +N > len(d):
            break
        else:    
            parallel_Ds.append(d[pin:pin+N])
        pin += int(N/2)    
    #print("DELAYS", delays)
    parallel_delays = np.array([delays for i in range(frames)])     
    parallel_xs = np.array([x for i in range(frames)])
    parallel_xss = np.array([xs for i in range(frames)])
    parallel_ts = np.array([ts for i in range(frames)])
    parallel_ps = np.array([p for i in range(frames)])
    parallel_Ns = np.array([N for i in range(frames)])
    parallel_rfs = np.array([rf for i in range(frames)])
    parallel_ys = np.array([y for i in range(frames)])
    parallel_zs = np.array([z for i in range(frames)])    
    parallel_x0s = np.array([x0 for i in range(frames)])
    parallel_fs = np.array([f for i in range(frames)])
    parallel_rls = np.array([Rl for i in range(frames)])
    parallel_rcs = np.array([Rc for i in range(frames)])    
    parallel_xls = np.array([xl for i in range(frames)])
    parallel_xcs = np.array([xc for i in range(frames)])
    parallel_ms = np.array([M for i in range(frames)])
    parallel_is = np.array([i for i in range(frames)])    
    parallel_field_fs = np.array([field_f for i in range(frames)])  
    parallel_thetas = np.array([theta for i in range(frames)])  
    #pool = ProcessPool(nodes=2)
    #all_ps = pool.amap(synthesize_frames, parallel_Ds, parallel_delays, parallel_weights, parallel_selections, parallel_xs, parallel_ns, parallel_as, parallel_secondary_sources, parallel_grids, parallel_ts, parallel_ps, parallel_Ns, parallel_rfs, parallel_is  )
    #while not all_ps.ready():
    #    pass    
    o = []
    #frames = 7
    for i in range(frames):
        try:
            o.append(synthesize_frames(parallel_Ds[i], parallel_delays[i],  
            parallel_xs[i], parallel_ts[i], parallel_ps[i], parallel_Ns[i], 
            parallel_rfs[i], parallel_ys[i], parallel_zs[i], parallel_x0s[i], parallel_fs[i], parallel_rls[i],
            parallel_rcs[i], parallel_xls[i], parallel_xcs[i], parallel_xss[i], parallel_ms, parallel_is[i],  
            parallel_field_fs[i], parallel_thetas[i]))
        except Exception as e:        
            logger.exception(e)
            break
        #o = all_ps.get()
    pin = 0
    w_x = np.zeros((int(N)*frames))
    lr_x = np.zeros((int(N)*frames))
    height_x = np.zeros((int(N)*frames))
    deep_x = np.zeros((int(N)*frames))
    #print("OUTPUT ARRAY", o[0][0])
    for i in range(len(o)):
        try:  
            #print("OUTPUT",i,  o[i][0])
            #print("OUTPUT",i,  o[i][1])
            #print("OUTPUT",i,  o[i][2])
            #print("OUTPUT",i,  o[i][3])
            w_x[pin:pin+N] += o[i][0]
            lr_x[pin:pin+N] += o[i][1]
            height_x[pin:pin+N] += o[i][2]
            deep_x[pin:pin+N] += o[i][3]           
            pin += int(M)
        except Exception as e:
            logger.exception(e)
            break
    wmax = max(np.abs(np.min(w_x)), np.max(w_x))
    lrmax = max(np.abs(np.min(lr_x)), np.max(lr_x))
    heightmax = max(np.abs(np.min(height_x)), np.max(height_x))
    deepmax = max(np.abs(np.min(deep_x)), np.max(deep_x))          
    #w_tml = np.nan_to_num(np.sum(np.array([BandPass(w_x, field_f[i], field_f[i] - field_f[i-1]) for i in range(1,len(field_f))]), axis=0), nan=10e-5, posinf=10e-5, neginf=10e-5)
    #lr_tml = np.nan_to_num(np.sum(np.array([BandPass(lr_x, field_f[i], field_f[i] - field_f[i-1]) for i in range(1,len(field_f))]), axis=0), nan=10e-5, posinf=10e-5, neginf=10e-5)
    #height_tml = np.nan_to_num(np.sum(np.array([BandPass(height_x, field_f[i], field_f[i] - field_f[i-1]) for i in range(1,len(field_f))]), axis=0), nan=10e-5, posinf=10e-5, neginf=10e-5)
    #deep_tml = np.nan_to_num(np.sum(np.array([BandPass(deep_x, field_f[i], field_f[i] - field_f[i-1]) for i in range(1,len(field_f))]), axis=0), nan=10e-5, posinf=10e-5, neginf=10e-5)                  
    octave_rolloff = -12
    rolloff_factor = 2 ** (-1 / (2 * octave_rolloff))
    b *= rolloff_factor
    a *= rolloff_factor
    w_filter_tml = np.nan_to_num(lfilter(b,a,w_x), nan=10e-5, posinf=10e-5, neginf=10e-5)
    lr_filter_tml = np.nan_to_num(lfilter(b,a,lr_x), nan=10e-5, posinf=10e-5, neginf=10e-5)
    height_filter_tml = np.nan_to_num(lfilter(b,a,height_x), nan=10e-5, posinf=10e-5, neginf=10e-5)
    deep_filter_tml = np.nan_to_num(lfilter(b,a,deep_x), nan=10e-5, posinf=10e-5, neginf=10e-5)                  
    wtml_filter_max = max(abs(np.min(w_filter_tml)), np.max(w_filter_tml))
    lrtml_filter_max = max(abs(np.min(lr_filter_tml)), np.max(lr_filter_tml))
    heighttml_filter_max = max(abs(np.min(height_filter_tml)), np.max(height_filter_tml))
    deeptml_filter_max = max(abs(np.min(deep_filter_tml)), np.max(deep_filter_tml))
    #wtmlmax = max(abs(np.min(w_tml)), np.max(w_tml))
    #lrtmlmax = max(abs(np.min(lr_tml)), np.max(lr_tml))
    #heighttmlmax = max(abs(np.min(height_tml)), np.max(height_tml))
    #deeptmlmax = max(abs(np.min(deep_tml)), np.max(deep_tml))
    w_x = (np.nan_to_num(w_x, nan=10e-5, posinf=10e-5, neginf=10e-5)/wmax) + (w_filter_tml/wtml_filter_max)
    lr_x =  (np.nan_to_num(lr_x, nan=10e-5, posinf=10e-5, neginf=10e-5)/lrmax) + (lr_filter_tml/lrtml_filter_max)
    height_x = (np.nan_to_num(height_x, nan=10e-5, posinf=10e-5, neginf=10e-5)/heightmax) + (height_filter_tml/heighttml_filter_max)
    deep_x = (np.nan_to_num(deep_x, nan=10e-5, posinf=10e-5, neginf=10e-5)/deepmax) + (deep_filter_tml/deeptml_filter_max)
    return w_x, lr_x, height_x, deep_x

def WFS_filter(x):
    song = MIR(x,48000)
    rolloffs = []
    hfc = []
    for frame in song.FrameGenerator():
        song.window()
        song.Spectrum(fft=True)
        Wn = song.nyquist(roll_off(song.magnitude_spectrum, 48000))
        #print("RollOff in Nyquist",Wn, "In Hz", roll_off(song.magnitude_spectrum, 48000))
        rolloffs.append(song.HFC())
    rolloff = np.mean(rolloffs)    
    #Bug: the ideal_order in the original paper is int(x.size/2),
    #but it returns overflow error
    ideal_order = 1479 #2559 without overflow
    lf_noise_filter = song.IIR( x, rolloff, 'highpass', ideal_order) # ideal_order or 3963530*6 
    hf_noise_filter = song.IIR( x, 20000, 'lowpass', ideal_order) # ideal_order or 3963530*6
    return hf_noise_filter    

def homogeneous_transformation_matrix(yaw, pitch, roll, torso_rotation):
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    torso_rotation = np.radians(torso_rotation)

    # Rotation matrices
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])

    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

    R_torso = np.array([[np.cos(torso_rotation), -np.sin(torso_rotation), 0],
                        [np.sin(torso_rotation), np.cos(torso_rotation), 0],
                        [0, 0, 1]])

    # Combine rotations
    M_combined = np.dot(R_yaw, np.dot(R_pitch, np.dot(R_roll, R_torso)))

    return M_combined
def binaural_impulse_lwfs(f,o):
    Rs = []
    N = 46
    angles = []
    R = np.radians(N) 
    # Head and pinna
    #i == yaw
    for i in np.arange(0, 160, 20):
        #pitch
        for pitch in np.arange(-90, 91, 45):
            for roll in np.arange(-90, 91, 45):
                for torso_rotation in np.arange(-90, 90, 18):
                    M_combined = homogeneous_transformation_matrix(yaw/R, pitch/R, roll/R, torso_rotation/R)
                    transformed_point = np.dot(M_combined, np.array([0, 0, N]))
                    Rs.append(transformed_point*N)
                    angles.append([yaw, pitch, roll, torso_rotation])
    #print("RS", angles)
    m = 0        
    for Ri in Rs:
        #Apply a xs for torso 
        #Apply xc for torso	
        xs = Ri  	
        w_path = o+'binaural_wfs_bandondeon_46speakers_yaw'+str(angles[m][0])+'_pitch'+str(angles[m][1])+'_roll'+str(angles[m][2])+'_torso'+str(angles[m][3])+'_mask:w_propacoord:wave_field'
        lr_path = o+'binaural_wfs_bandondeon_46speakers_yaw'+str(angles[m][0])+'_pitch'+str(angles[m][1])+'_roll'+str(angles[m][2])+'_torso'+str(angles[m][3])+'_mask:lr_propacoord:wave_field'
        height_path = o+'binaural_wfs_bandondeon_46speakers_yaw'+str(angles[m][0])+'_pitch'+str(angles[m][1])+'_roll'+str(angles[m][2])+'_torso'+str(angles[m][3])+'_mask:height_propacoord:wave_field'
        deep_path = o+'binaural_wfs_bandondeon_46speakers_yaw'+str(angles[m][0])+'_pitch'+str(angles[m][1])+'_roll'+str(angles[m][2])+'_torso'+str(angles[m][3])+'_mask:deep_propacoord:wave_field'
        if os.path.exists(w_path+'.ogg'):
            m += 1
            lwfs(w_path, 'with_lwfs_vss/'+w_path, Ri)
            continue
        if os.path.exists(lr_path+'.ogg'):
            m += 1
            lwfs(lr_path, 'with_lwfs_vss/'+lr_path, Ri)
            continue
        elif os.path.exists(height_path+'.ogg'):
            m += 1
            lwfs(height_path, 'with_lwfs_vss/'+height_path, Ri)
            continue
        elif os.path.exists(deep_path+'.ogg'):
            m += 1
            lwfs(deep_path, 'with_lwfs_vss/'+deep_path, Ri)   
            continue
        #continue by HFC or ITD/ILD/IPD cues    
        w_binaural, lr_binaural, height_binaural, deep_binaural = bioestimulated_signal(f, xs)
        #print("SYNTHESIZED LR AUDIO", lr_audio)
        #print("SYNTHESIZED HEIGHT AUDIO", height_audio)
        #print("SYNTHESIZED DEEP AUDIO", deep_audio)    
        #filtered_lr_soundfield = WFS_filter(lr_audio)
        #filtered_height_soundfield = WFS_filter(height_audio)
        #filtered_deep_soundfield = WFS_filter(deep_audio)    
        write_file(w_path,48000, w_binaural/np.max(w_binaural))
        write_file(lr_path,48000, lr_binaural/np.max(lr_binaural))
        write_file(height_path,48000, height_binaural/np.max(height_binaural))
        write_file(deep_path,48000,deep_binaural/np.max(deep_binaural))
        lwfs(w_path, 'with_lwfs_vss/'+w_path, Ri)
        lwfs(lr_path, 'with_lwfs_vss/'+lr_path, Ri)
        lwfs(height_path, 'with_lwfs_vss/'+height_path, Ri)
        lwfs(deep_path, 'with_lwfs_vss/'+deep_path, Ri)                
        m += 1        

def solve_wave_problem_noise_src(tract, need_to_extract_matrix_f, freq):
    
    last_sec = len(cx0) - 1

    if need_to_extract_matrix_f:
         F = cx0[tract.idx_sec_noise_source].get_matrix_f()[0]
         need_to_extract_matrix_f = False

    # Propagate impedance and admittance
    imped = propagate_imped_admit(f)

    # Propagate velocity and pressure
    axial_velocity, velocity, prev_pressure, admittance = propagate_velocity_pressure(f, select_aliasing + 1, last_sec)
    return velocity


def ctr_line_pt_out(x0i, normal_in, circle_arc_angle, r, length):
    if length > 0:
        x0i = np.array(x0i)
        N = np.array(normal_in)

        # Calcular el ángulo de rotación según el ángulo del arco de Rc
        theta = abs(circle_arc_angle) / 2 if abs(circle_arc_angle) >= .5 else np.pi / 4

        # Determinar la dirección de rotación
        if np.any(r * circle_arc_angle < 0) != np.any(r < 0):
            theta = np.pi / 2 - theta

        # Calcular la matriz de transformación de rotación
        rotate = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])

        # Calcular la matriz de transformación de traslación
        translate = length * np.dot(rotate, N)

        # Aplicar la traslación al punto Pt
        transformation = x0i + translate

        return transformation
    else:
        return ctr_line_pt_in

def move_point_from_exit_landmark_to_geo_landmark(pt, cx0, Rc, xc, yc, a, b, source_center_dist):
    pt_vert_plane = np.array([pt[0], pt[2]])
    end_normal = np.array(np.linalg.norm(cx0[-1]))
    vertical = np.array([0., 1.])
    circle_arc_angle = np.linalg.norm(cx0[-1]) / np.linalg.norm(source_center_dist)**3

    angle = np.arctan2(end_normal, end_normal) - np.arctan2(vertical[1], vertical[0]) + 2. * np.pi
    angle %= 2. * np.pi

    if np.any(abs(circle_arc_angle) > xc):
        if np.sign((1/cx0[-1]) * circle_arc_angle) < 0:
            angle -= np.pi
            rotation_matrix = np.array([[np.sin(angle), np.cos(angle)], [-np.cos(angle), np.sin(angle)]])
            pt_vert_plane = np.dot(rotation_matrix, np.array([pt_vert_plane[0], -pt_vert_plane[1]]))
        else:
            rotation_matrix = np.array([[np.sin(angle), np.cos(angle)], [-np.cos(angle), np.sin(angle)]])
            pt_vert_plane = np.dot(rotation_matrix, pt_vert_plane)
    else:
        ctl_vec = np.array(ctr_line_pt_out(cx0[-1])) - np.array(cx0[-1][:2])
        angle_ctl_norm = np.arctan2(ctl_vec[1], ctl_vec[0]) - np.arctan2(end_normal, end_normal) + 2. * np.pi
        angle_ctl_norm %= 2. * np.pi
        pt_vec = pt_vert_plane - np.array(ctr_line_pt_out(cx0[-1]))
        angle_pt_norm = np.arctan2(pt_vec[1], pt_vec[0]) - np.arctan2(end_normal, end_normal) + 2. * np.pi
        angle_pt_norm %= 2. * np.pi
        

        if not np.sign((angle_ctl_norm - np.pi) * angle_pt_norm) < 0:
            angle -= np.pi
            rotation_matrix = np.array([[np.sin(angle), np.cos(angle)], [-np.cos(angle), np.sin(angle)]])
            pt_vert_plane = np.dot(rotation_matrix, np.array([pt_vert_plane[0], -pt_vert_plane[1]]))
        else:
            rotation_matrix = np.array([[np.sin(angle), np.cos(angle)], [-np.cos(angle), np.sin(angle)]])
            pt_vert_plane = np.dot(rotation_matrix, pt_vert_plane)
            
    contour_out = ctr_line_pt_out(cx0[-1], np.array([0, np.linalg.norm(cx0[-1])]), circle_arc_angle, Rc, end_normal)
    
    return np.array([
        pt_vert_plane[0] + contour_out[0] ,
        pt[1],
        pt_vert_plane[1] + contour_out[1]
    ]) 


from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import griddata

def norm_out(circle_arc_angle):
    theta = 2 * np.pi * circle_arc_angle
    return np.array([[np.cos(np.cos(theta)), -np.sin(np.sin(theta))],
                           [np.sin(np.sin(theta)), np.cos(np.cos(theta))]])  

def compute_junction_matrices(modes, segIdx, next_sec, cx0, Rl):
        #Write glottis junction matrix
        matrixF = []
        #print("CONTOUR", cx0)
        #print("SEG IDX", segIdx)
        intersections = []
        scales_in = []
        scales_out = []
        nSecs = []
        ctlShifts = []
        for ns in range(2):
            nextSec = ns
            ctlShift = [cx0[nextSec][0] ,cx0[segIdx-1][-1]]
            #print("CTL SHIFT", ctlShift)
            #print("ROTATED  CX0", norm_out(cx0[segIdx-1])[-1].T)
            ctlShifts.append(ctlShift * norm_out(cx0[segIdx-1])[-1].T[-1])
            #Append twice to equivalent the size of interpolations
            nSecs.append(nextSec)
            nSecs.append(nextSec)
            nModes = modes
            nModesNext = modes
            F = np.zeros((nModes, nModesNext))

            # Get current cross-section contour
            contour = cx0[segIdx-1]
            scaling_out = cx0[segIdx-1]/1
            contour_scaled = contour * scaling_out

            # Get next cross-section contour
            nextContour = cx0[nextSec]
            scaling_in = cx0[nextSec]/1
            nextContour_scaled = nextContour * scaling_in
            
            #Copy scales per interpolation
            scales_in.append(scaling_in)
            scales_out.append(scaling_out)
            scales_in.append(scaling_in)
            scales_out.append(scaling_out)
            # Compute intersection of contours
            intersections.append(contour_scaled)
            intersections.append(nextContour_scaled)
        intersections = np.array(intersections)
        epsilon = 1e-6  # Puedes ajustar este valor según sea necesario
        intersections[2:] += epsilon * np.random.rand(2, 3)
        #print("INTERSECTIONS", intersections)
        cdt = Delaunay(intersections)
        # Mesh intersection contours and generate integration points
        interpolations = []
        count = 0
        #print("THROAT VERTICES", cdt.vertices)
        m_spacing = 4
        for K in range(len(intersections)):
            mesh_pts = intersections[cdt.vertices]
            # Interpolate modes
            #pts, m_points, m_modes, m_modesNumber, m_contour, m_spacing
            interpolation1 = interpolate_modes(mesh_pts, intersections, scales_out, modes, cdt, m_spacing, Rl)
            interpolation2 = interpolate_modes(mesh_pts, intersections, scales_in, modes, cdt, m_spacing, Rl)
            interpolations.append([interpolation1, interpolation2])
            count += 1
            
        for j in range(len(interpolations)):
            # Compute scattering matrix F
            for m in range(nModes):
                for n in range(nModesNext):
                    for i in range(len(cdt.simplices)):
                        try:
                            simplex = cdt.simplices[i]
                            #print("SIMPLEX", simplex)
                            #print("TRANSLATED SIMPLEX", cdt.transform[0][simplex[:-1], :-1][:2] )
                            area = np.abs(np.linalg.det(cdt.transform[0][simplex[:-1], :-1][:2]))
                            quadPtWeight = 1.0 / 3.0
                            #print("INTERPOLATIONS", interpolations[j] )
                            #print("I", i )
                            #print("M", m )   
                            #print("FMN", F[m] )   
                            f_value = area * np.nan_to_num(interpolations[j][i][m]) * np.nan_to_num(interpolations[j][i][n]) * quadPtWeight \
                                       / scales_out[j] / scales_in[j]                     
                            #print("F VALUE", f_value)  
                            F += f_value
                        except Exception as e:
                            continue

            matrixF.append(F)
        return np.array(matrixF)

def distance_polygon(poli, ptToTest):
    NON_SENS_VALUE = np.inf  # Valor no sensible, puedes ajustarlo según sea necesario
    distEdge = NON_SENS_VALUE
    
    # Loop sobre los bordes del contorno
    edges = []
    for edge in poli.simplices:
        # Definir el vector correspondiente al borde
        edges = edge
    vecEdge = np.array([edges[1] - edges[0], edges[1] - edges[1]])
        
    # Definir el vector que conecta el borde al punto de prueba
    vecEdgeVert = np.array([edges[0] - ptToTest[0], edges[1] - ptToTest[1]])
        
    # Calcular el producto escalar entre el vector del borde y el vector del borde al vértice del contorno
    scalarProd = np.dot(vecEdge, vecEdgeVert)
        
    # Normalizar el producto escalar con la norma del vector del borde
    scalarProd = -scalarProd / np.linalg.norm(vecEdge) ** 2
        
    # Verificar si la proyección del vértice está en el segmento del contorno
    #print("SCALAR PRODUCT", scalarProd)
    scalarProd = max(0, np.min((np.min(scalarProd), 1)))
        
    # Calcular el vector que une la proyección con el vértice
    distVec = []
    for vedge in vecEdge:
        distVec.append(vedge * scalarProd + vecEdgeVert)
    distVec = np.array(distVec)   
    # Calcular la distancia mínima entre el vértice y el vector
    distEdge = np.min((distEdge, np.linalg.norm(distVec)))
    
    # Aplicar la corrección según si el punto está dentro o fuera del contorno
    if np.any(ptToTest[:2] < poli.points[:2]):
        return distEdge
    else:
        return -distEdge
  
def bring_back_point_inside_contour(poli, pt, spacing, MINIMAL_DISTANCE):
    MINIMAL_DISTANCE = 1e-6  # Puedes ajustar este valor según sea necesario
    deltaXGrad = np.sqrt(MINIMAL_DISTANCE) * spacing
    distCont = distance_polygon(poli, pt)
    
    # Calcular gradientes parciales
    grad_x = (distance_polygon(poli, np.array([pt[0] + deltaXGrad, pt[1]])) - distCont) / deltaXGrad
    grad_y = (distance_polygon(poli, np.array([pt[0], pt[1] + deltaXGrad])) - distCont) / deltaXGrad
    
    # Calcular gradiente
    gradVec = np.array([grad_x, grad_y])
    #print("PT", pt[:2][:,2])
    #print("G", gradVec)    
    #print("DISTANCE FROM CONTOUR", MINIMAL_DISTANCE + distCont)   
    # Actualizar el punto
    return pt[:2][:,2] - ((MINIMAL_DISTANCE + distCont) * gradVec)

# Función de distancia a un polígono (distance_polygon) no proporcionada en el código original.
# Deberías implementar esta función o encontrar una implementación adecuada en una biblioteca como SciPy.

# Puedes llamar a esta función con tus datos de entrada:
# pt = bring_back_point_inside_contour(poli, pt, spacing)

def interpolate_modes(pts, m_points, m_modes, m_modesNumber, m_contour, m_spacing, Rl):
    numPts = len(pts)
    interpolation = np.zeros((numPts, m_modesNumber))
    points = []
    values = []
    tempValues = {}
    T = Delaunay(m_points)
    coords = []
    norm = 0
    ptNotFound = False

    # insert triangulation points
    numTriPts = len(m_points)
    for i in range(numTriPts):
        points.append([m_points[i][0], m_points[i][1]])

    # create the point values maps
    for m in range(m_modesNumber):
        tempValues.clear()
        for i in range(numTriPts):
            tempValues[tuple(points[i])] = m_modes[i][m]
        values.append(tempValues)

    # insert points to interpolate
    for i in range(numPts):
        #print("CNT", m_contour)
        #print("PT", pts[i])
        #Points lie outside of the contour
        if np.any(np.min(m_contour.points) < pts[i]):
            pt_back = bring_back_point_inside_contour(m_contour, pts[i], m_spacing, Rl)[:2]
            #print("PTS", pts[i])
            #print("PT BACK", pt_back)
            pts[i][:2][:,2] = pt_back
            

    # interpolate the field
    for i in range(numPts):
        #Points lie outside of the contour
        if np.any(np.min(m_contour.points) < pts[i]):
            #print("Error interpolating: point outside of the contour")
            interpolation[i, :] = np.nan
            ptNotFound = True
        else:
            coords.clear()
            coords = T.find_simplex(pts[i])
            norm = 1.0  # placeholder for norm calculation, as this is not directly available in the provided code
            for m in range(m_modesNumber):
                l_value = linear_interpolation(coords, norm, values[m])
                interpolation[i, m] = l_value

    return interpolation

# Función de interpolación lineal (linear_interpolation) no proporcionada en el código original.
# Deberías implementar esta función o encontrar una implementación adecuada en una biblioteca como SciPy.

# La función bring_back_point_inside_contour (traer_punto_de_vuelta_dentro_del_contorno) también
# no está definida en el código proporcionado, deberías implementarla según la lógica que
# se describe en los comentarios del código original.

# Puedes llamar a esta función con tus datos de entrada:
# interpolation = interpolate_modes(pts, m_points, m_modes, m_modesNumber, m_contour, m_spacing)

from scipy.signal import TransferFunction
from scipy.linalg import eigh

def compute_modes(m_mesh, m_points, m_contour, f, vertices, faces):
    # Declarar variables
    numVert = 9
    numTri = 16
    numSurf = 1

    # Inicializar matrices
    mass = np.zeros((numTri, numTri))
    massY = np.zeros((numTri, numTri))
    stiffness = np.zeros((numTri, numTri))
    stiffnessY = np.zeros((numTri, numTri))
    B = np.zeros((numTri, numTri))
    
    # Inicializar otras variables
    signFirstMode = 0
    m_eigenFreqs = []
    m_maxAmplitude = []
    m_minAmplitude = []
    m_modes = []
    m_DR = []
    m_KR2 = []
    m_surfIdxList = []
    dSdr = [-1., 1., 0. ]
    dSds = [-1., 0., 1. ]
    J = np.zeros((2,2))
    quadPtWeight = 1/3
    quadPtCoord = [[1. / 6., 1. / 6.], [2. / 3., 1. / 6.], [1. / 6., 2. / 3.]]
    S = np.zeros((3,3))
    for i in range(3):
        S[i][0] = 1. - quadPtCoord[i][0] - quadPtCoord[i][1]
        S[i][1] = quadPtCoord[i][0]
        S[i][2] = quadPtCoord[i][1]    
    for p in range(3):
      #print("FACE", faces[p][0]) 
      #print("DSDR", dSdr[p])
      J[0][0] += np.array(faces[p][0]) * np.array(dSdr[p])
      J[0][1] += np.array(faces[p][1]) * np.array(dSdr[p]);
      J[1][0] += np.array(faces[p][0]) * np.array(dSds[p]);
      J[1][1] += np.array(faces[p][1]) * np.array(dSds[p]);
    detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    #print("QUADPTWEIGHT", quadPtWeight)
    #print("DETJ", detJ)
    quadPtWeightDetJ = quadPtWeight * detJ / 2.;

    # Bucle sobre las caras
    for i in range(numVert):
        it = faces[i]
        #print("FACE", it[0])
        faceArea = 0.5 * abs(it[0] *
                             (it[1] - it[2])
                             + it[1] *
                             (it[2] - it[0])
                             + it[2] *
                             (it[0] - it[1])) + 1
        #print("FACE AREA", faceArea)                     
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        quadPtWeightDetJ = quadPtWeight * detJ / 2.
        
        dSdx = []
        dSdy = []

        # Cálculo de las derivadas parciales dS/dx y dS/dy
        for p in range(3):
            dSdx.append((J[1, 1] * dSdr[p] - J[0, 1] * dSds[p]) / detJ)
            dSdy.append((J[0, 0] * dSds[p] - J[1, 0] * dSdr[p]) / detJ)
        #print("DSDX", dSdx)    
        #print("DSDY", dSdy)    
        # Cálculo de las coordenadas x y y de los puntos de cuadratura
        Xrs = np.zeros(3)
        Yrs = np.zeros(3)
        for q in range(3):
            for p in range(3):
                Xrs[q] += it[p] * S[q, p]
                Yrs[q] += it[p] * S[q, p]

        # Bucle sobre los puntos de la cara
        for j in range(3):
            for k in range(3):
                idxM = it[j]
                idxN = it[k]-1

                # Matriz de masa
                mass[idxM, idxN] += (1. + int(j == k)) * faceArea / 12

                # Bucle sobre los puntos de cuadratura
                for q in range(3):
                    # Matriz necesaria para construir la matriz C (casi idéntica a la matriz de masa)
                    #print("Yrs",Yrs[q]) 
                    #print("S", S[q, j])
                    #print("S", S[q, k]) 
                    #print("QUADPTWEIGHTDETJ", quadPtWeightDetJ)
                    #print("MASSY", massY[idxM, idxN])
                    #print("FACE M", idxM, "FACE N", idxN)                    

                    massY[idxM, idxN] += Yrs[q] * S[q, j] * S[q, k] * quadPtWeightDetJ

                    # Matriz necesaria para construir la matriz D (casi idéntica a la matriz de rigidez)
                    stiffnessY[idxM, idxN] += Yrs[q] * (dSdx[j] * dSdx[k] + dSdy[j] * dSdy[k]) * quadPtWeightDetJ

                    # Matriz necesaria para construir la matriz E
                    B[idxM, idxN] += (Xrs[q] * S[q, j] * dSdx[k] + Yrs[q] * S[q, j] * dSdy[k]) * quadPtWeightDetJ

                # Matriz de rigidez
                stiffness[idxM, idxN] += ((faces[(j + 1) % 3][1] - faces[(j + 2) % 3][1] ) *
                                           (faces[(k + 1) % 3][1]  - faces[(k + 2) % 3][1] ) +
                                           (faces[(j + 2) % 3][0]  - faces[(j + 1) % 3][0] ) *
                                           (faces[(k + 2) % 3][0]  - faces[(k + 1) % 3][0] )) / faceArea / 4
                #Constrain stiffness to keep positive
                #if stiffness[idxM, idxN] < 0:
                #    stiffness[idxM, idxN] = 0

    #print("THROAT MASS", np.shape(mass))
    #Solve generalized eigenvalues
    stiffness = (stiffness + stiffness.T) / 1
    mass = (mass + mass.T) / 1
    alpha = 1e-16
    #Declare the symmetric hermitian of the stiffness and mass
    H = stiffness + alpha * mass
    H = (H + H.T) / 1
    stiffness_eigenvalues, stiffness_eigenvectors = np.linalg.eigh(H)

    # Extraer frecuencias propias
    maxCutOnFreq = f
    maxWaveNumber = (2 * np.pi * f / snd_speed) ** 2

    m_eigenFreqs = np.sqrt(stiffness_eigenvalues[stiffness_eigenvalues < maxWaveNumber]) * snd_speed / (2 * np.pi)
    #print("MASS", mass)
    #print("STIFFNESS", stiffness)
    #print("MAXIMUM WAVENUMBER", maxWaveNumber)
    #print("STIFFNESS VALUES", stiffness_eigenvalues)
    m_modesNumber = len(m_eigenFreqs)
    m_C = np.zeros((m_modesNumber, m_modesNumber))
    m_DN = np.zeros((m_modesNumber, m_modesNumber))
    m_E = np.zeros((m_modesNumber, m_modesNumber))
    # Inicializar primeras amplitudes
    signFirstMode = np.sign(stiffness_eigenvectors[:, 0][0])

    # Extraer modos
    m_modes = stiffness_eigenvectors[:, :m_modesNumber] * signFirstMode
    print("AUTOFRECUENCIAS", m_eigenFreqs)
    #print("MODOS", m_modes)
    # Calcular matrices multimodales
    for m in range(m_modesNumber):
        for n in range(m_modesNumber):
            #print("C", m_C[m, n])
            #print("MODE", m_modes[:, m].T)
            #print("MASS MODE PRODUCT", np.dot(massY, m_modes[:, n]))
            m_C[m, n] = np.dot(m_modes[:, m].T, np.dot(massY, m_modes[:, n]))
            m_DN[m, n] = np.dot(m_modes[:, m].T, np.dot(stiffnessY, m_modes[:, n]))
            m_E[m, n] = np.dot(m_modes[:, m].T, np.dot(B, m_modes[:, n]))


    return (np.nan_to_num(m_eigenFreqs), m_modes, m_C, m_DN, m_E)


# Función principal
def compute_transfer_function(naliasing, cx0, tf_f, m_plane_mode_input_impedance, nf, freq_steps, b, a, Rc, yc, source_center_dist, velocity, admittance, impedance, Rl, dcx0, field_f, grid, x0, xl, vertices, faces, modes, C, DN, E, junctions):
    # Precomputaciones
    freq_steps = 48000 / 2.0 / nf
    num_freq_computed = int(np.ceil(tf_f[0] / freq_steps))

    # Activate computation of radiated field since in most of the cases the reception point of the transfer functions is outside
    compute_radiated_field = True

    #print("TF POINTS", b, a )        
    # Compute rotated coordinates of the transfer function points around the expansion center 
    b_exit = move_point_from_exit_landmark_to_geo_landmark(b, Rc, cx0, Rc, yc, a, b, source_center_dist)
    a_exit = move_point_from_exit_landmark_to_geo_landmark(a, Rc, cx0, Rc, yc, a, b, source_center_dist)

    # Resize the frequency vector
    tf_points = [b_exit, a_exit]
    #print("EXIT TF POINTS", tf_points )  
    # Resize the frequency vector
    tf_freqs = []

    # Resize the plane mode input impedance vector
    plane_mode_input_impedance = np.zeros((num_freq_computed, 1), dtype=complex)

    # Inicialización de variables para el seguimiento del tiempo
    time_exp = 0
    # Actually solve the wave problem
    num_sec = len(cx0)
    #Take the arc angle from cx0    
    circle_arc_angle = np.linalg.norm(cx0[-1]) / np.linalg.norm(source_center_dist)**3
    
    # Propagate velocity and pressure
    axial_velocity, velocity, prev_pressure, admittance = propagate_velocity_pressure(velocity, admittance, tf_f[0], 0, 46/48000, 1, circle_arc_angle, Rl, cx0, dcx0, field_f, modes, C, DN, E, junctions)

    # Exporta la impedancia de entrada del modo plano
    impedance = abs(1j * 2. * np.pi * tf_f[0] * volumic_mass * impedance)
    #print("IMPEDANCE", impedance)
    #print("AXIAL VELOCITY", axial_velocity)
    #print("ADMITTANCE", admittance)
    return b_exit, a_exit, impedance, axial_velocity, admittance

def magnus_effect_scaling_derivative(m_circleArcAngle, m_curvatureRadius, m_scalingFactors, MINIMAL_DISTANCE, tau, m_length):
    #print("CIRC ARC ANGLE", m_circleArcAngle)
    #print("MINIMAL DIST", MINIMAL_DISTANCE)
    if np.any(abs(m_circleArcAngle) < MINIMAL_DISTANCE):
        al = m_length;
    else:
        al = abs(m_circleArcAngle) * abs(m_curvatureRadius);
    try:
        return(m_scalingFactors/ al); 
    except Exception as e:
        return 0

def propagate_magnus(Q0, freq, direction, quant, wall_admittance, C=None, E=None, DN=None, Rl=None, circle_arc_angle=0, field_f=[280,17000], R=46, o = []):
    # Seguimiento del tiempo

    numX = 165
    mn = 16
    #arc length
    Q0 = np.resize(Q0, (16))
    al = Q0/R  # longitud del arco
    #print("AL", al)
    dX = direction * al / (numX - 1)
    curv = 1/R
    k = 2 * np.pi * freq / snd_speed
    tau, l0, l1, dl0, dl1 = 0, 0, 0, 0, 0

    # Parámetros de la evolución del coeficiente l
    A0 = np.zeros((2 * mn, 2 * mn), dtype=np.complex128)
    A1 = np.zeros((2 * mn, 2 * mn), dtype=np.complex128)
    omega = np.zeros((2 * mn, 2 * mn), dtype=np.complex128)
    K2 = np.zeros((mn, mn), dtype=np.complex128)
    wall_admittance = 0
    bnd_spec_adm = np.zeros(mn, dtype=np.complex128)
    Y = 1

    if o == []:
        # Inicialización de las listas según el tipo de cantidad física
        if quant == 'IMPEDANCE':
            o.append(Q0)
            return o
        elif quant == 'ADMITTANCE':
            o.append(Q0)
            return o
        elif quant == 'PRESSURE':
            o.append(Q0)
            return o
        elif quant == 'VELOCITY':
            o.append(Q0)
            return o
    else:
        # Inicialización de las listas según el tipo de cantidad física
        if quant == 'IMPEDANCE' or quant == 'ADMITTANCE':
            o.append(Q0)
            dX = -al / (numX - 1)
        elif quant == 'PRESSURE' or quant == 'VELOCITY':
            o.append(Q0)
            dX = al / (numX - 1)
        #print("O", o)    

        # Calcular la admitancia de la pared


        # Calcular la admitancia específica de la frontera
        bnd_spec_adm = get_specific_bnd_adm(freq, bnd_spec_adm)

        # Calcular la matriz KR2
        KR2 = np.zeros((mn, mn), dtype=np.complex128)
        for s in range(len(KR2)):
            #print("KR2", KR2[s])
            #print("BND SPEC ADM", np.diag(bnd_spec_adm))
            #Compute KR2 using admittance in and out of the throat sound field
            KR2 += KR2[s] * np.diag(bnd_spec_adm)[0] + wall_admittance * KR2[s]
            KR2 += KR2[s] * np.diag(bnd_spec_adm)[1] + wall_admittance * KR2[s]

        matricesMag = 0
        propag = 0

        # Discretizar el eje X
        for i in range(numX - 1):
            # Implementación del esquema de Magnus
            if direction < 0:
                    tau = ((numX - i) - 1.5) / (numX - 1)
            else:
                    tau = (i + 0.5) / (numX - 1)
            l0 = magnus_effect_scaling(np.array([numX, numX]), tau)
            dl0 = -Y * magnus_effect_scaling_derivative(circle_arc_angle, curv, 1, Rl, tau, 0)

            # Construir la matriz K2
            K2 = np.zeros((mn, mn), dtype=np.complex128)
            for j in range(mn):
                #print("FIELD F", field_f)
                K2[j, j] = (2 * np.pi * freq / snd_speed) ** 2 - (k * l0) ** 2
            K2 += 1j * k * l0 * KR2
            #print("K2 (MAGNUS EFFECT FREQUENCY DERIVATIVES)", K2)
            #print("CURVE", curv)
            #print("MN", mn)
            #print("l0", l0)
            #print("C", C)
            # Construir la matriz A0
            omega = np.block([[dl0 / l0 * E],
                                  [np.identity(mn) - curv * l0 * C / l0 ** 2],
                                  [K2 + curv * l0 * (C * (k * l0) ** 2 - DN)],
                                  [-dl0 / l0 * np.conjugate(E).T]])

            # Exponenciación de la matriz omega
            #print("DX", dX)
            omega = np.exp(dX * np.nan_to_num(omega))
            #print("OMEGA OF MAGNUS EFFECT", omega[mn-1:, mn-1:])
            #print("O", omega[-1])
            # Computar la cantidad propagada en el siguiente punto
            if quant == 'IMPEDANCE':
                o.append((omega[:mn-1, :mn-1] @ o[-1] + omega[:mn-1, mn-1:]) @
                                   np.linalg.inv(omega[mn-1:, :mn-1] @ o[-1] + omega[mn-1:, mn-1:]))
                return o
            elif quant == 'ADMITTANCE':
                in_out_omega = omega[:mn-1, :mn-1] + np.dot(np.resize(omega[:mn-1, mn-1:][:16],(16)), o[-1])
                epsilon = 0.001
                in_out_omega = in_out_omega + epsilon * np.eye(15)
                #print("IN OUT OMEGA", in_out_omega)
                o.append((omega[mn-1:, :mn-1] + np.dot(np.resize(omega[mn-1:, mn-1:][:16],(16)),o[-1])) @
                                    np.linalg.inv(in_out_omega))
                return o  
            elif quant == 'PRESSURE':
                o.append((omega[:mn-1, :mn-1] + omega[:mn-1, mn-1:] @ o[numX - 1 - i]) @
                                    o[-1])
                return o   
            elif quant == 'VELOCITY':
                #print("OMEGA", np.shape(omega[mn-1:, :mn-1][:16,0]) )
                #print("O", np.shape(o[len(o) - 1 - i]))
                o.append(np.dot(omega[mn-1:, :mn-1][:16,0], o[len(o) - 1 - i]) + np.dot(omega[mn-1:, mn-1:][:16,0],
                                       o[-1]))
        return o                       


def propagate_velocity_pressure(start_velocity, start_pressure,
                                    freq, start_section, end_section,
                                    direction, circle_arc_angle, 
                                    Rl, cx0, dcx0, field_f, modes, C, DN, E, junctions):
    prev_velocity = np.copy(start_velocity)
    prev_pressure = np.copy(start_pressure)
    tmp_Q, P, Y = [], [], []
    F = []
    num_sections = len(cx0)
    num_x = 165
    next_sec, nI, nNs = None, None, None
    pressure = np.zeros_like(start_pressure, dtype=np.complex128)
    wall_interface_admit = 1j * 2. * np.pi * freq * thermal_bnd_spec_adm / snd_speed
        
    # Loop over sections
    i = start_section
    while i != end_section:
        try:
            next_sec = int(i + np.sign(dcx0))
            Q = np.sign(dcx0)
            P = np.sign(dcx0)
            i = int(i)
            next_sec = int(next_sec)
            #print("CX0I",cx0, "START SECTION",i)
            nI = len(cx0[int(i)-1])
            #print("NEXT SEC", next_sec)
            nNs = len(cx0[int(next_sec)-1])

            # Propagate axial velocity and acoustic pressure in the section
            #print("CX0", cx0)
            admittance = propagate_magnus(cx0[int(i)], freq, np.sign(dcx0), 'ADMITTANCE', prev_pressure, C, E, DN, Rl, circle_arc_angle, field_f)
            tmp_Q = []
            num_x = 1
            Y = cx0[0]
            P = cx0[0]
            #print("Y", Y)
            #print("P", P)
            for pt in range(int(num_x)):
                    if num_x > 1:
                        tau = pt / (num_x - 1) if direction == 1 else (num_x - 1 - pt) / (num_x - 1)
                    else:
                        tau = 1.
                    scaling = cx0[i]/tau
                    tmp_Q.append(Y[int(num_x) - 1 - pt] * P[pt])
            axial_velocity = tmp_Q
            #print("AXIAL VELOCITY", tmp_Q)
            # Get the scattering matrix
            #signed direction
            #direction = np.sign(dcx0)
            if np.sign(dcx0) == 1:
                #print("JUNCTIONS", junctions)
                F = junctions[i]
                G = np.eye(nI) - np.dot(F[0], F[0].T) if np.any(np.abs(cx0[i]) > np.abs(cx0[next_sec-1])) else np.eye(nNs) - np.dot(F[0].T, F[0])
            else:
                F = junctions[next_sec]
                G = np.eye(nI) - np.dot(F[0].T, F[0]) if np.any(np.abs(cx0[i]) > np.abs(cx0[next_sec-1])) else np.eye(nNs) - np.dot(F[0], F[0].T)

            prev_velocity = np.zeros(3, dtype=np.complex128)
            prev_pressure = np.zeros(3, dtype=np.complex128)
            #print("JUNCTIONS", F)            
            #print("COMPLEMENTARY JUNCTIONS", G)   
            if direction == -1:
                if cx0[i].area() * (cx0[i].scale_in() ** 2) > \
                            cx0[next_sec-1].area() * (cx0[next_sec-1].scale_out() ** 2):
                        prev_pressure += np.dot(F[0], cx0[i][0]) * scale_in(cx0[i]) / scale_out(cx0[next_sec-1])
                        prev_velocity += np.dot(cx0[next_sec-1][-1], prev_pressure)
                else:
                        #Junction losses in negative direction
                        prev_velocity += np.linalg.inv(np.eye(nNs) + wall_interface_admit * np.dot(G, cx0[next_sec-1][-1]) ) \
                                             @ (np.dot(F[0], cx0[i][0]) * scale_out(cx0[next_sec-1]) / scale_in(cx0[i]))
                        prev_pressure += np.dot(cx0[next_sec-1][-1], prev_velocity)
            else:
                if np.any((np.abs(cx0[i]) * (cx0[i]/1) ** 2) > \
                            (np.abs(cx0[next_sec-1]) * (cx0[next_sec-1]/1) ** 2)):
                        pp = np.dot((F[0].T), cx0[i][-1] * (cx0[i]/1) / (cx0[next_sec-1]/1))
                        pv = np.dot(cx0[next_sec-1][0], prev_pressure)
                        #print("PP", pp)
                        #print("PV", pv)
                        prev_pressure += pp
                        prev_velocity += np.dot(cx0[next_sec-1][0], prev_pressure)
                else:
                        #Junction losses in positive direction
                        wall_loss = np.dot(np.eye(nNs)[:2][:,2] - wall_interface_admit * G[:2][:,2], cx0[next_sec-1][-1].T)
                        #print("INV WALL LOSS", np.array([wall_loss, wall_loss]))
                        inv_wall_loss = np.linalg.inv(np.array([wall_loss, wall_loss])+1e-16)
                        F_to_cX0 = np.dot(F[0].T, cx0[i][-1])
                        F_from_cx0 = F_to_cX0  * (cx0[next_sec-1]/1) / (cx0[i]/1)
                        #print("F FROM CX0", F_from_cx0[:2])
                        prev_velocity += np.dot(inv_wall_loss[0], F_from_cx0[:2] )
                        prev_pressure += np.dot(cx0[next_sec-1][-1], prev_velocity)
            # Move to next section
            i += np.sign(dcx0)
            #print("JUNCTION VELOCITY LOSS", prev_velocity)
            #print("PRESSURE LOSS", prev_pressure)    
            # Propagate in the last section
            Q = np.sign(cx0[-1])
            P = np.sign(cx0[-1])
            velocity = propagate_magnus(cx0[-1], freq, dcx0, 'VELOCITY', prev_velocity, C=C,DN=DN,E=E,Rl=Rl, circle_arc_angle=circle_arc_angle)
            tmp_Q = []
            Y = cx0[-1]
            P = cx0[-1]
            num_x = dcx0
            for pt in range(int(num_x)):
                if num_x > 1:
                    tau = pt / (num_x - 1)
                else:
                    tau = 1.
                scaling = magnus_effect_scaling(cx0[end_section-1],tau)
                axial_velocity.append(Y[int(num_x) - 1 - pt] * P[pt])
            #print("AXIAL VELOCITY", axial_velocity)
            #print("VELOCITY", velocity) 
            #print("ADMITTANCE", admittance)               
            #print("PRESSURE", prev_pressure)      
        except Exception as e:
            #print("ANTI ALIASING FREQUENCY BEYOND THE AUDIBLE HUMAN RANGE")
            #logger.exception(e)
            velocity = 0
            return axial_velocity, velocity, prev_pressure, admittance       
    return axial_velocity, velocity, prev_pressure, admittance       
        
def lwfs(f, o, Ri):
    Rs = []
    N = 46
    R = np.radians(N)
    # Head and pinna
    xs = Ri  	
    w_path = (o+'_wfs_bandondeon_46speakers_head'+str(Ri[0])+'_torso'+str(Ri[1])+'_mask:w_propacoord:wave_field').replace('binaural/','')
    lr_path = (o+'_wfs_bandondeon_46speakers_'+str(Ri[0])+'_torso'+str(Ri[1])+'_mask:lr_propacoord:wave_field').replace('binaural/','')
    height_path = (o+'_wfs_bandondeon_46speakers_'+str(Ri[0])+'_torso'+str(Ri[1])+'_mask:height_propacoord:wave_field').replace('binaural/','')
    deep_path = (o+'_wfs_bandondeon_46speakers_'+str(Ri[0])+'_torso'+str(Ri[1])+'_mask:deep_propacoord:wave_field').replace('binaural/','')
    if os.path.exists(w_path+'.ogg'):
        return
    elif os.path.exists(lr_path+'.ogg'):
        return
    elif os.path.exists(height_path+'.ogg'):
        return
    elif os.path.exists(deep_path+'.ogg'):
        return
    audio = mono_stereo(read(f+'.ogg')[0])
    w_audio, lr_audio, height_audio, deep_audio = wfs_from_audio(audio,xsi=xs)
    #print("SYNTHESIZED LR AUDIO", lr_audio)
    #print("SYNTHESIZED HEIGHT AUDIO", height_audio)
    #print("SYNTHESIZED DEEP AUDIO", deep_audio)    
    #filtered_lr_soundfield = WFS_filter(lr_audio)
    #filtered_height_soundfield = WFS_filter(height_audio)
    #filtered_deep_soundfield = WFS_filter(deep_audio)    
    write_file(w_path,48000, np.float64(w_audio/np.max(w_audio)))
    write_file(lr_path,48000, np.float64(lr_audio/np.max(lr_audio)))
    write_file(height_path,48000, np.float64(height_audio/np.max(height_audio)))
    write_file(deep_path,48000,np.float64(deep_audio/np.max(deep_audio)))
    
#for f in list(os.walk('/home/mc/bandoneon'))[-1][-1]:
#   binaural_impulse_lwfs('bandoneon/'+f, 'binaural/'+f)




