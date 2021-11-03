# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 19:02:27 2020

@author: sofia
"""
from scipy import integrate
import numpy as np
import copy as copy
import math
import scipy as sp
import random as rn
import scipy.interpolate as inter
import numpy.fft as fft
import time
import matplotlib.pyplot as plt
#import random_fields as fd

mpc_metre = 3.085678e+22
G = 6.67430e-11 / mpc_metre ** 3
mean_density = 1e-26 * mpc_metre ** 3
c = 3e8 / mpc_metre


def coord_array(size, spacing):
   positions = []
   values = sp.arange(0,size,spacing) 
   for i in range(len(values)):
       for j in range(len(values)):
           positions.append([values[i],values[j]]) 
           
   return positions

def adding_error(convergence, std=0.3):
    if type(convergence)==list:
        convergence=np.array(convergence)
    noise = np.random.normal(scale = std, size = convergence.shape)
    return convergence+noise


def gridding_galaxies(size, bin_spacing, galaxies, conv_galaxies_clust):
    pos_array = np.array(coord_array(size, bin_spacing))
    pos_array = pos_array.reshape(math.trunc(size/bin_spacing), math.trunc(size/bin_spacing), 2)
    conv_field_gridded = [[[0,0,0] for i in range(len(pos_array))] for j in range(len(pos_array))]
    galaxies = np.array(galaxies)/bin_spacing #rescale so that truncating is easy
    for i in range(len(galaxies)):
        curr_bin = conv_field_gridded[math.trunc(galaxies[i][0])][math.trunc(galaxies[i][1])]
        new_val =  conv_galaxies_clust[i]
        #curr_bin[0]: mean convergence
        #curr_bin[1]: std
        #curr_bin[2]: number of galaxies
        updt_mean = (curr_bin[0]*curr_bin[2] + new_val)/(curr_bin[2]+1)
        updt_var = curr_bin[2]*(curr_bin[1]**2 +(curr_bin[0]+updt_mean)**2) + (new_val - updt_mean)**2
        updt_var = (updt_var)/(curr_bin[2]+1)
        updt_num = curr_bin[2]+1
        conv_field_gridded[math.trunc(galaxies[i][0])][math.trunc(galaxies[i][1])] = [updt_mean,updt_var,updt_num]        
    
    return conv_field_gridded


def k_vector_shifted(size, factor, r_spacing = 1, real= False):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y k_z.
        
        Input:
            size (type: integer) : size of array
            factor (type: integer) : factor by which x will be larger than y,z
            r_spacing (type: integer) : for the rest of the code I assume r_spacing
            is 1, so I will have to revisit some functions if I want to change this
            real (type: Boolean) : will determine whether we want to create negative &
            positive values of k_x or just negative (real=True)
        Returns:
            k_vector, numpy array of shape (size, factor*size, size), so (y,x,z)
            
        """
    shape = np.array([factor*size, size, size])
    k_spacing = 2 * sp.pi / (shape * r_spacing)
#    k_x = sp.arange(((-shape[0] + 1) // 2), (shape[0] + 1) // 2) * k_spacing[0]
    k_y = sp.arange(((-shape[1] + 1) // 2), (shape[1] + 1) // 2) * k_spacing[1]
    k_z = sp.arange(((-shape[2] + 1) // 2), (shape[2] + 1) // 2) * k_spacing[2]


    if real:
        k_x = sp.arange(0, (shape[2] + 1) // 2 + 1) * k_spacing[0]
    else:
        k_x = sp.arange(((-shape[0] + 1) // 2), (shape[0] + 1) // 2) * k_spacing[0]

    
#    k_vector = np.meshgrid(k_x, k_y, k_z)
    k_vector = fft.fftshift(np.meshgrid(k_x, k_y, k_z))
    
    print("K-VECTOR CALCULATED")
    print("DIMENSIONS K-VECTOR", sp.shape(k_vector))
    return k_vector #, k_spacing


def spec_func(k_mag):
    """
    :param k_mag: magnitude of k vector
    :return: returns power for given k vector using spectral data
    """
    h = 0.7
    log_spec = sp.loadtxt("Matter_Power_Spectrum_Data_amol.csv", skiprows=1, delimiter=",", usecols=(6, 7))
    log_spec_func = inter.interp1d(log_spec[:, 0], log_spec[:, 1], fill_value="extrapolate")
    print("SPEC FUNC GENERATED")
    
    return (10 ** log_spec_func(sp.log10(k_mag * h))) / (h ** 3)


def normalise_range(array, max_val_norm=1):
    max_abs = abs(array).max()
    factor=max_val_norm/max_abs
    norm_array= factor*array
    return norm_array


def show_galaxies(galaxies):
    y,x =zip(*galaxies)
#    x = x[::-1]
    plt.scatter(x,y,s=1)
    plt.show()



def clustered_galaxies(k_vector, max_val_norm, spacing, constant=0.005):
    """
    Mass_r_source is the overdensity \delta
    """
    
#    alpha = 3
    size=len(k_vector[0])  
    positions=coord_array(size, spacing) #creates an array with all the positions on the source plane
    
    #Create the overdensity field (\delta)

    k_magnitude = np.power(k_vector[0]**2 + k_vector[1]**2 + k_vector[2]**2, 1/2 )    
    power_spec = spec_func(k_magnitude) / ((2 * sp.pi) ** 3) #lambaCDM power spec
    noise = np.random.normal(size = power_spec.shape) + 1j * np.random.normal(size = power_spec.shape) 
    
    mass_k = noise*sp.sqrt(power_spec / 2)
    mass_r = convert_rspace(mass_k)
    mass_r = rewrite(mass_r) #just rewrites it as (x,y,z) instead of (y,x,z)
    mass_r_source = normalise_range(np.array(mass_r[0]),max_val_norm=max_val_norm) #takes first slice and normalises


#    mass_r_source = normalise_range(fd.gaussian_random_field(alpha = alpha, size=size),max_val_norm=max_val_norm)
    
    
    #interpolate
    x_arr = sp.arange(0,size,1)
    y_arr = copy.deepcopy(x_arr)
    mass_r_source_int = inter.RegularGridInterpolator((x_arr, y_arr), mass_r_source, bounds_error=False, fill_value=None)
    galaxies=[] #initialise an empty array of galaxy positions
    repetition = 0
    
    for i in range(len(positions)):
        mu = constant*(1+mass_r_source_int(positions[i])[0])
        if mu < 0:
            mu=0
        num_galaxies = np.random.poisson(mu) #number of galaxies at positons[i]

        if num_galaxies > 1:
            repetition +=1
        for j in range(num_galaxies): #in case the Poisson returns >1
            galaxies.append(positions[i]) #append the positions of the galaxy
    print("NUMBER OF POINTS WITH +1 galaxy:", repetition)
    return galaxies, mass_r_source



def grav_pot_k(k_vector, k_spacing, normalize_k = False, real= False):
    
    """ Returns a continuous and random gravitational potential
        
        Input:
            alpha (double): 
                The power of the power-law momentum distribution. Will indicate
                dominating modes
            size (integer):
                The size of the square face of the field
            flag_normalize (boolean, default = True):
                Normalizes the Gaussian Field:
                    - average of 0
                    - std of 1.0

        Returns:
            grav_field_k (numpy array of shape (size, factor*size, size)):
                The random gaussian random field in k-space
                
        """
    
    size = len(k_vector[0])     
    
    # Defines the magnitude as |k|
    k_magnitude = np.power(k_vector[0]**2 + k_vector[1]**2 + k_vector[2]**2, 1/2 )    
    
    power_spec = spec_func(k_magnitude)
    
    noise = np.random.normal(size = power_spec.shape) \
        + 1j * np.random.normal(size = power_spec.shape) 
    
    # Create field
#    coeff_k_re = sp.random.standard_normal(power_spec.shape) * sp.sqrt(power_spec / 2)
#    coeff_k_im = sp.random.standard_normal(power_spec.shape) * sp.sqrt(power_spec / 2)
    
    mass_k = noise*sp.sqrt(sp.prod(k_spacing)*(power_spec/ ((2 * sp.pi) ** 3)) / 2)
    
    #removes the zero in k_magnitude to avoid singularity
    #if size is even the location of 0 is in the 0,0,0 and size-1,0,size-1 if odd
    #and even factor 
    
    size_times_factor = len(k_magnitude[0])
    
    if size%2 == 0:
        k_magnitude[0,0,0] = 1
        
    elif size_times_factor %2 == 0: 
        k_magnitude[size-1,0,size-1]=1
        
    elif size_times_factor %2 != 0:
        k_magnitude[size-1,size_times_factor-1,size-1]=1
        

    grav_pot_k = - (4 * sp.pi * G * mean_density * mass_k) / (k_magnitude ** 2)
          
#    print("GRAVITATIONAL FIELD (K SPACE) FINALISED")
#    print("DIMENSIONS GRAV FIELD (y,x,z):", sp.shape(grav_pot_k))
    
    if normalize_k: 
        grav_pot_k = norm_std(grav_pot_k)
    
    return grav_pot_k, k_magnitude


def rewrite(array):
    """
    Takes a field expressed in coordinates (y,x,z) and rewrites it as (x,y,z), 
    where x is the 'longer' axis.
    """
    new_array = [[[0 for i in range(len(array))]for j in range(len(array))] for k in range(len(array[0]))]
    for j in range(len(array[0])):
        for i in range(len(array)):
            new_array[j][i] = array[i][j].tolist()
       
    return new_array



def norm_std(field):
    """
    Sets the mean to 0 and standard deviation to 1 for a field
    """
    field = field - np.mean(field)
    field = field/np.std(field)
    
    return field



def convert_rspace(k_space, field=False, normalize = False):
    """
    Converts an array from k_space into r_space
    """
#    k_space = sp.fftpack.fftshift(k_space)
    size = sp.shape(k_space)
    if len(size)==3: #for 3D arrays
        output_size = size[0]*size[1]*size[2]
    if len(size)==2: #for 2D arrays
        output_size = size[0]*size[1]
        
    r_space = fft.ifftn(k_space).real
    
    if field: print("GRAV POTENTIAL CONVERTED TO REAL SPACE")    
    else: print("ARRAY CONVERTED TO REAL SPACE")
    ("DIMENSIONS:", sp.shape(r_space))
    
    return output_size*r_space


def convert_kspace(r_space):
    """
    Converts an array from k_space into r_space
    """
#    k_space = sp.fftpack.fftshift(k_space)
    size = sp.shape(r_space)
    if len(size)==3: #for 3D arrays
        output_size = size[0]*size[1]*size[2]
    if len(size)==2: #for 2D arrays
        output_size = size[0]*size[1]

    r_space = fft.fftn(r_space)
    
    print("ARRAY CONVERTED TO K SPACE")
    
    return r_space/output_size



def lensing_potential_parallel(size, grav_potential, axis = "x_axis"):
    """
    Computes all the integrals for parallel lines from source plane to observer
    plane. This is for the approximation in Scenario 1 (see theory).
    """
    lensing_pot = [[0 for x in range(size)] for y in range(size)]
    if axis == "x_axis":
        for y_value in range(size):
            for z_value in range(size):
                integrand= lambda distance: (2 / (c ** 2)) * ((size - distance) * grav_potential[math.trunc(distance)][y_value][z_value])
                lens_pot = integrate.quad(integrand, 0, size)
                lensing_pot[int(y_value)][int(z_value)] = lens_pot[0]
    return lensing_pot




def integral(size, derivative, initial_coord = [0,0], final_coord=[0,0], factor = 5):
    """
    The coordinates determine the y and z position, because the x position is 
    fixed to x=0 (initial) and x=size (final).
    """
    i_coord = np.array([0, initial_coord[0], initial_coord[1]])
    f_coord = np.array([size, final_coord[0], final_coord[1]])
    
    r_vector = np.array([f_coord[0] - i_coord[0], f_coord[1] - i_coord[1], f_coord[2] - i_coord[2]])
    mod_r = np.sqrt(np.dot(r_vector, r_vector))
    precision = math.trunc(mod_r*factor)
    delta_r = r_vector*(1/precision)#creates a small vector dr with the size determined by precision
    displacement = list([i_coord + n*delta_r for n in range(precision+1)]) 
    
    
    #We create an array with all the coordinates of the boxes(the field is 
    #fundamentally discrete)that have been crossed
    boxes_crossed = [[0 for x in range(3)] for y in range(len(displacement))]
    for i in range(len(displacement)):
        for j in range(3):
            boxes_crossed[i][j] = math.trunc(displacement[i][j])
    
    
    integrand= lambda chi: (2 / (mod_r * c ** 2)) * ((mod_r - chi)*(chi)* 
                                 derivative[boxes_crossed[math.trunc(chi)][0]][boxes_crossed[math.trunc(chi)][1]][boxes_crossed[math.trunc(chi)][2]])
    integral = integrate.quad(integrand, 0, mod_r)
#    integral = 0
#    chi=0
#    for i in range(0,len(displacement)):
#        chi+=1/precision
#        integral += integrand(chi)
        
    return integral[0]


def int_field(which_field, derivatives_r, size, source = [0,0], observer=[0,0]):
    int_field=0
    f_coord = observer
    i_coord = source
    
    if which_field == "convergence":
        int_field = integral(size, derivatives_r[0], initial_coord=i_coord, final_coord=f_coord) + integral(size, derivatives_r[1], initial_coord=i_coord, final_coord=f_coord)
    
    if which_field == "real_shear":
        int_field = integral(size, derivatives_r[0], initial_coord=i_coord, final_coord=f_coord) - integral(size, derivatives_r[1], initial_coord=i_coord, final_coord=f_coord)
    
    if which_field == "convergence":
        int_field = 2*integral(size, derivatives_r[2], initial_coord=i_coord, final_coord=f_coord)
    
    return 0.5*int_field


def total_fields(which_field, derivatives_r, size):
    """
    Computes the total field at a point in the observer plane (mid point by default)
    Possible values for which_field = convergence, real_shear, img_shear
    """
    f_coord = [size//2, size//2]
    print("OBSERVER LOCATED AT:", f_coord)
    field = [[0 for i in range(size)] for j in range(size)]
    
    for i in range(size):
        for j in range(size):
            i_coord = [i,j]
            print("SOURCE POINT", i,j," INTEGRATED")
            field[i][j] = int_field(which_field=which_field, derivatives_r=derivatives_r, size=size, source=i_coord, observer=f_coord)
    
    return field



def flatten(array, dimensions=3):
    """
    Turns a ND array into a 1D array
    """
    flattened_array = []
    
    if dimensions==2:
        for i in range(len(array)):
            for j in range(len(array)):
                flattened_array.append(array[i][j])
            
    if dimensions==3:
        for i in range(len(array)):
            for j in range(len(array)):
                for k in range(len(array)):
                    flattened_array.append(array[i][j][k])
    
    return flattened_array



def derivatives_k(grav_pot_k, k_vector):
    """
    Computes the derivatives of the gravitational potential in k-space.
    diff[0] = dyy
    diff[1] = dzz
    diff[2] = dydz
    """
    diff = [0,0,0]
    diff[0] = - (k_vector[1] ** 2) * grav_pot_k
    diff[1] = - (k_vector[2] ** 2) * grav_pot_k
    diff[2] = - (k_vector[1] * k_vector[2]) * grav_pot_k
    
    return diff


def random_galaxies(num_galaxies, size):
    positions = sp.arange(0,size, 1)
    galaxies = [[rn.choice(positions), rn.choice(positions)] for i in range(num_galaxies)]
    
    return galaxies
        

def two_point_statistics(sources, field, size):
    """
    returns an array with all the bins (spaces by 1 unit), with the structure:
    [mean,std,number of pairs]
    """
    bins = [[0,0,0] for i in range(math.trunc(np.sqrt(2*(size**2))))]
    #np.sqrt(2*size**2) is the maximum allowed separation between two points (diagonal)
    for i in range(len(sources)):
        for j in range(i, len(sources)): #start at i+1 to not consider the source with itself
            sep_vect = [sources[i][0] - sources[j][0], sources[i][1] - sources[j][1]]
            separation = np.sqrt(np.dot(sep_vect, sep_vect))
            curr_bin = bins[math.trunc(separation)]
            new_val =  field[i]*field[j]
            #curr_bin[0]: mean
            #curr_bin[1]: std
            #curr_bin[2]: number of pairs
            updt_mean = (curr_bin[0]*curr_bin[2] + new_val)/(curr_bin[2]+1)
            updt_var = curr_bin[2]*(curr_bin[1]**2 +(curr_bin[0]+updt_mean)**2) + (new_val - updt_mean)**2
            updt_var = (updt_var)/(curr_bin[2]+1)
            updt_num = curr_bin[2]+1
            bins[math.trunc(separation)] = [updt_mean,updt_var,updt_num]
    
    return bins



def power_spec(num_points, conv_field_k, bins):

    k_spacing = 2 * sp.pi / num_points
    k_x = sp.arange(((-num_points + 1) // 2), (num_points + 1) // 2) * k_spacing
    k_y = k_x*(-1)
    print("LEN KX",len(k_x))
    
    """
    coords=[]
    for i in range(len(k_x)):
        for j in range(len(k_x)):
            coords.append([k_x[j], k_y[i]])
    #print("COORDS",coords)

    #rewrite it in our desired shape
    final_coords=[[] for i in range(len(k_x))]
    for i in range(len(k_x)):
        temporary=[]
        index = i*(len(k_x))
        for j in range(len(k_x)):
            temporary.append(coords[index+j])
        final_coords[i] = temporary
    
    print("FINAL COORDS", final_coords)
    """
    
    k_val_x, k_val_y = np.meshgrid(k_x,k_y)
    k_mag_grid = np.power(k_val_x**2 + k_val_y**2, 1/2)
    
    k_mag_grid_exp = k_mag_grid*(bins/k_mag_grid[0][0]) #scale all the values so that truncating means choosing the bin 
    print("SHAPE K MAG GRID EXPANDED", np.shape(k_mag_grid_exp))
    bins_arr = [[0,0,0] for i in range(bins+1)]
    
    for i in range(len(k_mag_grid_exp[0])):
        for j in range(len(k_mag_grid_exp[0])):
        
            curr_bin = bins_arr[math.trunc(k_mag_grid_exp[i][j])]
            new_val =  (conv_field_k[i][j]*np.conj(conv_field_k[i][j])).real
            #curr_bin[0]: mean
            #curr_bin[1]: std
            #curr_bin[2]: number of pairs
            updt_mean = (curr_bin[0]*curr_bin[2] + new_val)/(curr_bin[2]+1)
            updt_var = curr_bin[2]*(curr_bin[1]**2 +(curr_bin[0]+updt_mean)**2) + (new_val - updt_mean)**2
            updt_var = (updt_var)/(curr_bin[2]+1)
            updt_num = curr_bin[2]+1
            bins_arr[math.trunc(k_mag_grid_exp[i][j])] = [updt_mean,updt_var,updt_num]

    bins_power_grid = sp.arange(0, len(bins_arr), 1)*(k_mag_grid[0][0]/bins)
    return bins_arr, bins_power_grid

#%%

def amol_two_point(input_val, shift, weighting=None, mean_only=False):
    """
    Calculates two point statistics on 2D input for a given vector separation.
    Applying the weighting parameter allows specification of the number of sources at each grid point.
    :param input_val: 2D array of input values
    :param shift: 2D vector for separation in units of grid points
    :param weighting: 2D array indicating number of sources at each grid point coordinate; if None, assumed uniform
    :param mean_only: choose whether to output just mean or also variance and sample size
    :return: two point correlation value, variance, number of points sampled
    """
    xlen, ylen = sp.shape(input_val)
    xshift, yshift = shift

    if bool(xshift > 0) == bool(yshift > 0) or xshift == 0 or yshift == 0:
        computed = (input_val[abs(xshift):, abs(yshift):] * input_val[:xlen - abs(xshift), :ylen - abs(yshift)])
    else:
        computed = (input_val[abs(xshift):, :ylen - abs(yshift)] * input_val[:xlen - abs(xshift), abs(yshift):])

    if weighting is not None:
        if xshift == 0 and yshift == 0:
            return amol_combo_stats(computed, 0, (weighting + 1) * weighting / 2, mean_only=mean_only)
        else:
            if bool(xshift > 0) == bool(yshift > 0) or xshift == 0 or yshift == 0:
                shift_weighting = (
                        weighting[abs(xshift):, abs(yshift):] * weighting[:xlen - abs(xshift), :ylen - abs(yshift)])
            else:
                shift_weighting = (
                        weighting[abs(xshift):, :ylen - abs(yshift)] * weighting[:xlen - abs(xshift), abs(yshift):])
            return amol_combo_stats(computed, 0, shift_weighting, mean_only=mean_only)
    else:
        mean = sp.mean(computed)
        if mean_only:
            return mean
        else:
            variance = sp.var(computed)
            sample_size = sp.size(computed)
            return mean, variance, sample_size


def amol_two_point_all(input_val, weighting=None, mean_only=False):
    """
    Calculates two point statistics for all possible separation vectors on a regular 2D grid.
    Applying the weighting parameter allows specification of the number of sources at each grid point.
    :param input_val: 2D array of input values
    :param weighting: 2D array indicating number of sources at each grid point coordinate; if None, assumed uniform
    :param mean_only: choose whether to output just mean or also variance and sample size
    :return: two point correlation value, variance, number of points sampled AND corresponding array of vector shifts
    """
    xlen, ylen = sp.shape(input_val)
    shift_xpos = sp.reshape(sp.mgrid[0:xlen, 0:ylen], (2, -1)).T
    shift_xneg = sp.reshape(sp.mgrid[-(xlen - 1):0, 1:ylen], (2, -1)).T
    shift = sp.concatenate((shift_xneg, shift_xpos))

    if mean_only:
        output_arr = sp.zeros(len(shift))
    else:
        output_arr = sp.zeros([len(shift), 3])

    for i, shift_vec in enumerate(shift):
        if i % 1000 == 0:
            print(i, "/", len(shift))
        else:
            pass
        output_arr[i] = amol_two_point(input_val, shift_vec, weighting, mean_only)

    return output_arr, shift


def amol_combo_stats(mean_arr, variance_arr, sample_size_arr, mean_only=False):
    """
    Returns mean, variance and sample size for the population formed by merging together each sub-population described by the input arrays
    :param mean_arr: array of mean values for each sub-population
    :param variance_arr: array of variance values for each sub-population
    :param sample_size_arr: array of sample size values for each sub-population
    :param mean_only: if True, only returns mean value for merged population
    :return: mean, variance and sample size for combined population
    """

    if sp.any(sample_size_arr):
        combo_mean = sp.average(mean_arr, weights=sample_size_arr)

        if mean_only:
            return combo_mean
        else:
            combo_sample_size = sp.sum(sample_size_arr)
            combo_variance = sp.sum(sample_size_arr * (variance_arr + (mean_arr - combo_mean) ** 2)) / combo_sample_size
            return combo_mean, combo_variance, combo_sample_size
    else:
        if mean_only:
            return 0
        else:
            return 0, 0, 0

def amol_two_point_radialise(two_point_raw, shift_array, tolerance=1e-4, mean_only=False):
    """
    Collects two point statistics into form dependent only on magnitude of shift (not direction).
    Requires input of mean, variance and number of sample points for each shift vector.
    :param two_point_raw: mean, variance and sample number data of two point statistics for each vector
    :param shift_array: corresponding array of shift vectors
    :param tolerance: tolerance for magnitudes of shift vectors to be considered the same
    :param mean_only: choose whether to output just mean or also variance and sample size
    :return: two-point mean, variance and sample number, along with corresponding shift magnitudes
    """

    shift_mag = sp.sqrt(shift_array[:, 0] ** 2 + shift_array[:, 1] ** 2)
    ordered_shift_mag = shift_mag[shift_mag.argsort()]
    ordered_two_point_raw = two_point_raw[shift_mag.argsort()]
    shift_mag_output = []
    two_point_output = []
    i = 0
    while i < len(ordered_shift_mag):

        container_shift_mag = [ordered_shift_mag[i]]
        container_two_point = [ordered_two_point_raw[i]]
        init_shift_mag = ordered_shift_mag[i]
        if i < len(ordered_shift_mag) - 1:

            while abs(ordered_shift_mag[i + 1] - init_shift_mag) < tolerance:
                if i % 1000 == 0:
                    print(i, "/", len(ordered_shift_mag))
                else:
                    pass
                container_shift_mag.append(ordered_shift_mag[i + 1])
                container_two_point.append(ordered_two_point_raw[i + 1])
                i += 1
                if i < len(ordered_shift_mag) - 1:
                    pass
                else:
                    break
            i += 1
            container_shift_mag = sp.array(container_shift_mag)
            container_two_point = sp.array(container_two_point)
            if mean_only:
                new_mean = amol_combo_stats(container_two_point[:, 0],
                                       container_two_point[:, 1],
                                       container_two_point[:, 2],
                                       mean_only=True)
                shift_mean = amol_combo_stats(container_shift_mag,
                                         0,
                                         container_two_point[:, 2],
                                         mean_only=True)
                two_point_output.append(new_mean)
                shift_mag_output.append(shift_mean)
            else:
                new_mean, new_variance, new_sample_size = amol_combo_stats(container_two_point[:, 0],
                                                                      container_two_point[:, 1],
                                                                      container_two_point[:, 2])
                shift_mean, shift_variance, shift_sample_size = amol_combo_stats(container_shift_mag,
                                                                            0,
                                                                            container_two_point[:, 2])
                two_point_output.append([new_mean, new_variance, new_sample_size])
                shift_mag_output.append([shift_mean, shift_variance, shift_sample_size])
        else:
            i += 1
            new_mean = container_two_point[:, 0]
            shift_mean = container_shift_mag
            if mean_only:
                two_point_output.append(new_mean)
                shift_mag_output.append(shift_mean)
            else:
                new_variance = container_two_point[:, 1]
                new_sample_size = container_two_point[:, 2]
                shift_variance = 0
                shift_sample_size = container_two_point[:, 2]
                two_point_output.append([new_mean, new_variance, new_sample_size])
                shift_mag_output.append([shift_mean, shift_variance, shift_sample_size])

    return sp.array(two_point_output), sp.array(shift_mag_output)


    
    