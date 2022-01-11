# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 18:33:03 2020

@author: sofia
"""

import methods_d as methods
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.interpolate as inter
import math

print("START!")
size = 200
factor = 5
#spacing_grid_galaxies = 4

mpc_metre = 3.085678e+22
G = 6.67430e-11 / mpc_metre ** 3
mean_density = 1e-26 * mpc_metre ** 3
c = 3e8 / mpc_metre

# %%
"""
Create 3 different types of galaxy positions: clustered, grids, random
"""

k_vector = methods.k_vector_shifted(size=size, factor=factor)
shape = np.array([factor*size, size, size])
k_spacing = 2 * sp.pi / shape

# %%
galaxies_clust1, mass_r_source1 = methods.clustered_galaxies(k_vector=k_vector, max_val_norm=0.5, spacing=0.2, constant=0.01)
print("NUM GALAXIES CLUSTERED 1", len(galaxies_clust1))

galaxies_clust2, mass_r_source2 = methods.clustered_galaxies(k_vector=k_vector, max_val_norm=1, spacing=0.2, constant=0.01)
print("NUM GALAXIES CLUSTERED 2", len(galaxies_clust2))

galaxies_clust3, mass_r_source3 = methods.clustered_galaxies(k_vector=k_vector, max_val_norm=2, spacing=0.2, constant=0.01)
print("NUM GALAXIES CLUSTERED 3", len(galaxies_clust3))

galaxies_clust4, mass_r_source4 = methods.clustered_galaxies(k_vector=k_vector, max_val_norm=5, spacing=0.2, constant=0.01)
print("NUM GALAXIES CLUSTERED 4", len(galaxies_clust4))

#methods.show_galaxies(galaxies_clust)

plt.imshow(mass_r_source1, cmap='Greys') #show mass field to check
plt.colorbar()
x1,y1 =zip(*galaxies_clust1)
plt.scatter(x1,y1,s=0.3)
plt.savefig(fname= "clust1.png",dpi=1000)
plt.show()

plt.imshow(mass_r_source2, cmap='Greys') #show mass field to check
plt.colorbar()
x2,y2 =zip(*galaxies_clust2)
plt.scatter(x2,y2,s=0.3)
plt.savefig(fname= "clust2.png",dpi=1000)
plt.show()

plt.imshow(mass_r_source3, cmap='Greys') #show mass field to check
plt.colorbar()
x3,y3 =zip(*galaxies_clust3)
plt.scatter(x3,y3,s=0.3)
plt.savefig(fname= "clust3.png",dpi=1000)
plt.show()

plt.imshow(mass_r_source4, cmap='Greys') #show mass field to check
plt.colorbar()
x4,y4 =zip(*galaxies_clust4)
plt.scatter(x4,y4,s=0.3)
plt.savefig(fname= "clust4.png",dpi=1000)
plt.show()

# %%
spacing_grid_galaxies= 2
galaxies_grid = methods.coord_array(size, spacing_grid_galaxies)
print("NUM GALAXIES GRID", len(galaxies_grid))

x_grid, y_grid = zip(*galaxies_grid)
plt.scatter(x_grid, y_grid, s = 0.3)
plt.savefig(fname= "grid.png",dpi=1000)
plt.show()
# %%
galaxies_rand = methods.random_galaxies(len(galaxies_clust), size)

# %%
grav_pot_k, k_magnitude = methods.grav_pot_k(k_vector, k_spacing)

print("GRAV POT K",type(grav_pot_k), sp.shape(grav_pot_k))

# %%
"""
Plot the grav potential in r space
"""
grav_pot_r = methods.convert_rspace(grav_pot_k)
grav_pot_r = methods.rewrite(grav_pot_r)
grav_pot_r = np.array(grav_pot_r)/c**2
plt.xlabel('Grav Potential (slice)')
plt.imshow(grav_pot_r[size//3][:][:], cmap='Greens')
plt.colorbar()
plt.savefig(fname= "grav_pot.png",dpi=1000)
plt.show()


#k_magnitude_flat = methods.flatten(k_magnitude)
#power_spec_flat = np.log10(methods.flatten(power_spec))
#plt.plot(k_magnitude_flat, power_spec_flat, 'ro')
#plt.xlabel('k magnitude (logged)')
#plt.ylabel('power spec (logged)')
#print(k_magnitude_flat)
#plt.hist(k_magnitude_flat, bins=30)
#plt.xlabel('k magnitude')
#plt.show()

# %%
derivatives_k = methods.derivatives_k(grav_pot_k, k_vector)
derivatives_r = [0,0,0]
for i in range(3):
    derivatives_r[i] = methods.convert_rspace(derivatives_k[i])

print("DIMENSIONS OF DERIVATIVES", sp.shape(derivatives_r))

#rewrite to the coordinates I use (x,y,z) from (y,x,z)
derivatives_r = np.array([methods.rewrite(derivatives_r[0]), methods.rewrite(derivatives_r[1]), methods.rewrite(derivatives_r[2])])
print("NEW DIMENSIONS OF DERIVATIVES", sp.shape(derivatives_r))


# %%

conv_field_back = methods.total_fields(which_field = "convergence", derivatives_r = derivatives_r, size = size)
plt.xlabel('Conv Field background')
plt.imshow(conv_field_back, cmap='Oranges')
plt.colorbar()
plt.savefig(fname= "conv_back.png",dpi=1000)
plt.show()

f = open("saveytime.py", "w")
f.write("galaxies_clust1=" + str(galaxies_clust1))
f.write("galaxies_clust2=" + str(galaxies_clust2))
f.write("galaxies_clust3=" + str(galaxies_clust3))
f.write("grav_pot_k=" + str(grav_pot_k))
f.write("grav_pot_r=" + str(grav_pot_r))
f.write("conv_field_back=" + str(conv_field_back))
f.write("derivatives_k=" + str(derivatives_k))
f.write("derivatives_r=" + str(derivatives_r))
f.write("galaxies_grid=" + str(galaxies_grid))
f.write("k_vector=" + str(k_vector))
f.write("mass_r_source1=" + str(mass_r_source1))
f.write("mass_r_source2=" + str(mass_r_source2))
f.write("mass_r_source3=" + str(mass_r_source3))
f.close()
# %%

"""
Obtain two point correlation function for clustered galaxies
"""
conv_galaxies_clust1 =[0 for i in range(len(galaxies_clust1))]
for i in range(len(galaxies_clust1)):
    conv_galaxies_clust1[i] = methods.int_field(which_field="convergence", derivatives_r=derivatives_r, size=size, source = galaxies_clust1[i], observer=[size//2,size//2])

conv_galaxies_clust1 = methods.adding_error(conv_galaxies_clust1)

two_point_clust1 = methods.two_point_statistics(galaxies_clust1, conv_galaxies_clust1, size)
means_clust1 = []
yerrors_clust1 = []
num_pairs_clust1= []

for i in range(0,len(two_point_clust1)):
    means_clust1.append(two_point_clust1[i][0])
    yerrors_clust1.append(two_point_clust1[i][1])
    num_pairs_clust1.append(two_point_clust1[i][2])


# %%
"""
Obtain two point correlation function for clustered galaxies
"""
conv_galaxies_grid =[0 for i in range(len(galaxies_grid))]
for i in range(len(galaxies_grid)):
    conv_galaxies_grid[i] = methods.int_field(which_field="convergence", derivatives_r=derivatives_r, size=size, source = galaxies_grid[i], observer=[size//2,size//2])

#two_point_grid = methods.two_point_statistics(galaxies_grid, conv_galaxies_grid, size)
#means_grid = []
#yerrors_grid = []
#num_pairs_grid= []
#
#for i in range(0,len(two_point_grid)):
#    means_grid.append(two_point_grid[i][0])
#    yerrors_grid.append(two_point_grid[i][1])
#    num_pairs_grid.append(two_point_grid[i][2])

# %%
"""
Obtain two point correlation function for clustered galaxies
"""
conv_galaxies_rand =[0 for i in range(len(galaxies_rand))]
for i in range(len(galaxies_rand)):
    conv_galaxies_rand[i] = methods.int_field(which_field="convergence", derivatives_r=derivatives_r, size=size, source = galaxies_rand[i], observer=[size//2,size//2])

two_point_rand = methods.two_point_statistics(galaxies_rand, conv_galaxies_rand, size)
means_rand = []
yerrors_rand = []
num_pairs_rand= []

for i in range(0,len(two_point_rand)):
    means_rand.append(two_point_rand[i][0])
    yerrors_rand.append(two_point_rand[i][1])
    num_pairs_rand.append(two_point_rand[i][2])


# %%

"""
Obtain Power Spectrum of convergence
"""
binning_spacing = 1 #binnig spacing for galaxy gridding

print("SIZE PLANE (mpc):", size , "x", size)
print("GALAXY GRIDDING SPACING:", binning_spacing)

#add error k_f = k_o + err (\sim N(0.3))
conv_galaxies_clust_err = methods.adding_error(conv_galaxies_clust)
conv_field_gridded = methods.gridding_galaxies(size, binning_spacing, galaxies_clust, conv_galaxies_clust_err)


#rewrite these arrays
conv_field_grid_means = [[0 for i in range(len(conv_field_gridded))] for j in range(len(conv_field_gridded))]
num_galaxies_point = [[0 for i in range(len(conv_field_gridded))] for j in range(len(conv_field_gridded))]
for i in range(len(conv_field_gridded)):
    for j in range(len(conv_field_gridded)):
        conv_field_grid_means[i][j] = conv_field_gridded[i][j][0]
        num_galaxies_point[i][j] = conv_field_gridded[i][j][2]
              
        
#conv_grid_k = methods.convert_rspace(np.fft.fftshift(conv_grid_r))
conv_grid_k = methods.convert_kspace(conv_field_grid_means)

power_spec_clust, bins_power_clust = methods.power_spec(len(conv_grid_k), conv_grid_k, bins = 50)
power_clust = [] 
yerrors_power_clust = []
num_pairs_power_clust= []

for i in range(0,len(power_spec_clust)):
    power_clust.append(power_spec_clust[i][0])
    yerrors_power_clust.append(power_spec_clust[i][1])
    num_pairs_power_clust.append(power_spec_clust[i][2])


plt.xlabel('k - magnitude')
plt.ylabel('P_k')
#plt.errorbar(bins_clust, means_clust, yerr= sp.sqrt(np.array(yerrors_clust)/np.array(num_pairs_clust)))
#plt.errorbar(bins_grid, means_grid, yerr= sp.sqrt(np.array(yerrors_grid)/np.array(num_pairs_grid)))
plt.errorbar(bins_power_clust, power_clust, yerr = sp.sqrt(np.array(yerrors_power_clust)/np.array(num_pairs_power_clust)), fmt = '.k')
plt.savefig(fname= "power_spec12.png",dpi=800)
plt.show()
# %%

"""
Plot clustered clustered vs unclustered
"""

bins_clust = sp.arange(0, len(two_point_clust), 1)
bins_rand = sp.arange(0, len(two_point_rand), 1)

plt.xlabel('separation 'r' (binned)')
plt.ylabel('<kappa(x) kappa(x+r)>')
plt.errorbar(bins_clust, means_clust, yerr= sp.sqrt(np.array(yerrors_clust)/np.array(num_pairs_clust)))
plt.errorbar(bins_rand, means_rand, yerr= sp.sqrt(np.array(yerrors_rand)/np.array(num_pairs_rand)))
#plt.plot(bins_clust, means_clust)
#plt.plot(bins_grid, means_grid)
plt.savefig(fname= "comparison_clust_vs_unclust.png",dpi=800)
plt.legend(["clustered", "random"], loc ="upper right") 
plt.show()

plt.xlabel('separation 'r' (binned)')
plt.ylabel('number of pairs')
plt.plot(bins_clust, num_pairs_clust)
plt.plot(bins_rand, num_pairs_rand)
plt.savefig(fname= "comparison_position.png",dpi=800)
plt.legend(["clustered", "random"], loc ="upper right") 
plt.show()