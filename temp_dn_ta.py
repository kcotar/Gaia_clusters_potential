

import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Constants

r_p = 0.587  # AE
e = 0.967
P = 76.  # yr
a = r_p/(1.-e)

# ------------------------
# Functions

def r_dist(a, e, v):
	return (a*(1.-e**2.))/(1.+e*np.cos(v))

def v_anom(e, E):
	return 2.*np.arctan(np.sqrt((1.+e)/(1.-e))*np.tan(E/2.))

def E_fun(e, E, t, P):
	return (E - e*np.sin(E) - 2.*np.pi*(t-0.)/P)

def dE_fun(e, E):
	return (1. - E*e*np.cos(E))

def dE_fun2(P):
	return (2.*np.pi/P)

def get_E(t, P, E_init=1e-10, prec=1e-5):
	E_new = E_init - E_fun(e, E_init, t, P)/dE_fun(e, E_init)
	while (np.abs(E_init) - np.abs(E_new))/E_init > prec:
		E_init = float(E_new)
		E_new = E_init - E_fun(e, E_init, t, P)/dE_fun(e, E_init)
		# print E_new
	return E_new

# ------------------------
# Newton metoda

t_step = np.double(1e-5)
t_array = np.arange(0., 150., t_step)  # in years
r_NR = np.zeros_like(t_array)

for i_t in range(len(t_array)):
	print 'Working on point '+str(i_t+1)
	E = get_E(t_array[i_t], P)
	v = v_anom(e, E)
	# print i_t, a, e, P
	r_NR[i_t] = r_dist(a, e, v)

# ------------------------
# Euler in RK4

G_konst = 6.67e-11  # N*kg^2/m^2
M_konst = 2.e30  # m
AU = 1.49e11  # m
year_to_sec = 365.*24.*60.*60.
v_p = np.sqrt(G_konst*M_konst*(2./(r_p*AU) - 1./(a*AU)))
print 'v_p', v_p
print 'v_p', v_p * year_to_sec / AU

def size_vect(x):
	return (np.sqrt(np.sum(x**2)))

def acceleration(R):
	R_norm_vect = R/size_vect(R)
	a_G = G_konst*M_konst/size_vect(R)**2
	return -1.*a_G*R_norm_vect

# t_step = 0.00001
# t_array = np.arange(0., 150., t_step)
n_t = len(t_array)
print 'Total: '+str(n_t)

r_Euler = np.zeros_like(t_array)
R_init = np.array([-r_p, 0.]) * AU
V_init = np.array([0., -v_p])
t_step_s = t_step*year_to_sec
for i_t in range(1, len(t_array)):
	if i_t % 1000 == 0:
		print 'Done: {:.1f}%'.format(100.*i_t/n_t)
	a_star = acceleration(R_init)
	#print a_star
	#print V_init
	#print R_init
	V_new = V_init + a_star*t_step_s
	R_new = R_init + V_init*t_step_s
	R_init = np.array(R_new)
	V_init = np.array(V_new)
	r_Euler[i_t] = size_vect(R_new)/AU

r_RK4_2 = np.zeros_like(t_array)
R_init = np.array([-r_p, 0.]) * AU
V_init = np.array([0., -v_p])
for i_t in range(1, len(t_array)):
	if i_t % 1000 == 0:
		print 'Done: {:.1f}%'.format(100.*i_t/n_t)
	k1 = acceleration(R_init)
	k2 = acceleration(R_init+0.5*k1)
	k3 = acceleration(R_init+0.5*k2)
	k4 = acceleration(R_init+k3)
	#print k1, k2, k3, k4
	a_star = 1./6.*(k1 + 2.*k2 + 2.*k3 + k4)
	V_new = V_init + a_star*t_step_s
	R_new = R_init + V_init*t_step_s
	R_init = np.array(R_new)
	V_init = np.array(V_new)
	r_RK4_2[i_t] = size_vect(R_new)/AU

# ------------------------
# Grafi


plt.plot(t_array, r_NR, color='black')
#plt.plot(t_array, r_RK4_2, color='red')
#plt.plot(t_array, r_Euler, color='green')
plt.ylim((0,40))
plt.savefig('r_od_t_metode.png', dpi=400)
plt.close()

# ------------------------
# Main - Geoidno polje EGM2008
"""
GM_earth = 3986004.415e8  # m^3/s^2
a_earth = 6378136.3  # m

def compute_g(r, lat, lon, model):
	n_max = np.max()
	G_multi = 0.
	for n in range(2, n_max):  # -1 offset
		r_multi = (a_earth/r)**n
		CS_sum = 0.
		for m in range(0, n):
			CS_sum += C_nm*np.cos(m*np.deg2rad(lat)) + S_nm*np.sin(m*np.deg2rad(lat))
		G_multi += r_multi * CS_sum
	return GM_earth/r*(1+G_multi)
"""







