import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src")))

from QPDE_Bayes_session_MPSBWMPO_RD import *

backend_name = "aer" #"faketorino", "aer", "kawasaki", "least_busy", "torino", "faketokyo", "osaka","sherbrooke","nazca","kyoto","cusco"
model = "Hubbard"

shots = int(1e4)
dd = True #dynamical decoupling
V2 = True #sampler V2
rep_delay_option = "def" #delay option in qiskit
max_session_retries = 1 


#set load file names
num_sites = 4

U = 10.0
t = 1.0

time_step = 0.1

num_time_slice = 100
num_depth_evolMPOBW = 5
num_sweep_BW = 1000

num_depth_prepMPSBW = 6
num_sweep_MPSBW = 1000

name_prepMPSBW = f"Hubbard_site{num_sites}_U{U}_t{t}_depth{num_depth_prepMPSBW}_sweep{num_sweep_MPSBW}_cp_vac"
name_ham = f"ham_Hubbard_site{num_sites}_U{U}_t{t}_cp"
name_evolMPOBW = f"Hubbard_site{num_sites}_U{U}_t{t}_slice{num_time_slice}_step{time_step}_depth{num_depth_evolMPOBW}_sweep{num_sweep_BW}_cp"

# Bayes setting
max_iter_bayes = 30
num_points = 21  # samples in each iter
threshold_var = 0.005  # thoreshold for termination

# initial gaussian setting
initial_mu = 20.23102405089335  # mean
initial_sigma2 = 4.0  # variance
initial_peak = 1.0 # peak (not used)

max_eval_gaussian_fit = 1
initial_layout = None #qiskit option

get_prob_list = True #get measured probability distributions
qctrl=False #use qctrl
gaussian_means, gaussian_variances, prob_list = QPDE_Bayes_session_MPSBWMPO_RD(backend_name, time_step, name_prepMPSBW, num_depth_prepMPSBW, name_ham, name_evolMPOBW, num_depth_evolMPOBW, shots, dd, V2, rep_delay_option, max_session_retries, initial_layout, max_iter_bayes, num_points, threshold_var, initial_mu, initial_sigma2, initial_peak, max_eval_gaussian_fit, get_prob_list,qctrl)

