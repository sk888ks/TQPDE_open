import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src")))

from QPDE_Bayes_session_MPSBWMPO_RD import *

backend_name = "aer" #"faketorino", "aer", "kawasaki", "least_busy", "torino", "faketokyo", "osaka","sherbrooke","nazca","kyoto","cusco"

HF = False

if HF:
    name = "hexatriene_HF"
else:
    name = "hexatriene"

shots = int(1e4)
dd = True #dynamical decoupling
V2 = True #sampler V2
rep_delay_option = "def" #delay option in qiskit
max_session_retries = 2


#set load file names
time_step = 0.1

num_time_slice = 100
num_depth_evolMPOBW = 8
num_sweep_BW = 10000

num_depth_prepMPSBW = 10
num_sweep_MPSBW = 1000

#load Hamiltonian
name_prepMPSBW = f"{name}_depth{num_depth_prepMPSBW}_sweep{num_sweep_MPSBW}"
name_ham = name
name_evolMPOBW = f"{name}_slice{num_time_slice}_step{time_step}_depth{num_depth_evolMPOBW}_sweep{num_sweep_BW}"


# Bayes setting
max_iter_bayes = 30 
num_points = 21  # samples in each iter
threshold_var = 0.005  # thoreshold for termination

# initial gaussian setting. mean, variance, and peak
initial_mu = 0.00 
initial_sigma2 = 4.0
initial_peak = 1.0

max_eval_gaussian_fit = 1
initial_layout = None #qiskit option

get_prob_list = True #get measured probability distributions
#use qctrl
if backend_name == "aer" or backend_name == "faketorino":
    qctrl=False
else:
    qctrl=True
gaussian_means, gaussian_variances, prob_list = QPDE_Bayes_session_MPSBWMPO_RD(backend_name, time_step, name_prepMPSBW, num_depth_prepMPSBW, name_ham, name_evolMPOBW, num_depth_evolMPOBW, shots, dd, V2, rep_delay_option, max_session_retries, initial_layout, max_iter_bayes, num_points, threshold_var, initial_mu, initial_sigma2, initial_peak, max_eval_gaussian_fit, get_prob_list,qctrl)

