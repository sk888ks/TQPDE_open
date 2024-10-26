#dir path
from TQPDEinput import get_dir_QPDE_path, get_token, get_api_key, get_organization_slug, get_hub_group_project
dir_QPDE_path = get_dir_QPDE_path()
ibm_token = get_token()
api_key = get_api_key()
get_organization_slug = get_organization_slug()

import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Session
import pickle
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino
from datetime import datetime

from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 300
import copy
import sys
# from get_superposition_from_constrained_Ham import *
from scipy.optimize import curve_fit
# Alternative
from qiskit.qasm2 import dumps
from tenacity import retry, wait_chain, wait_fixed, retry_if_result
import math

# FIREOPAL
try:
    import fireopal 
    from fireopal import credentials, execute, get_action_metadata, get_result
    fireopal.config.configure_organization(organization_slug=get_organization_slug) 
    fireopal.authenticate_qctrl_account(api_key=api_key)
except:
    print("No fireopal module")

# Define function, setting gap to 0.1
def function(x, t):
    return 0.5 * (1 + np.cos((x - 0.1) * t))

# Define Gaussian function
def gaussian(x, a, b, c):
    return a * np.exp(-((x - b)**2) / (2 * c**2))

@retry( 
    wait=wait_chain(*[wait_fixed(2) for _ in range(15)] + [wait_fixed(10)]), 
    retry=retry_if_result(lambda status: status not in ("SUCCESS", "FAILURE", "REVOKED")) )
def poll_for_completion():
    most_recent_action = get_action_metadata()[0] 
    return most_recent_action.status

def QPDE_Bayes_session_MPSBWMPO_RD(backend_name, time_step, name_prepMPSBW, num_depth_prepMPSBW, name_ham, name_evolMPOBW, num_depth_evolMPOBW, shots=int(1e5), dd=True, V2=True, rep_delay_option=None, max_session_retries=3, initial_layout=None, max_iter_bayes=5, num_points=21, threshold_var=None, initial_mu=0.0, initial_sigma2=1.0, initial_peak=1.0, max_eval_gaussian_fit=4, get_prob_list=False, qctrl=False): 
    if backend_name not in ["faketorino", "aer", "kawasaki", "least_busy", "torino", "faketokyo", "osaka","sherbrooke","nazca","kyoto","cusco","fez"]:
        raise Exception("Incorrect backend_name")
    hub, group, project = get_hub_group_project()
    
    # Store the mean and variance of the Gaussian at each step
    gaussian_means = [initial_mu]
    gaussian_variances = [initial_sigma2]
    posterior_peaks = [initial_peak]
    prob_list = []

    if qctrl:
        print("qctrl")
        if backend_name == "kawasaki":
            hub = hub
            group = group
            project = project
            token = ibm_token
            credentials = fireopal.credentials.make_credentials_for_ibmq(
                token=token, hub=hub, group=group, project=project
            )

            service = QiskitRuntimeService(token=token, instance=hub + "/" + group + "/" + project, channel="ibm_quantum")
            backend_name_qctrl = "ibm_kawasaki"
        else:
            hub = hub
            group = group
            project = project
            token = ibm_token
            credentials = fireopal.credentials.make_credentials_for_ibmq(
                token=token, hub=hub, group=group, project=project
            )

            service = QiskitRuntimeService(token=token, instance=hub + "/" + group + "/" + project, channel="ibm_quantum")
            if backend_name == "osaka":
                backend_name_qctrl = "ibm_osaka"
            elif backend_name == "sherbrooke":
                backend_name_qctrl = "ibm_sherbrooke"
            elif backend_name == "torino":
                backend_name_qctrl = "ibm_torino"
            elif backend_name == "brisbane":
                backend_name_qctrl = "ibm_brisbane"
            elif backend_name == "kyoto":
                backend_name_qctrl = "ibm_kyoto"
            elif backend_name == "fez":
                backend_name_qctrl = "ibm_fez"
            else:
                raise Exception(f"{backend_name} is not supported")
            
        backend = service.backend(backend_name_qctrl)

        print(backend)

        # Open session
        # define sampler
        with open(f"{dir_QPDE_path}/pickle_dat/{name_ham}_pso.pickle", "rb") as f:
            ham = pickle.load(f)

        num_so = ham.num_qubits

        # Load MPSBW
        MPSBW_list = []
        for depth_MPSBW in range(num_depth_prepMPSBW):
            MPSBW_list_temp = []
            for unitary_idx in range(int(num_so/2)):
                gate = np.load(f"{dir_QPDE_path}/npy/{name_prepMPSBW}_MPSBW_depth{depth_MPSBW+1}_{unitary_idx+1}.npy")
                MPSBW_list_temp.append(gate)
            MPSBW_list.append(MPSBW_list_temp)

        # Load MPO
        BW_list = []
        for depth_MPO in range(num_depth_evolMPOBW):
            BW_list_temp = []
            if depth_MPO %2 == 0:
                for unitary_idx in range(int(num_so/2)):
                    gate = np.load(f"{dir_QPDE_path}/npy/{name_evolMPOBW}_MPO_depth{depth_MPO+1}_{unitary_idx+1}.npy")
                    BW_list_temp.append(gate)
                BW_list.append(BW_list_temp)
            else:
                for unitary_idx in range(int(num_so/2)-1):
                    gate = np.load(f"{dir_QPDE_path}/npy/{name_evolMPOBW}_MPO_depth{depth_MPO+1}_{unitary_idx+1}.npy")
                    BW_list_temp.append(gate)
                BW_list.append(BW_list_temp)

        # Set initial Gaussian distribution
        prior_mu = copy.deepcopy(gaussian_means[-1])  # Mean of the prior distribution
        prior_sigma2 = copy.deepcopy(gaussian_variances[-1])  # Variance of the prior distribution
        prior_peak = copy.deepcopy(posterior_peaks[-1])

        # Repeated Bayesian estimation
        for step_bayes in range(max_iter_bayes):
            # Fit Gaussian to observed data to obtain mean and variance
            prior_sigma = np.sqrt(prior_sigma2)
            time = 1.8 / prior_sigma2
            num_step = int(time/time_step)
            # Select sample points from Gaussian distribution
            eps_list = np.linspace(prior_mu - prior_sigma2, prior_mu + prior_sigma2, num_points)

            eps = Parameter("ε")

            # Initialize quantum circuit
            qc = QuantumCircuit(num_so+1)

            # Apply MPSBW
            for depth_idx, MPSBW_one_layer_list in enumerate(MPSBW_list):
                for unitary_idx, MPSBW_unitary in enumerate(MPSBW_one_layer_list): 
                    if depth_idx %2 == 0:
                        qc.unitary(MPSBW_unitary, [unitary_idx*2, unitary_idx*2+1])
                    else:
                        qc.unitary(MPSBW_unitary, [unitary_idx*2+1, unitary_idx*2+2])

            # Apply MPO
            for step in range(num_step):
                for depth_idx, BW_one_layer_list in enumerate(BW_list):
                    for unitary_idx, BW_unitary in enumerate(BW_one_layer_list): 
                        if depth_idx %2 == 0:
                            qc.unitary(BW_unitary, [unitary_idx*2+1, unitary_idx*2+2])
                        else:
                            qc.unitary(BW_unitary, [unitary_idx*2+2, unitary_idx*2+3])

            # Apply phase
            qc.p(time_step*num_step*eps, 0)

            # Apply the dagger of MPS
            for depth_idx, MPSBW_one_layer_list in enumerate(reversed(MPSBW_list)):
                for unitary_idx, MPSBW_unitary in enumerate(MPSBW_one_layer_list): 
                    if num_depth_prepMPSBW % 2 == 0:
                        if depth_idx %2 == 1:
                            qc.unitary(np.matrix.getH(MPSBW_unitary), [unitary_idx*2, unitary_idx*2+1])
                        else:
                            qc.unitary(np.matrix.getH(MPSBW_unitary), [unitary_idx*2+1, unitary_idx*2+2])
                    else:
                        if depth_idx %2 == 0:
                            qc.unitary(np.matrix.getH(MPSBW_unitary), [unitary_idx*2, unitary_idx*2+1])
                        else:
                            qc.unitary(np.matrix.getH(MPSBW_unitary), [unitary_idx*2+1, unitary_idx*2+2])


            qc.measure_all()

            pm = generate_preset_pass_manager(optimization_level=3, backend=backend, initial_layout=initial_layout)
            qc_transpiled = pm.run(qc)

            # Count the number of two-qubit gates after transpilation
            two_qubit_gates_count = qc_transpiled.count_ops().get('cz', 0) + qc_transpiled.count_ops().get('ecr', 0)
            print(f"2-qubit gate count: {two_qubit_gates_count}")

            qc_list = []
            results = []
            for val in eps_list:
                qasm_str = dumps(qc_transpiled.assign_parameters({eps:val}))
                qc_list.append(qasm_str)  # qasm convert

            validate_results = fireopal.validate(
            circuits=qc_list, credentials=credentials, backend_name=backend_name_qctrl
            )

            if validate_results["results"] == []:
                print("No errors found.")
            else:
                print("The following errors were found:")
                for error in validate_results["results"]:
                    print(error)
                raise RuntimeError("Validation failed")

            # Calculate posterior distribution
            def to_probability(value, shots):
                if isinstance(value, float):
                    return value
                return value / shots
            # Execute job
            previous_action_id = get_action_metadata()[0].model_id
            max_retry_fireopal_execute = max_eval_gaussian_fit
            for iter_retry_fireopal_execute in range(max_retry_fireopal_execute):
                try:
                    real_hardware_results = fireopal.execute(
                    circuits=qc_list, shot_count=shots,
                    credentials=credentials, backend_name=backend_name_qctrl)
                    results = real_hardware_results.result()["results"]
                    most_recent_action_id = get_action_metadata()[0].model_id
                except Exception as error: 
                    final_status = poll_for_completion()
                    if final_status != "SUCCESS": 
                        raise RuntimeError("Action did not successfully complete.") from error 
                    most_recent_action_id = get_action_metadata()[0].model_id
                    real_hardware_results = get_result(most_recent_action_id)
                    results = real_hardware_results.result()["results"]
                if previous_action_id == most_recent_action_id:
                    print("previous_action_id == most_recent_action_id")
                    if iter_retry_fireopal_execute == max_retry_fireopal_execute-1:
                        raise RuntimeError("Exceeded maximum retries") from error 
                    else:
                        continue
                else:
                    print(f"most_recent_action_id {most_recent_action_id}")
                    previous_action_id = copy.deepcopy(most_recent_action_id)
            
                print(get_action_metadata()[0])
                
                print("eps val")
                prob_list_temp = []

                counts = None
                for epsilon, re in zip(eps_list, results):
                    try:
                        counts = re["0"*(num_so+1)]
                        prob = to_probability(counts, shots)
                        print(f"{epsilon:.7f} {prob:.7f}")
                        prob_list_temp.append(prob)
                    except Exception as e:
                        counts = 0.0
                        prob = counts
                        print(f"{epsilon:.7f} {prob:.7f}")
                        prob_list_temp.append(prob)

                # Fit Gaussian. Initial peak is fixed at 1.0
                try:
                    popt, pcov = curve_fit(gaussian, eps_list, prob_list_temp, p0=[1.0, prior_mu, prior_sigma], maxfev=10000)
                    break
                except: 
                    raise Exception("Curve_fit failed")
                

            a_fit, y_mean, y_std = popt
            y_var = y_std ** 2  # Convert to variance

            prob_list.append(prob_list_temp)
            
            # Calculate posterior distribution parameters
            posterior_mu = (prior_sigma2 * y_mean + y_var * prior_mu) / (prior_sigma2 + y_var)
            posterior_sigma2 = (prior_sigma2 * y_var) / (prior_sigma2 + y_var)

            # Calculate the peak of the posterior distribution
            posterior_peak = gaussian(posterior_mu, a_fit, y_mean, np.sqrt(posterior_sigma2))

            # Add mean and variance to the list
            gaussian_means.append(posterior_mu)
            gaussian_variances.append(posterior_sigma2)
            posterior_peaks.append(posterior_peak)

            print(f"Iter {step_bayes+1}: Mean = {posterior_mu:.6f} Variance = {posterior_sigma2:.6f} Peak = {posterior_peak:.6f}")

            # Update prior distribution for next step
            prior_mu = posterior_mu
            prior_sigma2 = posterior_sigma2

            # Check termination condition
            if posterior_sigma2 < threshold_var:
                break
    else:
        if backend_name == "faketorino":
            backend = FakeTorino()
        elif backend_name == "faketokyo":
            backend = FakeTorino()
        elif backend_name == "aer":
            backend = AerSimulator()
        service = QiskitRuntimeService(channel="ibm_quantum", instance = hub + "/" + group + "/" + project)
        if backend_name == "kawasaki":
            backend = service.backend("ibm_kawasaki")
        else:
            if backend_name == "least_busy":
                backend = service.least_busy(simulator=False, operational=True)
            elif backend_name == "torino":
                backend = service.backend("ibm_torino")
            elif backend_name == "osaka":
                backend = service.backend("ibm_osaka")
            elif backend_name == "sherbrooke":
                backend = service.backend("ibm_sherbrooke")
            elif backend_name == "brisbane":
                backend = service.backend("ibm_brisbane")
            elif backend_name == "nazca":
                backend = service.backend("ibm_nazca")
            elif backend_name == "kyoto":
                backend = service.backend("ibm_kyoto")
            elif backend_name == "cusco":
                backend = service.backend("ibm_cusco")
            elif backend_name == "fez":
                backend = service.backend("ibm_fez")
            # Set the repetition delay
        if backend_name not in ["aer", "faketorino", "faketokyo"]:
            print("rep_delay")
            print(backend.configuration().rep_delay_range)
            print(backend.configuration().default_rep_delay)

            if rep_delay_option == "max":
                rep_delay = backend.configuration().rep_delay_range[1]
            else:
                rep_delay = backend.configuration().default_rep_delay
        else:
            rep_delay = 0.0

            print(f"rep_delay: {rep_delay}")

        print(backend)

        # Store the means and variances of the Gaussian at each step
        gaussian_means = [initial_mu]
        gaussian_variances = [initial_sigma2]
        posterior_peaks = [initial_peak]

        prob_list = []

        # Open session
        for _ in range(max_session_retries):
            try:
                with Session(backend=backend) as session:
                    # Define sampler
                    if V2:
                        from qiskit_ibm_runtime import SamplerV2 as Sampler, Options
                        sampler = Sampler(session=session)
                        options = sampler.options
                        if dd:
                            options.dynamical_decoupling.enable = dd
                            print(f">>> dynamical decoupling is turned on: {sampler.options.dynamical_decoupling.enable}")
                        options.execution.rep_delay = rep_delay
                        print(f">>> rep_delay = {sampler.options.execution.rep_delay}")
                    else:
                        from qiskit_ibm_runtime import SamplerV1 as Sampler, Options
                        if dd:
                            options = Options(optimization_level=3, resilience_level=1)
                        else:
                            options = Options(optimization_level=3)
                        options.rep_delay = rep_delay
                        sampler = Sampler(session=session, options=options)

                    with open(f"{dir_QPDE_path}/pickle_dat/{name_ham}_pso.pickle", "rb") as f:
                        ham = pickle.load(f)

                    num_so = ham.num_qubits

                    # Load MPSBW
                    MPSBW_list = []
                    for depth_MPSBW in range(num_depth_prepMPSBW):
                        MPSBW_list_temp = []
                        for unitary_idx in range(int(num_so / 2)):
                            gate = np.load(f"{dir_QPDE_path}/npy/{name_prepMPSBW}_MPSBW_depth{depth_MPSBW + 1}_{unitary_idx + 1}.npy")
                            MPSBW_list_temp.append(gate)
                        MPSBW_list.append(MPSBW_list_temp)

                    # Load MPO
                    BW_list = []
                    for depth_MPO in range(num_depth_evolMPOBW):
                        BW_list_temp = []
                        if depth_MPO % 2 == 0:
                            for unitary_idx in range(int(num_so / 2)):
                                gate = np.load(f"{dir_QPDE_path}/npy/{name_evolMPOBW}_MPO_depth{depth_MPO + 1}_{unitary_idx + 1}.npy")
                                BW_list_temp.append(gate)
                            BW_list.append(BW_list_temp)
                        else:
                            for unitary_idx in range(int(num_so / 2) - 1):
                                gate = np.load(f"{dir_QPDE_path}/npy/{name_evolMPOBW}_MPO_depth{depth_MPO + 1}_{unitary_idx + 1}.npy")
                                BW_list_temp.append(gate)
                            BW_list.append(BW_list_temp)

                    # Set initial Gaussian distribution
                    prior_mu = copy.deepcopy(gaussian_means[-1])  # Mean of the prior distribution
                    prior_sigma2 = copy.deepcopy(gaussian_variances[-1])  # Variance of the prior distribution
                    prior_peak = copy.deepcopy(posterior_peaks[-1])

                    # Bayesian estimation loop
                    for step_bayes in range(max_iter_bayes):
                        # Fit Gaussian to observed data to get the mean and variance
                        prior_sigma = np.sqrt(prior_sigma2)
                        time = 1.8 / prior_sigma2
                        num_step = math.ceil(time / time_step)
                        # Select sample points from Gaussian distribution
                        eps_list = np.linspace(prior_mu - prior_sigma2, prior_mu + prior_sigma2, num_points)

                        eps = Parameter("ε")

                        # Initialize quantum circuit
                        qc = QuantumCircuit(num_so + 1)

                        # Apply MPSBW
                        for depth_idx, MPSBW_one_layer_list in enumerate(MPSBW_list):
                            for unitary_idx, MPSBW_unitary in enumerate(MPSBW_one_layer_list):
                                if depth_idx % 2 == 0:
                                    qc.unitary(MPSBW_unitary, [unitary_idx * 2, unitary_idx * 2 + 1])
                                else:
                                    qc.unitary(MPSBW_unitary, [unitary_idx * 2 + 1, unitary_idx * 2 + 2])

                        # Apply MPO
                        for step in range(num_step):
                            for depth_idx, BW_one_layer_list in enumerate(BW_list):
                                for unitary_idx, BW_unitary in enumerate(BW_one_layer_list):
                                    if depth_idx % 2 == 0:
                                        qc.unitary(BW_unitary, [unitary_idx * 2 + 1, unitary_idx * 2 + 2])
                                    else:
                                        qc.unitary(BW_unitary, [unitary_idx * 2 + 2, unitary_idx * 2 + 3])

                        # Apply phase
                        qc.p(time_step * num_step * eps, 0)

                        # Apply the dagger of MPS
                        for depth_idx, MPSBW_one_layer_list in enumerate(reversed(MPSBW_list)):
                            for unitary_idx, MPSBW_unitary in enumerate(MPSBW_one_layer_list):
                                if num_depth_prepMPSBW % 2 == 0:
                                    if depth_idx % 2 == 1:
                                        qc.unitary(np.matrix.getH(MPSBW_unitary), [unitary_idx * 2, unitary_idx * 2 + 1])
                                    else:
                                        qc.unitary(np.matrix.getH(MPSBW_unitary), [unitary_idx * 2 + 1, unitary_idx * 2 + 2])
                                else:
                                    if depth_idx % 2 == 0:
                                        qc.unitary(np.matrix.getH(MPSBW_unitary), [unitary_idx * 2, unitary_idx * 2 + 1])
                                    else:
                                        qc.unitary(np.matrix.getH(MPSBW_unitary), [unitary_idx * 2 + 1, unitary_idx * 2 + 2])

                        qc.measure_all()

                        pm = generate_preset_pass_manager(optimization_level=3, backend=backend, initial_layout=initial_layout)
                        qc_transpiled = pm.run(qc)

                        # Count the number of two-qubit gates after transpilation
                        two_qubit_gates_count = qc_transpiled.count_ops().get('cz', 0) + qc_transpiled.count_ops().get('ecr', 0)
                        print(f"2-qubit gate count: {two_qubit_gates_count}")

                        qc_list = []
                        for val in eps_list:
                            qc_list.append(qc_transpiled.assign_parameters({eps: val}))

                        print()

                        # Calculate posterior distribution
                        job = sampler.run(qc_list, shots=shots)
                        print(f">>> Job ID: {job.job_id()}")
                        print(f">>> Job Status: {job.status()}")

                        # Execute
                        results = job.result()
                        print("eps val")
                        prob_list_temp = []

                        if results is None:
                            job = service.job(job.job_id())
                            results = job.result()

                        counts = None
                        if V2:
                            for epsilon, re in zip(eps_list, results):
                                try:
                                    counts = re.data.meas.get_counts()["0" * (num_so + 1)]
                                    prob = counts / shots
                                    print(f"{epsilon:.7f} {prob:.7f}")
                                    prob_list_temp.append(prob)
                                except Exception as e:
                                    counts = 0.0
                                    prob = counts / shots
                                    print(f"{epsilon:.7f} {prob:.7f}")
                                    prob_list_temp.append(prob)
                        else:
                            for epsilon, re in zip(eps_list, results.quasi_dists):
                                prob = re[0]
                                print(epsilon, prob)
                                prob_list_temp.append(prob)

                        # Fit Gaussian. Initial peak is fixed at 1.0
                        popt, pcov = curve_fit(gaussian, eps_list, prob_list_temp, p0=[1.0, prior_mu, prior_sigma], maxfev=10000)
                        a_fit, y_mean, y_std = popt
                        y_var = y_std ** 2  # Convert to variance

                        prob_list.append(prob_list_temp)

                        # Calculate posterior distribution parameters
                        posterior_mu = (prior_sigma2 * y_mean + y_var * prior_mu) / (prior_sigma2 + y_var)
                        posterior_sigma2 = (prior_sigma2 * y_var) / (prior_sigma2 + y_var)

                        # Calculate the peak of the posterior distribution
                        posterior_peak = gaussian(posterior_mu, a_fit, y_mean, np.sqrt(posterior_sigma2))

                        # Add mean and variance to the list
                        gaussian_means.append(posterior_mu)
                        gaussian_variances.append(posterior_sigma2)
                        posterior_peaks.append(posterior_peak)

                        print(f"Iter {step_bayes + 1}: Mean = {posterior_mu:.6f} Variance = {posterior_sigma2:.6f} Peak = {posterior_peak:.6f}")

                        # Update prior distribution for next step
                        prior_mu = posterior_mu
                        prior_sigma2 = posterior_sigma2

                        # Check termination condition
                        if posterior_sigma2 < threshold_var:
                            break
                break
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected. Exiting...")
                sys.exit(0)
            except SystemExit as e:
                print(f"Exiting: {e}")
                sys.exit(0)
            except Exception as e:
                print(f"A session error occurred: {e}")
        else:
            raise Exception(f"Job failed after maximum session retries {max_session_retries}")
    
    if get_prob_list:
        return gaussian_means, gaussian_variances, prob_list
    else:
        return gaussian_means, gaussian_variances

