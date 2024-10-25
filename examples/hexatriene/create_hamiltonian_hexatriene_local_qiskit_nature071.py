import numpy as np
from pyscf import gto, scf, lo, ao2mo, fci, mcscf
from pyscf import tools as pyscf_tools
from pyscf.lib import logger
import copy

from deap import base, creator, tools, algorithms
from qiskit_nature.second_q.mappers import InterleavedQubitMapper, JordanWignerMapper

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src")))

from TQPDEinput import get_dir_QPDE_path
QPDE_path = get_dir_QPDE_path()

pickle_path = f"{QPDE_path}/pickle"
molden_path = f"{QPDE_path}/molden"

from pyscf import gto, scf, lo, ao2mo, fci
from pyscf import tools as pyscf_tools
import pickle
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import FermionicOp

def term_to_list(Hamiltonian): 
    # Convert the coefficients and Pauli operators of the Qiskit Hamiltonian into list form
    try:
        ham_coeff_list=list(Hamiltonian.coeffs)
        ham_pauli_list=list(Hamiltonian.primitive.paulis.settings["data"])
    except: 
        print("only one parameter is included in the Hamiltonian")
        try:
            ham_coeff_list=[Hamiltonian.coeff]
            ham_pauli_list=[Hamiltonian.primitive.settings["data"]]
        except:
            ham_coeff_list = Hamiltonian._coeffs.tolist()
            ham_pauli_list = Hamiltonian._pauli_list.settings["data"]
    
    return ham_coeff_list, ham_pauli_list

# Perform the specified calculation on each term of the Hamiltonian
def calculate_cost(pauli_list, coeff_list):
    total_cost = 0
    for pauli, coeff in zip(pauli_list, coeff_list):
        indices = [i for i, p in enumerate(pauli) if p in ['X', 'Y', 'Z']]
        if indices:
            max_index = max(indices)
            min_index = min(indices)
            total_cost += (max_index - min_index + 1) * abs(coeff)
    return total_cost

# Cost function for the genetic algorithm
def evaluate(individual):
    cost = 0
    for i, j, interaction in interactions_for_cost:
        squared_distance = (individual.index(i) - individual.index(j))**2
        cost += squared_distance * abs(interaction)
    return cost,

# Print the correspondence between MOs and AOs
from pyscf.tools import mo_mapping
def print_coeff_from_to(coeff, name=None):
    mo_list = []
    pz_pop = mo_mapping.mo_comps(['C 2pz','C 3pz'], mol, coeff)
    for i, p in enumerate(pz_pop):
        if p>0.9:
            mo_list.append(i)
    print(name, mo_list)

def max_xyz_length(strings):
    max_length = 0
    count = 0
    count_max = 0
    for s in strings:
        length = sum(1 for char in s if char in "XYZ")
        if length > max_length:
            max_length = length
            count_max = count
            # print(count_max)
        count += 1
    return max_length, count_max



# input ========================================
exact_calc = False
pickle_save = False
molden = False
calc_cost = False
# end input ====================================

#Octatetraene
pickle_name="hexatriene"
norb = 6
nelec = 6
coeff_idx1 = [19, 20, 21, 22, 23, 24]
coeff_idx2 = [19, 20, 21, 22, 23, 24]

# Defining the molecule
mol = gto.M(atom='''
   C  -0.593277   0.300058  -0.000035
   C   0.593162  -0.299976  -0.000023
   C  -1.871455  -0.412251  -0.000023
   C   1.871462   0.412124  -0.000031
   C  -3.054902   0.181227  -0.000027
   C   3.054973  -0.181236   0.000001
   H  -0.642698   1.377401  -0.000053
   H   0.642500  -1.377321  -0.000007
   H  -1.819439  -1.488952  -0.000011
   H   1.819480   1.488822  -0.000066
   H  -3.968533  -0.384394  -0.000016
   H  -3.151326   1.253321  -0.000039
   H   3.968596   0.384400  -0.000004
   H   3.151455  -1.253323   0.000034
''', basis='sto-3g')

mol.incore_anyway = True

molden_name=pickle_name

# Running Hartree-Fock calculation
mf = scf.RHF(mol)
mf.kernel()

# Extracting the MO coefficients and energies
mo_coeff = mf.mo_coeff
mo_energies = mf.mo_energy


# Display the correspondence between AO indices and labels
ao_labels = mol.spheric_labels()
# for i, label in enumerate(ao_labels):
#     print(f"AO {i}: {label}")

# Display the correspondence between MOs and AOs
print_coeff_from_to(mo_coeff, "mo_coeff before casci")

mc = mcscf.CASCI(mf, norb, nelec)
mo = mc.sort_mo(coeff_idx1, mo_coeff, base=0)
mc.run(mo)
print_coeff_from_to(mc.mo_coeff, "mo_coeff after casci")

h1, ecore = mc.get_h1cas() # h1 in CAS
h2 = mc.get_h2cas() # h2 in CAS
h2 = ao2mo.restore(1, h2, h1.shape[0]) # restore symmetry

mc_coeff = mc.mo_coeff[:,coeff_idx2] #coeff_idx
coeff_for_loc = copy.deepcopy(mc.mo_coeff)


mc_coeff_temp = copy.deepcopy(mc_coeff)
size_cas = len(mc_coeff[0])
# Construct the Hamiltonian
hamiltonian_HF = ElectronicEnergy.from_raw_integrals(h1, h2)
fermionic_op_HF = hamiltonian_HF.second_q_op()

# Jordan-Wigner transformation
blocked_mapper_HF = JordanWignerMapper()
interleaved_mapper_HF = InterleavedQubitMapper(blocked_mapper_HF)
qubit_hamiltonian_HF = interleaved_mapper_HF.map(fermionic_op_HF)
# qubit_hamiltonian_HF = blocked_mapper_HF.map(fermionic_op_HF)


# Get the list of Pauli operators and coefficients
coeff_list_HF, pauli_list_HF = term_to_list(qubit_hamiltonian_HF)

max_length_HF,arg_HF = max_xyz_length(pauli_list_HF)
print(max_length_HF,coeff_list_HF[arg_HF],pauli_list_HF[arg_HF],arg_HF)

# Execute localization
loc_orb = lo.Boys(mol, mc_coeff).kernel()
loc_coeff = loc_orb

# 1-electron integrals
h1 = np.einsum('pi,pq,qj->ij', loc_coeff, mf.get_hcore(), loc_coeff) #Is transpose needed?

# 2-electron integrals
eri = ao2mo.incore.full(mf._eri, loc_coeff)

# Calculate exchange integrals between all molecular orbitals
eri_4d = ao2mo.restore(1, eri, size_cas)
exchange_integrals = []
nmo = loc_coeff.shape[1]
for i in range(nmo):
    for j in range(nmo):
        if i > j:
            K_ij = eri_4d[i, j, j, i]
            if K_ij > 1e-16:
                exchange_integrals.append([i, j, K_ij])

# Interaction list
interactions_for_cost = exchange_integrals  

# Get the total number of orbitals
num_orbitals = nmo

print(f"Initial Fitness: {evaluate([i for i in range(nmo*2)])[0]}")

# Genetic algorithm settings
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Set the optimization objective to minimization
creator.create("Individual", list, fitness=creator.FitnessMin)  # Define an individual as a list

toolbox = base.Toolbox()
# Generate a random permutation of orbital indices based on the number of orbitals
toolbox.register("indices", np.random.permutation, num_orbitals)  # Generate a random permutation of orbital indices
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)  # Initialize individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Initialize population

toolbox.register("mate", tools.cxOrdered)  # Use ordered crossover
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)  # Register mutation with index shuffling, mutation probability is 5%
toolbox.register("select", tools.selTournament, tournsize=3)  # Use tournament selection, tournament size is 3
toolbox.register("evaluate", evaluate)  # Register the evaluation function

# Run the genetic algorithm
population = toolbox.population(n=50)  # Generate an initial population of 50 individuals
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, verbose=False)  # Run the genetic algorithm
# Crossover probability 70%, mutation probability 20%, generations 100, no detailed output

# Display the best individual
best_ind = tools.selBest(population, 1)[0]
print("Best Individual:", best_ind)
print("Best Fitness:", evaluate(best_ind)[0])

# Display the correspondence between MOs and AOs
# print_coeff_from_to(loc_coeff, "loc_coeff",0,9)

# Display the correspondence between MOs and AOs
print_coeff_from_to(coeff_for_loc, "mc.mo_coeff")

# Replace localized orbitals with the original CASCI orbitals
coeff_for_loc[:,coeff_idx2] = loc_coeff[:, best_ind]
coeff_for_sa = copy.deepcopy(coeff_for_loc)

# Run a new CASCI calculation with localized orbitals
print('Running a new CASCI calculation with localized orbitals')
sorted_mc = mcscf.CASCI(mf, norb, nelec)
# Display the correspondence between MOs and AOs
sorted_mc.mo_coeff = coeff_for_loc
print_coeff_from_to(coeff_for_loc, "sorted_mc.mo_coeff before replace")
mo = mc.sort_mo(coeff_idx2, coeff_for_loc, base=0)
sorted_mc.kernel()

# Obtain 1-electron and 2-electron integrals in the CAS space and the core energy shift
h1_sorted, ecore = sorted_mc.get_h1cas() # h1 in CAS
h2_sorted = sorted_mc.get_h2cas() # h2 in CAS
h2_sorted = ao2mo.restore(1, h2_sorted, h1_sorted.shape[0]) # expand to 4-rank tensor

# Construct the Hamiltonian
hamiltonian = ElectronicEnergy.from_raw_integrals(h1_sorted, h2_sorted)
fermionic_op = hamiltonian.second_q_op()

# Jordan-Wigner transformation
blocked_mapper = JordanWignerMapper()
interleaved_mapper = InterleavedQubitMapper(blocked_mapper)
qubit_hamiltonian = interleaved_mapper.map(fermionic_op)
# qubit_hamiltonian = blocked_mapper.map(fermionic_op)

# Get the list of Pauli operators and coefficients
coeff_list, pauli_list = term_to_list(qubit_hamiltonian)

max_length,arg = max_xyz_length(pauli_list)
print(max_length)
max_length,arg = max_xyz_length(pauli_list)
print(max_length,coeff_list[arg],pauli_list[arg],arg)

# Calculate excited state energy
print('Taking the average of two states')
mc_sa = mcscf.CASCI(mf, norb, nelec)
mc_sa.mo_coeff = coeff_for_sa 
mc_sa.state_average_([0.5, 0.5])  # Take the average of two states
mc_mo = mc_sa.sort_mo(coeff_idx2, coeff_for_sa, base=0)

# Display the correspondence between MOs and AOs
print_coeff_from_to(mc_sa.mo_coeff, "mc_mo")

mc_sa.kernel(mc_mo)

# Obtain the ground and first excited state energies
state_energies = mc_sa.e_states

# Energy gap between the ground state and the first excited state
energy_gap = state_energies[1] - state_energies[0]

print(f"CASCI Ground State Energy: {state_energies[0]}")
print(f"CASCI First Excited State Energy: {state_energies[1]}")
print(f"CASCI Energy Gap: {energy_gap}")

if calc_cost:
    # Calculate the cost for each term in the Hamiltonian
    total_cost_HF = calculate_cost(pauli_list_HF, coeff_list_HF)
    print(f"Total cost HF: {total_cost_HF}")

    total_cost = calculate_cost(pauli_list, coeff_list)
    print(f"Total cost: {total_cost}")

if exact_calc:
    eigvals_HF, eigvecs_HF = np.linalg.eigh(qubit_hamiltonian_HF.to_matrix())
    ground_state_energy_HF = eigvals_HF[0]
    print(f"Ground state energy HF: {ground_state_energy_HF + ecore}")
    
    # Diagonalize the Hamiltonian
    eigvals, eigvecs = np.linalg.eigh(qubit_hamiltonian.to_matrix())
    eigvecsT = eigvecs.T
    for i in range(256):
        bin_string = bin(np.argmax(np.abs(eigvecs.T[i])))[2:].zfill(8)
        count = bin_string.count("1")
        # print(i, count, bin_string, eigvals[i])
        if count == 4:
            print(i, count, bin_string, eigvals[i])
    ground_state_energy = eigvals[0]
    print(f"Ground state energy: {ground_state_energy + ecore}")
    print(f"Exact gap {eigvals[1]-eigvals[0]}")

if pickle_save:
    with open(f"{pickle_path}/{pickle_name}.pickle", "wb") as f:
        pickle.dump(qubit_hamiltonian, f)

    try:
        with open(f"{pickle_path}/{pickle_name}_pso.pickle", "wb") as f:
            pickle.dump(qubit_hamiltonian.primitive, f)
    except:
        with open(f"{pickle_path}/{pickle_name}_pso.pickle", "wb") as f:
            pickle.dump(qubit_hamiltonian, f)

    with open(f"{pickle_path}/{pickle_name}_HF.pickle", "wb") as f:
        pickle.dump(qubit_hamiltonian_HF, f)

    try:
        with open(f"{pickle_path}/{pickle_name}_HF_pso.pickle", "wb") as f:
            pickle.dump(qubit_hamiltonian_HF.primitive, f)
    except:
        with open(f"{pickle_path}/{pickle_name}_HF_pso.pickle", "wb") as f:
            pickle.dump(qubit_hamiltonian_HF, f)

if molden:
    pyscf_tools.molden.from_mo(mol, f"{molden_path}/{molden_name}.molden", loc_coeff)
    pyscf_tools.molden.from_mo(mol, f"{molden_path}/{molden_name}_HF.molden", mc_coeff_temp)
