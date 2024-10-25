using Serialization

let
    #input =================================
    src_dir = "../../src/" #Directry of Gate_generation_from_MPS_MPO.jl

    num_sites = 10 #sytem qubits = 2*num_sites

    #reference MPO settings
    time_step = 0.1
    num_time_slice = 100
    scratch = false #true: create reference MPO, false: load the MPO file

    num_depth_BW = 5 # d_evol
    num_sweep_BW = 1000 # optimization cycles for Uevol
    # =================================
    U = 10.0
    t = 1.0

    include(joinpath(joinpath(pwd(), src_dir), "Gate_generation_from_MPS_MPO.jl"))

    name_ham = "ham_Hubbard_site$(num_sites)_U$(U)_t$(t)_cp_pso"
    name_target_MPO = "Hubbard_site$(num_sites)_U$(U)_t$(t)_slice$(num_time_slice)_step$(time_step)_cp"
    name_BW = "Hubbard_site$(num_sites)_U$(U)_t$(t)_slice$(num_time_slice)_step$(time_step)_depth$(num_depth_BW)_sweep$(num_sweep_BW)_cp"

    ham = hamcall(name_ham)

    #coeff for itensor
    coeffs, paulis = term_to_list(ham)

    # Create N spin-one indices
    N = length(paulis[1])
    # println("#sites: ",N)
    sites = siteinds("S=1/2",N)

    #Hamiltonian = MPO
    H = get_Hamiltonian(coeffs, paulis, sites)

    #scratch
    if scratch
        evol_MPO = get_gate_second_trotter(ham, N, time_step, num_time_slice)
        serialize("$(pickle_dir)/$(name_target_MPO).dat", evol_MPO)
        # @infiltrate
    else
        evol_MPO = deserialize("$(pickle_dir)/$(name_target_MPO).dat")
    end

    #BW unitary
    evol_BW_2d_list, cost_list = get_and_save_evol_BW_2d_list(evol_MPO, num_depth_BW, num_sweep_BW, name_BW)

    nothing 
end


