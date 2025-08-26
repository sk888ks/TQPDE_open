using Serialization

let
    #input =================================
    src_dir = "../../src/" #Directry of Gate_generation_from_MPS_MPO.jl

    #reference MPO settings
    time_step = 0.1
    num_time_slice = 100
    scratch = false #true: create reference MPO, false: load the MPO file

    num_depth_BW = 8 # d_evol
    num_sweep_BW = 10000 # optimization cycles for Uevol
    # =================================

    include(joinpath(joinpath(pwd(), src_dir), "Gate_generation_from_MPS_MPO.jl"))

    name_ham = "octatetraene"
    name_target_MPO = "$(name_ham)_slice$(num_time_slice)_step$(time_step)"
    name_BW = "$(name_ham)_slice$(num_time_slice)_step$(time_step)_depth$(num_depth_BW)_sweep$(num_sweep_BW)"

    ham = hamcall(name_ham)
    # ham = hamcall_from_txt(name_ham) # if you want to read from txt file

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


