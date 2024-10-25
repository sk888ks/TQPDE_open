using Serialization

let
    # input =================================
    src_dir = "../../src/" #Directry of Gate_generation_from_MPS_MPO.jl
    #DMRG settings
    scratch = false #true: create reference MPS by DMRG, false: load the MPS file
    nsweeps = 20 #sweep
    maxdim =  append!([10 for i in 1:3], [50 for i in 1:12], [1000 for i in 1:5]) # max bond dimention
    cutoff = 1e-8 # threshold for the singular value

    num_depth_MPSBW = 10  #d_prep
    num_sweep_MPSBW = 1000 # optimization cycles for Uprep

    seed = 1

    # end input =================================

    mindim = 2

    include(joinpath(joinpath(pwd(), src_dir), "Gate_generation_from_MPS_MPO.jl"))

    name_ham = "octatetraene"
    name_MPSBW = "$(name_ham)_depth$(num_depth_MPSBW)_sweep$(num_sweep_MPSBW)" 
    name_target_MPS = "$(name_ham)"


    #scratch
    if scratch
        psi_sp = get_psi_sp(name_ham, nsweeps, maxdim, mindim, cutoff)
        serialize("$(pickle_dir)/$(name_target_MPS).dat", psi_sp)
        # @infiltrate
    else
        psi_sp = deserialize("$(pickle_dir)/$(name_target_MPS).dat")
    end
    
    #BW unitary
    psi_list_random, amplitude_list_random = get_and_save_prep_MPSBW_2d_list(psi_sp, num_depth_MPSBW, num_sweep_MPSBW, name_MPSBW, seed = seed)

    nothing

end