# save directrie paths
try
    include("input_temp.jl")
catch e
    save_dir_pickle = "/xxxx/TQPDE_open/pickle_dat"
    save_dir_npy = "/xxxz/TQPDE_open/npy"
    pickle_dir = save_dir_pickle 
end
