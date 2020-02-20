using Pkg
Pkg.activate("../..")
using TimerOutputs, Statistics
include("peps.jl")
# get basic simulation parameters 
Nx  = tryparse(Int, ARGS[1])
Ny  = tryparse(Int, ARGS[1])
env_add = tryparse(Int, ARGS[2])
chi = tryparse(Int, ARGS[3])
env_maxdim = chi + env_add
simple_update_cutoff = 3

prefix = "twosite/$(Nx)_$(env_add)_$(chi)_twosite"
if length(ARGS) > 3
    run_int  = tryparse(Int, ARGS[4])
    prefix   = "twosite/$(Nx)_$(env_add)_$(chi)_$(run_int)_twosite"
end


# log file which keeps track of more detailed info about the simulation, not super exciting
io = open("full_peps_$(string(Nx))_$chi.txt", "w+")
logger = SimpleLogger(io)
global_logger(logger)

# Hamiltonian parameters
J  = 1.0
sites = siteinds("S=1/2",Nx*Ny)
CuArrays.allowscalar(false)
A = checkerboardPEPS(sites, Nx, Ny, mindim=chi)
# to the user, these appear as normal ITensors, but they have on-device storage
# Julia can detect this at runtime and appropriately dispatch to CUTENSOR
cA = cuPEPS(A)
H  = makeCuH_XXZ(Nx, Ny, J)
@info "Built cA and H"

println("After making A and H")
CuArrays.memory_status()
flush(stdout)
# run heaviest functions one time to make Julia compile everything
@info "Built cA and H"
Ls = buildLs(cA, H; mindim=1, maxdim=chi)
@info "Built first Ls"
println("After building Ls")
CuArrays.memory_status()
flush(stdout)
Rs = buildRs(cA, H; mindim=1, maxdim=chi)
@info "Built first Rs"

println("After building envs")
CuArrays.memory_status()
flush(stdout)
# actual profiling run
cA, tS, bytes, gctime, memallocs = @timed doSweeps(cA, Ls, Rs, H; mindim=chi, maxdim=chi, simple_update_cutoff=simple_update_cutoff, sweep_count=50, cutoff=0.0, env_maxdim=env_maxdim, do_mag=true, prefix=prefix)
println("Done sweeping GPU $tS")
flush(stdout)
flush(io)
