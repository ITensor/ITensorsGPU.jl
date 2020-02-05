using Pkg
Pkg.activate("../..")
using TimerOutputs, Statistics
include("peps.jl")
# get basic simulation parameters 
Nx         = tryparse(Int, ARGS[1])
Ny         = tryparse(Int, ARGS[1])
env_add    = tryparse(Int, ARGS[2])
chi        = tryparse(Int, ARGS[3])
loadname   = ARGS[4]
env_maxdim = chi + env_add
simple_update_cutoff = -1

# log file which keeps track of more detailed info about the simulation, not super exciting
io = open("full_peps_$(string(Nx))_$chi.txt", "w+")
logger = SimpleLogger(io)
global_logger(logger)

# Hamiltonian parameters
J  = 1.0
sites = siteinds("S=1/2",Nx*Ny)

## GPU RUN
# disallow scalar indexing on GPU, which is very slow 
CuArrays.allowscalar(false)
A = loadMPStoPEPS(loadname, Nx, Ny, chi)
#A = checkerboardPEPS(sites, Nx, Ny, mindim=chi)
# to the user, these appear as normal ITensors, but they have on-device storage
# Julia can detect this at runtime and appropriately dispatch to CUTENSOR
cA = cuPEPS(A)
H  = makeCuH_XXZ(Nx, Ny, J)
@info "Built cA and H"
# run heaviest functions one time to make Julia compile everything
@info "Built cA and H"
Ls = buildLs(cA, H; mindim=1, maxdim=maxlinkdim(cA), env_maxdim=env_maxdim)
@info "Built first Ls"
Rs = buildRs(cA, H; mindim=1, maxdim=maxlinkdim(cA), env_maxdim=env_maxdim)
@info "Built first Rs"

dummyI = MPS(Ny, fill(ITensor(1.0), Ny), 0, Ny+1)
dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny)) 
Ancs = buildAncs(cA, dummyEnv, Rs[2], H, 1)
N, E = measureEnergy(cA, dummyEnv, Rs[2], Ancs, H, 1, 1)
@show E/(Nx*Ny)

#cA, Ls, Rs = rightwardSweep(cA, Ls, Rs, H; sweep=0, mindim=chi, simple_update_cutoff=simple_update_cutoff)
#cA, Ls, Rs = leftwardSweep(cA, Ls, Rs, H; sweep=0, mindim=chi, simple_update_cutoff=simple_update_cutoff)
println("Starting main sweep, chi = $chi")
# actual profiling run
prefix = "$(Nx)_$(env_add)_$(chi)_load"
cA, tS, bytes, gctime, memallocs = @timed doSweeps(cA, Ls, Rs, H; mindim=1, maxdim=chi, simple_update_cutoff=simple_update_cutoff, sweep_count=30, cutoff=0.0, env_maxdim=env_maxdim, do_mag=true, prefix=prefix)
println("Done sweeping GPU $tS")
flush(stdout)
flush(io)
