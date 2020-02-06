using Pkg
Pkg.activate("../..")
using TimerOutputs, Statistics, HDF5 
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
#A = loadPEPSfromFile(loadname)
A = checkerboardPEPS(sites, Nx, Ny, mindim=chi)
@show loadname, isfile(loadname)
if isfile(loadname)
    global A
    fi = h5open(loadname,"r")
    for ii in 1:Ny, jj in 1:Nx
        At = read(fi, "A_$(ii)_$(jj)", ITensor)
        A[ii, jj] = At
    end
    close(fi)
end
@show A
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

println("Starting main sweep, chi = $chi")
# actual profiling run
prefix = "$(Nx)_$(env_add)_$(chi)_loadgauge"
cA, tS, bytes, gctime, memallocs = @timed doSweeps(cA, Ls, Rs, H; mindim=1, maxdim=chi, simple_update_cutoff=simple_update_cutoff, sweep_count=50, cutoff=0.0, env_maxdim=env_maxdim, do_mag=true, prefix=prefix)

A = collect(cA)
fo = h5open("peps_L_$(Nx)_chi_$(chi)_env_$(env_add).h5","w")
for ii in 1:Ny, jj in 1:Nx
    write(fo, "A_$(ii)_$(jj)", A[ii, jj])
end
close(fo)
println("Done sweeping GPU $tS")
flush(stdout)
flush(io)
