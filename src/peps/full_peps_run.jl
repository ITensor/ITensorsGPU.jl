using TimerOutputs, Statistics
include("peps.jl")
# get basic simulation parameters 
Nx  = tryparse(Int, ARGS[1])
Ny  = tryparse(Int, ARGS[1])

chi = tryparse(Int, ARGS[2])
simple_update_cutoff = -1 

# log file which keeps track of more detailed info about the simulation, not super exciting
io = open("full_peps_$(string(Nx))_$chi.txt", "w+")
logger = SimpleLogger(io)
global_logger(logger)

# Hamiltonian parameters
J  = 1.0
sites = siteinds("S=1/2",Nx*Ny)

## CPU RUN
A = checkerboardPEPS(sites, Nx, Ny, mindim=chi)
H = makeH_XXZ(Nx, Ny, J)
@info "Built A and H"

for col in reverse(2:Nx)
    global A
    A = gaugeColumn(A, col, :left; mindim=1, maxdim=chi)
end
# run heaviest functions one time to make Julia compile everything
Ls = buildLs(A, H; mindim=1, maxdim=chi)
@info "Built first Ls"
Rs = buildRs(A, H; mindim=1, maxdim=chi)
@info "Built first Rs"
A, Ls, Rs = rightwardSweep(A, Ls, Rs, H; sweep=0, mindim=1, maxdim=chi, simple_update_cutoff=simple_update_cutoff)
A, Ls, Rs = leftwardSweep(A, Ls, Rs, H; sweep=0, mindim=1, maxdim=chi, simple_update_cutoff=simple_update_cutoff)

# actual profiling run
(tL, tR), tS, bytes, gctime, memallocs = @timed doSweeps(A, Ls, Rs, H; mindim=1, maxdim=chi, simple_update_cutoff=simple_update_cutoff, sweep_count=10)
println("Done sweeping CPU $tS")
flush(stdout)
flush(io)
## GPU RUN

# disallow scalar indexing on GPU, which is very slow 
CuArrays.allowscalar(false)
A = checkerboardPEPS(sites, Nx, Ny, mindim=chi)
# to the user, these appear as normal ITensors, but they have on-device storage
# Julia can detect this at runtime and appropriately dispatch to CUTENSOR
cA = cuPEPS(A)
H  = makeCuH_XXZ(Nx, Ny, J)
@info "Built cA and H"

# run heaviest functions one time to make Julia compile everything
@info "Built cA and H"
Ls = buildLs(cA, H; mindim=chi, maxdim=chi)
@info "Built first Ls"
Rs = buildRs(cA, H; mindim=chi, maxdim=chi)
@info "Built first Rs"
cA, Ls, Rs = rightwardSweep(cA, Ls, Rs, H; sweep=0, mindim=chi, maxdim=chi, simple_update_cutoff=simple_update_cutoff)
cA, Ls, Rs = leftwardSweep(cA, Ls, Rs, H; sweep=0, mindim=chi, maxdim=chi, simple_update_cutoff=simple_update_cutoff)

# actual profiling run
(tL, tR), tS, bytes, gctime, memallocs = @timed doSweeps(cA, Ls, Rs, H; mindim=chi, maxdim=chi, simple_update_cutoff=simple_update_cutoff, sweep_count=10)
println("Done sweeping GPU $tS")
flush(stdout)
flush(io)
