using TimerOutputs, Statistics
include("peps.jl")
function doSweeps(A::PEPS, Ls::Vector{Environments}, Rs::Vector{Environments}, H; mindim::Int=1, maxdim::Int=1, simple_update_cutoff::Int=4, sweep_count::Int=10, cutoff::Float64=0.)
    Ls, tL, bytes, gctime, memallocs = @timed buildLs(A, H; mindim=mindim, maxdim=maxdim)
    Rs, tR, bytes, gctime, memallocs = @timed buildRs(A, H; mindim=mindim, maxdim=maxdim)
    for sweep in 1:sweep_count
        if isodd(sweep)
            println("SWEEP RIGHT $sweep")
            A, Ls, Rs = rightwardSweep(A, Ls, Rs, H; sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
        else
            println("SWEEP LEFT $sweep")
            A, Ls, Rs = leftwardSweep(A, Ls, Rs, H; sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
        end
        flush(stdout)
        if sweep == simple_update_cutoff - 1
            for col in 1:Nx-1
                A = gaugeColumn(A, col, :right; mindim=1, maxdim=chi)
            end
            Ls = buildLs(A, H; mindim=mindim, maxdim=maxdim)
            Rs = buildRs(A, H; mindim=mindim, maxdim=maxdim)
        end
        if sweep == sweep_count
            A_ = deepcopy(A)
            L_s = buildLs(A_, H; mindim=mindim, maxdim=maxdim)
            R_s = buildRs(A_, H; mindim=mindim, maxdim=maxdim)
            x_mag = measureXmag(A_, L_s, R_s; mindim=mindim, maxdim=maxdim)
            z_mag = measureZmag(A_, L_s, R_s; mindim=mindim, maxdim=maxdim)
            display(z_mag)
            println()
            v_mag = measureSmagVertical(A_, L_s, R_s; mindim=mindim, maxdim=maxdim)
            display(v_mag)
            println()
            h_mag = measureSmagHorizontal(A_, L_s, R_s; mindim=mindim, maxdim=maxdim)
            display(h_mag)
            println()
        end
    end
    return tL, tR
end

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

#dummyI = MPS(Ny, fill(ITensor(1.0), Ny), 0, Ny+1)
#dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny)) 
## CPU RUN
A = checkerboardPEPS(sites, Nx, Ny, mindim=chi)
H = makeH_XXZ(Nx, Ny, J)
@info "Built A and H"
#=col = Nx
println("building Ls")
L_s = buildLs(A, H; mindim=chi, maxdim=chi)
println("building Rs")
R_s = buildRs(A, H; mindim=chi, maxdim=chi)
EAncEnvs = buildAncs(A, L_s[col - 1], dummyEnv, H, col)
N, E     = measureEnergy(A, L_s[col - 1], dummyEnv, EAncEnvs, H, 1, col)
println("Energy at mid:", E/(Nx*Ny), " supposed norm: ", N)
=#

# run heaviest functions one time to make Julia compile everything
Ls = buildLs(A, H; mindim=1, maxdim=chi)
@info "Built first Ls"
Rs = buildRs(A, H; mindim=1, maxdim=chi)
@info "Built first Rs"
A, Ls, Rs = rightwardSweep(A, Ls, Rs, H; sweep=0, mindim=chi, maxdim=chi, simple_update_cutoff=simple_update_cutoff)
A, Ls, Rs = leftwardSweep(A, Ls, Rs, H; sweep=0, mindim=chi, maxdim=chi, simple_update_cutoff=simple_update_cutoff)

# actual profiling run
(tL, tR), tS, bytes, gctime, memallocs = @timed doSweeps(A, Ls, Rs, H; mindim=chi, maxdim=chi, simple_update_cutoff=simple_update_cutoff, sweep_count=10)
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
#=col = Nx
println("building Ls")
L_s = buildLs(cA, H; mindim=chi, maxdim=chi)
println("building Rs")
R_s = buildRs(cA, H; mindim=chi, maxdim=chi)
dummyI = MPS(Ny, fill(cuITensor(1.0), Ny), 0, Ny+1)
dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny)) 
EAncEnvs = buildAncs(cA, L_s[col - 1], dummyEnv, H, col)
N, E     = measureEnergy(cA, L_s[col - 1], dummyEnv, EAncEnvs, H, 1, col)
println("Energy at mid:", E/(Nx*Ny), " supposed norm: ", N)=#

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
