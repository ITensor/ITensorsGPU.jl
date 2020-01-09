using Pkg
Pkg.activate("../..")
using Profile, TimerOutputs
include("peps.jl")
#=function doSweeps(A::PEPS, Ls::Vector{Environments}, Rs::Vector{Environments}, H; mindim::Int=1, maxdim::Int=1, simple_update_cutoff::Int=4, sweep_count::Int=10, cutoff::Float64=0.)
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
end=#

Nx  = 6
Ny  = 6
mdim = 4
io = open("prof_$(string(Nx))_$mdim.txt", "w+")
logger = SimpleLogger(io)
global_logger(logger)
J  = 1.0
sites = siteinds("S=1/2",Nx*Ny)
println("Beginning A")
A = checkerboardPEPS(sites, Nx, Ny, mindim=mdim)
cA = cuPEPS(A)
H  = makeCuH_XXZ(Nx, Ny, J)
for col in reverse(2:Nx)
    global cA
    cA = gaugeColumn(cA, col, :left; maxdim=mdim)
end
@info "Built cA and H"
Ls = buildLs(cA, H; mindim=mdim, maxdim=mdim)
@info "Built first Ls"
Rs = buildRs(cA, H; mindim=mdim, maxdim=mdim)
@info "Built first Rs"
cA, Ls, Rs = rightwardSweep(cA, Ls, Rs, H; mindim=mdim, maxdim=mdim)
cA, Ls, Rs = leftwardSweep(cA, Ls, Rs, H; mindim=mdim, maxdim=mdim)
doSweeps(cA, Ls, Rs, H; mindim=mdim, maxdim=mdim)
