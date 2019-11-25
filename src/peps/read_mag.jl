using Pkg
Pkg.activate("..")
using TimerOutputs
using ITensors, ITensors.CuITensors
using CuArrays
using Random, Logging, LinearAlgebra

include("peps.jl")
Nx = 6
Ny = 6
io = open("Astart.dat", "r")
A_ = read(io, Vector{Vector{ITensor}}; format="cpp")
A = PEPS(Nx, Ny, hcat(A_...))
close(io)
H = makeH_XXZ(Nx, Ny, 1.0)
mindim = 3
maxdim = 3
Ls = buildLs(A, H; mindim=mindim, maxdim=maxdim)
Rs = buildRs(A, H; mindim=mindim, maxdim=maxdim)
doSweeps(A, Ls, R, H; mindim=mindim, maxdim=maxdim, simple_update_cutoff=-1, sweep_count=7, cutoff=0.)
println()
println("all done!")
