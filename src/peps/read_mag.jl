using Pkg
Pkg.activate("..")
using TimerOutputs
using ITensors, ITensors.CuITensors
using CuArrays
using Random, Logging, LinearAlgebra

include("peps.jl")
io = open("Aend.dat", "r")
A_  = read(io, Vector{Vector{ITensor}}; format="cpp")
A = PEPS(4, 4, hcat(A_...))
close(io)
H = makeH_XXZ(4, 4, 1.0)
mindim = 3
maxdim = 3
L_s = buildLs(A, H; mindim=mindim, maxdim=maxdim)
R_s = buildRs(A, H; mindim=mindim, maxdim=maxdim)
x_mag = measureXmag(A, L_s, R_s; mindim=mindim, maxdim=maxdim)
z_mag = measureZmag(A, L_s, R_s; mindim=mindim, maxdim=maxdim)
display(z_mag)
println()
v_mag = measureSmagVertical(A, L_s, R_s; mindim=mindim, maxdim=maxdim)
display(v_mag)
println()
h_mag = measureSmagHorizontal(A, L_s, R_s; mindim=mindim, maxdim=maxdim)
display(h_mag)
println()
println("all done!")
