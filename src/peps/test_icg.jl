using TimerOutputs, Statistics
include("peps.jl")
Nx  = 6
Ny  = 6

chi = 3
simple_update_cutoff = -1
Random.seed!(1234)
# Hamiltonian parameters
J  = 1.0
sites = spinHalfSites(Nx*Ny)
## CPU RUN
A = checkerboardPEPS(sites, Nx, Ny, mindim=chi)
H = makeH_XXZ(Nx, Ny, J)
@info "Built A and H"

Ls = buildLs(A, H; mindim=chi, maxdim=chi)
Rs = buildRs(A, H; mindim=chi, maxdim=chi)
col = div(Nx, 2)
EAncEnvs = buildAncs(A, Ls[col - 1], Rs[col + 1], H, col)
N, E = measureEnergy(A, Ls[col - 1], Rs[col + 1], EAncEnvs, H, 1, col)
println("Energy at mid: ", E/(Nx*Ny), " norm at mid: ", N)
flush(stdout)
for row in 1:Ny
    @show norm(A[row, col])
end
A = intraColumnGauge(A, col; mindim=chi, maxdim=chi)
for row in 1:Ny
    @show norm(A[row, col])
end
println("Building envs again")
Ls = buildLs(A, H; mindim=chi, maxdim=chi)
Rs = buildRs(A, H; mindim=chi, maxdim=chi)
col = div(Nx, 2)
EAncEnvs = buildAncs(A, Ls[col - 1], Rs[col + 1], H, col)
N, E = measureEnergy(A, Ls[col - 1], Rs[col + 1], EAncEnvs, H, 1, col)
println("Energy at mid: ", E/(Nx*Ny), " norm at mid: ", N)
flush(stdout)
