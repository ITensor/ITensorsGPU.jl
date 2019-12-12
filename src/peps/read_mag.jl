using Pkg
Pkg.activate("../..")
using TimerOutputs
using ITensors, ITensorsGPU
using CuArrays
using Random, Logging, LinearAlgebra

include("peps.jl")
Nx = 6
Ny = 6
H = makeH_XXZ(Nx, Ny, 1.0)
mindim = 4
maxdim = 4

function doSweepsLoading(H; mindim::Int=1, maxdim::Int=1, simple_update_cutoff::Int=4, sweep_start::Int=1, sweep_count::Int=10, cutoff::Float64=0.)
    Nx = 6
    Ny = 6
    #mindim = 3
    #maxdim = 3
    #Ls, tL, bytes, gctime, memallocs = @timed buildLs(A, H; mindim=mindim, maxdim=maxdim)
    #Rs, tR, bytes, gctime, memallocs = @timed buildRs(A, H; mindim=mindim, maxdim=maxdim)
    for sweep in sweep_start:sweep_count
        io = open("Astart_$sweep.dat", "r")
        A_ = readcpp(io, Vector{Vector{ITensor}})
        A  = PEPS(Nx, Ny, hcat(A_...))
        close(io)
        #Ls = buildLs(A, H; mindim=mindim, maxdim=maxdim)
        #Rs = buildRs(A, H; mindim=mindim, maxdim=maxdim)
        Ls = Vector{Environments}(undef, Nx)
        Rs = Vector{Environments}(undef, Nx)
        for col in 1:Nx
            println("loading col $col")
            flush(stdout)
            if col > 1
                io = open("Rstart_I$(col - 1)_$sweep.dat", "r")
                RI = readcpp(io, Vector{ITensor})
                close(io)
                io = open("Rstart_H$(col - 1)_$sweep.dat", "r")
                RH = readcpp(io, Vector{ITensor})
                close(io)
                io = open("Rstart_IP$(col - 1)_$sweep.dat", "r")
                RIP = readcpp(io, Vector{Vector{ITensor}})
                close(io)
                R = Environments(MPS(Ny, RI), MPS(Ny, RH), hcat(RIP...))
                Rs[col] = R
            end
            if col < Nx
                io = open("Lstart_I$(col - 1)_$sweep.dat", "r")
                LI = readcpp(io, Vector{ITensor})
                close(io)
                io = open("Lstart_H$(col - 1)_$sweep.dat", "r")
                LH = readcpp(io, Vector{ITensor})
                close(io)
                io = open("Lstart_IP$(col - 1)_$sweep.dat", "r")
                LIP = readcpp(io, Vector{Vector{ITensor}})
                close(io)
                L = Environments(MPS(Ny, LI), MPS(Ny, LH), hcat(LIP...))
                Ls[col] = L
            end
        end
        if iseven(sweep)
            println("SWEEP RIGHT $sweep")
            A, Ls, Rs = rightwardSweep(A, Ls, Rs, H; sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
        else
            println("SWEEP LEFT $sweep")
            A, Ls, Rs = leftwardSweep(A, Ls, Rs, H; sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
            #=Ny, Nx = size(A)
            dummyI = MPS(Ny, fill(ITensor(1.0), Ny), 0, Ny+1)
            dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny)) 
            prev_cmb_l = Vector{ITensor}(undef, Ny)
            next_cmb_l = Vector{ITensor}(undef, Ny)
            @inbounds for col in reverse(2:Nx)
                R = col == Nx ? dummyEnv : Rs[col + 1]
                @debug "Sweeping col $col"
                #@timeit "sweep" begin
                #A = sweepColumn(A, Ls[col - 1], R, H, col; sweep=sweep, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
                #end
                io = open("Aup_$(col-1).dat", "r")
                A_ = readcpp(io, Vector{Vector{ITensor}})
                A  = PEPS(Nx, Ny, hcat(A_...))
                close(io)
                #A = gaugeColumn(A, col, :left; sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
                #=if col == Nx
                    right_H_terms = getDirectional(H[Nx - 1], Horizontal)
                    @timeit "right edge env" begin
                        Rs[col] = buildEdgeEnvironment(A, H, right_H_terms, prev_cmb_l, :right, Nx; sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
                    end
                else
                    @timeit "right next env" begin
                        I, H_, IP = buildNextEnvironment(A, Rs[col+1], H, prev_cmb_l, next_cmb_l, :right, col; sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
                        Rs[col]   = Environments(I, H_, IP)
                    end
                    prev_cmb_l = deepcopy(next_cmb_l)
                end=#
                #Rs = buildRs(A, H; sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
                # already at col 5 something is wrong with Rs.H specifically
                R_s               = Vector{Environments}(undef, Nx)
                for col_ in 1:Nx
                    println("loading col_ $col_")
                    flush(stdout)
                    if col_ > 1
                        io = open("Rup_I$(col_ - 1)_$(col - 1).dat", "r")
                        RI = readcpp(io, Vector{ITensor})
                        close(io)
                        io = open("Rup_H$(col_ - 1)_$(col - 1).dat", "r")
                        RH = readcpp(io, Vector{ITensor})
                        close(io)
                        io = open("Rup_IP$(col_ - 1)_$(col - 1).dat", "r")
                        RIP = readcpp(io, Vector{Vector{ITensor}})
                        close(io)
                        R = Environments(MPS(Ny, RI), MPS(Ny, RH), hcat(RIP...))
                        R_s[col_] = R
                    end
                    println("loaded col_ $col_")
                    flush(stdout)
                end
                previous_combiners = Vector{ITensor}(undef, Ny)
                next_combiners     = Vector{ITensor}(undef, Ny)
                start_col = col
                #=@show R_s[col].I[1]
                @show Rs[col].I[1]
                @show R_s[col].H[1]
                @show Rs[col].H[1]
                @show R_s[col].InProgress[1, 1]
                @show Rs[col].InProgress[1, 1]=#
                if col == Nx
                    right_H_terms = getDirectional(H[Nx-1], Horizontal)
                    upR        = buildEdgeEnvironment(A, H, right_H_terms, previous_combiners, :right, Nx;  sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
                    @show R_s[col].InProgress[1]
                    @show upR.InProgress[1]
                end
                if col < Nx
                    previous_combiners = ITensor[reconnect(commonindex(A[row, start_col], A[row, start_col + 1]), R_s[start_col+1].I[row]) for row in 1:Ny]
                    I_, H_, IP_ = buildNextEnvironment(A, R_s[col+1], H, previous_combiners, next_combiners, :right, col; sweep=sweep, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
                    upR   = Environments(I_, H_, IP_)
                    @show R_s[col].I[1]
                    @show upR.I[1]
                    @show R_s[col].H[1]
                    @show upR.H[1]
                    @show R_s[col].InProgress[1]
                    @show upR.InProgress[1]
                end
                Rs[col] = R_s[col]
            end=#
        end
        flush(stdout)
        if sweep == simple_update_cutoff - 1
            for col in 1:Nx-1
                A = gaugeColumn(A, col, :right; mindim=1, maxdim=chi)
            end
            Ls = buildLs(A, H; mindim=mindim, maxdim=maxdim)
            Rs = buildRs(A, H; mindim=mindim, maxdim=maxdim)
        end
    end
    return tL, tR
end

io = open("Apeps.dat", "r")
A_ = readcpp(io, Vector{Vector{ITensor}})
A = PEPS(Nx, Ny, hcat(A_...))
close(io)

n  = 35
As = MPO[]
for ii in 0:n
    io = open("A_$ii.dat", "r")
    A_ = readcpp(io, Vector{ITensor})
    A  = MPO(6, A_)
    close(io)
    push!(As, A)
    if ii == 0
        @show A[1]
    end
end
#=L = Environments(MPS(Ny, LI), MPS(Ny, LH), hcat(LIP...))
R = Environments(MPS(Ny, RI), MPS(Ny, RH), hcat(RIP...))
Ls = buildLs(A, H; mindim=mindim, maxdim=maxdim)
Rs = buildRs(A, H; mindim=mindim, maxdim=maxdim)=#
#=@show inner(L.I, Ls[Nx - 1].I)
for row in 1:Ny
    @show L.H[row]
    @show Ls[Nx - 1].H[row]
    println()
end=#
#doSweepsLoading(H; mindim=mindim, maxdim=maxdim, simple_update_cutoff=-1, sweep_start=3, sweep_count=9, cutoff=0.)

#=@show A[1, 1]
@show Ls[2].I[1]
@show LI[1]
@show Ls[2].H[1]
@show LH[1]
@show Ls[2].InProgress[1, 1]
@show LIP[1, 1]=#

#A = intraColumnGauge(A, Nx - 1; mindim=mindim, maxdim=maxdim, simple_update_cutoff=-1, sweep_count=7, cutoff=0.)
#io = open("Astart16.dat", "r")
#A_ = readcpp(io, Vector{Vector{ITensor}})
#A = PEPS(Nx, Ny, hcat(A_...))
#close(io)
#=AncEnvs = buildAncs(A, Ls[Nx - 1], dummyEnv, H, Nx)
Hs, N = buildLocalH(A, Ls[Nx - 1], dummyEnv, AncEnvs, H, 1, Nx, A[1, Nx]; verbose=true)
localH = sum(Hs)
init_N    = real(scalar(collect(N * dag(A[1,Nx])')))
println("Initial energy : $(scalar(localH*dag(A[1, Nx])')/(init_N*Nx*Ny)) and norm : $init_N")
mapper   = ITensorMap(A, H, Ls[Nx - 1], dummyEnv, AncEnvs, 1, Nx)
λ, new_A = davidson(mapper, A[1, Nx]; maxiter=2, mindim=mindim, maxdim=maxdim, simple_update_cutoff=-1, sweep_count=7, cutoff=0.)=#
#println(localH)
#@show λ
#Hs, N  = buildLocalH(A, Ls[Nx - 1], dummyEnv, AncEnvs, H, 1, Nx, new_A)
#localH = sum(Hs)
#@show norm(new_A)
#@show localH
#@show norm(localH - λ*dag(new_A)')
#N        = buildN(A, Ls[Nx - 1], dummyEnv, AncEnvs[:I], 1, Nx, new_A)
#new_N    = real(scalar(collect(N * dag(new_A)')))
#println("Optimized energy : $(λ/(new_N*Nx*Ny)) and norm : $new_N")
println()
println("all done!")
