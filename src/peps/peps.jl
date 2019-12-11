using ITensors, ITensorsGPU
using CuArrays
using Random, Logging, LinearAlgebra

import ITensors: tensors

mutable struct PEPS
    Nx::Int
    Ny::Int
    A_::AbstractMatrix{ITensor}

    PEPS() = new(0, 0, Matrix{ITensor}(),0,0)

    PEPS(Nx::Int, Ny::Int, A::Matrix{ITensor}) = new(Nx, Ny, A)
    function PEPS(sites, lattice::Lattice, Nx::Int, Ny::Int; mindim::Int=1, is_gpu::Bool=false)
        p  = Matrix{ITensor}(undef, Nx, Ny)
        right_links = [ Index(mindim, "Link,c$j,r$i,r") for i in 1:Nx, j in 1:Ny ]
        up_links    = [ Index(mindim, "Link,c$j,r$i,u") for i in 1:Nx, j in 1:Ny ]
        T           = is_gpu ? cuITensor : ITensor
        @inbounds for ii in eachindex(sites)
            row = div(ii-1, Nx) + 1
            col = mod(ii-1, Nx) + 1
            s   = sites[ii]
            if 1 < row < Ny && 1 < col < Nx 
                p[row, col] = T(right_links[row, col], up_links[row, col], right_links[row, col-1], up_links[row-1, col], s)
            elseif row == 1 && 1 < col < Nx
                p[row, col] = T(right_links[row, col], up_links[row, col], right_links[row, col-1], s)
            elseif 1 < row < Ny && col == 1
                p[row, col] = T(right_links[row, col], up_links[row, col], up_links[row-1, col], s)
            elseif row == Ny && 1 < col < Nx 
                p[row, col] = T(right_links[row, col], right_links[row, col-1], up_links[row-1, col], s)
            elseif 1 < row < Ny && col == Nx 
                p[row, col] = T(up_links[row, col], right_links[row, col-1], up_links[row-1, col], s)
            elseif row == Ny && col == 1 
                p[row, col] = T(right_links[row, col], up_links[row-1, col], s)
            elseif row == Ny && col == Nx 
                p[row, col] = T(right_links[row, col-1], up_links[row-1, col], s)
            elseif row == 1 && col == 1 
                p[row, col] = T(right_links[row, col], up_links[row, col], s)
            elseif row == 1 && col == Nx
                p[row, col] = T(up_links[row, col], right_links[row, col-1], s)
            end
        end
        new(Nx, Ny, p)
    end
end
Base.eltype(A::PEPS) = eltype(A.A_[1,1])
include("environments.jl")
include("ancillaries.jl")
include("gauge.jl")
include("observables.jl")

function cudelt(left::Index, right::Index)
    d_data   = CuArrays.zeros(Float64, dim(left), dim(right))
    ddi = diagind(d_data, 0)
    d_data[ddi] = 1.0
    delt = cuITensor(vec(d_data), left, right)
    return delt
end

function mydelt(left::Index, right::Index)
    d_data   = zeros(Float64, dim(left), dim(right))
    ddi = diagind(d_data, 0)
    d_data[ddi] = ones(Float64, length(ddi))
    delt = ITensor(vec(d_data), left, right)
    return delt
end

function checkerboardPEPS(sites, Nx::Int, Ny::Int; mindim::Int=1)
    lattice = squareLattice(Nx,Ny,yperiodic=false)
    A = PEPS(sites, lattice, Nx, Ny, mindim=mindim)
    @inbounds for ii ∈ eachindex(sites)
        row = div(ii-1, Nx) + 1
        col = mod(ii-1, Nx) + 1
        spin_side = isodd(row - 1) ⊻ isodd(col - 1) ? 2 : 1
        si  = findindex(A[ii], "Site") 
        lis = findinds(A[ii], "Link") 
        ivs = [li(1) for li in lis]
        ivs = vcat(ivs, si(spin_side))
        A[ii][ivs...] = 1.0
    end
    for row in 1:Ny, col in 1:Nx
        A[row, col] += randomITensor(inds(A[row, col]))/10.0
    end
    return A
end

function randomPEPS(sites, Nx::Int, Ny::Int; mindim::Int=1)
    lattice = squareLattice(Nx,Ny,yperiodic=false)
    A = PEPS(sites, lattice, Nx, Ny, mindim=mindim)
    @inbounds for ii ∈ eachindex(sites)
        randn!(A[ii])
        normalize!(A[ii])
    end
    return A
end

function randomCuPEPS(sites, Nx::Int, Ny::Int; mindim::Int=1)
    lattice = squareLattice(Nx,Ny,yperiodic=false)
    A = PEPS(sites, lattice, Nx, Ny; mindim=mindim, is_gpu=true)
    @inbounds for ii ∈ eachindex(sites)
        randn!(A[ii])
        normalize!(A[ii])
    end
    return A
end

function cuPEPS(A::PEPS)
    cA = copy(A)
    @inbounds for i in 1:Nx, j in 1:Ny
        cA[i, j] = cuITensor(A[i, j])
    end
    return cA
end

function Base.collect(cA::PEPS)
    A = copy(cA)
    @inbounds for i in 1:Nx, j in 1:Ny
        A[i, j] = collect(cA[i, j])
    end
    return A
end

tensors(A::PEPS)   = A.A_
Base.size(A::PEPS) = (A.Ny, A.Nx)

Base.getindex(A::PEPS, i::Integer, j::Integer) = getindex(tensors(A), i, j)::ITensor
Base.getindex(A::PEPS, ::Colon,    j::Integer) = getindex(tensors(A), :, j)::Vector{ITensor}
Base.getindex(A::PEPS, i::Integer, ::Colon)    = getindex(tensors(A), i, :)::Vector{ITensor}
Base.getindex(A::PEPS, i::Integer)             = getindex(tensors(A), i)::ITensor

Base.setindex!(A::PEPS, val::ITensor, i::Integer, j::Integer)       = setindex!(tensors(A), val, i, j)
Base.setindex!(A::PEPS, vals::Vector{ITensor}, ::Colon, j::Integer) = setindex!(tensors(A), vals, :, j)
Base.setindex!(A::PEPS, vals::Vector{ITensor}, i::Integer, ::Colon) = setindex!(tensors(A), vals, i, :)

Base.copy(A::PEPS)    = PEPS(A.Nx, A.Ny, copy(tensors(A)))
Base.similar(A::PEPS) = PEPS(A.Nx, A.Ny, similar(tensors(A)))

function Base.show(io::IO, A::PEPS)
  print(io,"PEPS")
  (size(A)[1] > 0 && size(A)[2] > 0) && print(io,"\n")
  @inbounds for i in 1:A.Nx, j in 1:A.Ny
      println(io,"$i $j $(A[i,j])")
  end
end

@enum Op_Type Field=0 Vertical=1 Horizontal=2
struct Operator
    sites::Vector{Pair{Int,Int}}
    ops::Vector{ITensor}
    site_ind::Index
    dir::Op_Type
end

getDirectional(ops::Vector{Operator}, dir::Op_Type) = collect(filter(x->x.dir==dir, ops))

function spinI(s::Index; is_gpu::Bool=false)::ITensor
    I_data      = is_gpu ? CuArrays.zeros(Float64, dim(s), dim(s)) : zeros(Float64, dim(s), dim(s))
    idi         = diagind(I_data, 0)
    I_data[idi] = is_gpu ? CuArrays.ones(Float64, dim(s)) : ones(Float64, dim(s))
    I           = is_gpu ? cuITensor( vec(I_data), IndexSet(s, s') ) : ITensor(vec(I_data), IndexSet(s, s'))
    return I
end

function makeH_XXZ(Nx::Int, Ny::Int, J::Real; pinning::Bool=false)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    P = ITensor(s, s')
    M = ITensor(s, s')
    P[s(1), s'(2)] = 1.0
    M[s(2), s'(1)] = 1.0
    H = Matrix{Vector{Operator}}(undef, Ny, Nx)
    @inbounds for col in 1:Nx, row in 1:Ny
        H[row, col] = Vector{Operator}()
        if row < Ny
            op_a  = 0.5 * P
            op_b  = copy(M)
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))

            op_a  = 0.5 * M
            op_b  = copy(P)
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))

            if J != 0.0
                op_a  = J * Z
                op_b  = copy(Z)
                sites = [row=>col, row+1=>col]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))
            end
        end
        if col < Nx
            op_a  = 0.5 * P
            op_b  = copy(M)
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))

            op_a  = 0.5 * M
            op_b  = copy(P)
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))

            if J != 0.0
                op_a  = J * Z
                op_b  = copy(Z)
                sites = [row=>col, row=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))
            end
        end
    end
    # pinning fields
    J_ = 1.;
    @inbounds for row in 1:Ny
        op = isodd(row-1) ? (J_/2.0) * Z : (-J_/2.0) * Z   
        push!(H[row, 1], Operator([row=>1], [op], s, Field))    
        op = isodd(row-1) ? (-J_/2.0) * Z : (J_/2.0) * Z   
        push!(H[row, Nx], Operator([row=>Nx], [op], s, Field))    
    end
    @inbounds for col in 2:Nx-1
        op = isodd(col-1) ? (-J_/2.0) * Z : (J_/2.0) * Z   
        push!(H[Ny, col], Operator([Ny=>col], [op], s, Field))    
        op = isodd(col-1) ? (J_/2.0) * Z : (-J_/2.0) * Z   
        push!(H[1, col], Operator([1=>col], [op], s, Field))    
    end
    return H
end

function makeCuH_XXZ(Nx::Int, Ny::Int, J::Real; pinning::Bool=false)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    #Ident = spinI(s)
    P = ITensor(s, s')
    M = ITensor(s, s')
    P[s(1), s'(2)] = 1.0
    M[s(2), s'(1)] = 1.0
    H = Matrix{Vector{Operator}}(undef, Ny, Nx)
    @inbounds for col in 1:Nx, row in 1:Ny
        H[row, col] = Vector{Operator}()
        if row < Ny
            op_a  = 0.5 * P
            op_b  = copy(M)
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Vertical))

            op_a  = 0.5 * M
            op_b  = copy(P)
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Vertical))

            if J != 0.0
                op_a  = J * Z
                op_b  = copy(Z)
                sites = [row=>col, row+1=>col]
                push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Vertical))
            end
        end
        if col < Nx
            op_a  = 0.5 * P
            op_b  = copy(M)
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Horizontal))

            op_a  = 0.5 * M
            op_b  = copy(P)
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Horizontal))

            if J != 0.0
                op_a  = J * Z
                op_b  = copy(Z)
                sites = [row=>col, row=>col+1]
                push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Horizontal))
            end
        end
    end
    # pinning fields
    J_ = 1.
    @inbounds for row in 1:Ny
        op = isodd(row-1) ? (J_/2.0) * Z : (-J_/2.0) * Z   
        push!(H[row, 1], Operator([row=>1], [cuITensor(op)], s, Field))    
        op = isodd(row-1) ? (-J_/2.0) * Z : (J_/2.0) * Z   
        push!(H[row, Nx], Operator([row=>Nx], [cuITensor(op)], s, Field))    
    end
    @inbounds for col in 2:Nx-1
        op = isodd(col-1) ? (-J_/2.0) * Z : (J_/2.0) * Z   
        push!(H[Ny, col], Operator([Ny=>col], [cuITensor(op)], s, Field))    
        op = isodd(col-1) ? (J_/2.0) * Z : (-J_/2.0) * Z   
        push!(H[1, col], Operator([1=>col], [cuITensor(op)], s, Field))    
    end
    return H
end


function combine(Aorig::ITensor, Anext::ITensor, tags::String)::ITensor
    ci  = commonindex(Aorig, Anext)
    cmb,cmbi = combiner(IndexSet(ci, prime(ci)), tags=tags)
    return cmb
end

function reconnect(combiner_ind::Index, environment::ITensor)
    environment_combiner        = findindex(environment, "Site")
    new_combiner, combined_ind  = combiner(IndexSet(combiner_ind, prime(combiner_ind)), tags="Site")
    combiner_transfer           = δ(combined_ind, environment_combiner)
    #return new_combiner*combiner_transfer
    replaceindex!(new_combiner, combined_ind, environment_combiner)
    return new_combiner
end

function buildN(A::PEPS, L::Environments, R::Environments, IEnvs, row::Int, col::Int, ϕ::ITensor)::ITensor
    Ny, Nx   = size(A)
    is_gpu   = !(data(store(A[1,1])) isa Array)
    N        = spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
    workingN = N
    if row > 1
        workingN *= IEnvs[:below][row - 1]
    end
    if row < Ny
        workingN *= IEnvs[:above][end - row]
    end
    if col > 1
        ci = commonindex(A[row, col], A[row, col-1])
        workingN *= multiply_side_ident(A[row, col], ci, copy(L.I[row])) 
    end
    workingN *= ϕ
    if col < Nx
        ci = commonindex(A[row, col], A[row, col+1])
        workingN *= multiply_side_ident(A[row, col], ci, copy(R.I[row])) 
    end
    return workingN
end


function multiply_side_ident(A::ITensor, ci::Index, side_I::ITensor)
    is_gpu      = !(data(store(A)) isa Array)
    scmb        = findindex(side_I, "Site")
    acmb, acmbi = combiner(IndexSet(ci, ci'), tags="Site")
    replaceindex!(acmb, acmbi, scmb)
    msi         = side_I * acmb
    return msi
end

function nonWorkRow(A::PEPS, L::Environments, R::Environments, H::Operator, row::Int, col::Int)::ITensor
    Ny, Nx  = size(A)
    op_rows = H.sites
    is_gpu  = !(data(store(A[1,1])) isa Array)
    ops     = deepcopy(H.ops)
    @inbounds for op_ind in 1:length(ops)
        as = findindex(A[op_rows[op_ind][1][1], col], "Site")
        ops[op_ind] = replaceindex!(ops[op_ind], H.site_ind, as)
        ops[op_ind] = replaceindex!(ops[op_ind], H.site_ind', as')
    end
    op = spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
    op_ind = findfirst( x -> x == row, op_rows)
    AA = A[row, col] * op * dag(A[row, col])'
    if col > 1
        ci = commonindex(A[row, col], A[row, col-1])
        msi = multiply_side_ident(A[row, col], ci, L.I[row])
        AA *= msi
    end
    if col < Nx
        ci = commonindex(A[row, col], A[row, col+1])
        msi = multiply_side_ident(A[row, col], ci, R.I[row])
        AA *= msi 
    end
    return AA
end

function sum_rows_in_col(A::PEPS, L::Environments, R::Environments, H::Operator, row::Int, col::Int, low_row::Int, high_row::Int, above::Bool, IA::ITensor, IB::ITensor)::ITensor
    Ny, Nx = size(A)
    op_rows = H.sites
    is_gpu = !(data(store(A[1,1])) isa Array)
    ops = deepcopy(H.ops)
    start_row_ = row == op_rows[1][1] ? low_row + 1 : high_row
    stop_row_ = row == op_rows[1][1] ? high_row : low_row + 1
    start_row_ = min(start_row_, Ny)
    stop_row_ = min(stop_row_, Ny)
    step_row = row == op_rows[1][1] ? 1 : -1;
    start_row, stop_row = minmax(start_row_, stop_row_)
    op_row_a = H.sites[1][1]
    op_row_b = H.sites[2][1]
    @inbounds for op_ind in 1:length(ops)
        this_A = A[op_rows[op_ind][1][1], col]
        as = findindex(this_A, "Site")
        ops[op_ind] = replaceindex!(ops[op_ind], H.site_ind, as)
        ops[op_ind] = replaceindex!(ops[op_ind], H.site_ind', as')
    end
    nwrs  = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    nwrs_ = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    Hterm = ITensor()
    if row == op_row_a
        Hterm = IA
        Hterm *= nonWorkRow(A, L, R, H, op_row_b, col)
        if col > 1
            ci  = commonindex(A[row, col], A[row, col-1])
            msi = multiply_side_ident(A[row, col], ci, copy(L.I[row]))
            Hterm *= msi 
        end
        if col < Nx
            ci  = commonindex(A[row, col], A[row, col+1])
            msi = multiply_side_ident(A[row, col], ci, copy(R.I[row]))
            Hterm *= msi
        end
        Hterm *= IB
    else
        Hterm = IB
        Hterm *= nonWorkRow(A, L, R, H, op_row_a, col)
        if col > 1
            ci  = commonindex(A[row, col], A[row, col-1])
            msi = multiply_side_ident(A[row, col], ci, copy(L.I[row]))
            Hterm *= msi 
        end
        if col < Nx
            ci  = commonindex(A[row, col], A[row, col+1])
            msi = multiply_side_ident(A[row, col], ci, copy(R.I[row]))
            Hterm *= msi
        end
        Hterm *= IA
    end
    op = spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
    op_ind = findfirst( x -> x[1] == row, op_rows)
    if op_ind > 0 
        op = ops[op_ind]
    end
    Hterm = Hterm*op
    return Hterm
end

function buildHIedge( A::PEPS, E::Environments, row::Int, col::Int, side::Symbol, ϕ::ITensor )
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    HI = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    IH = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    next_col = side == :left ? 2 : Nx - 1
    @inbounds for work_row in 1:row-1
        AA = A[work_row, col] * dag(prime(A[work_row, col], "Link"))
        ci = commonindex(A[work_row, col], A[work_row, next_col])
        cmb = findindex(E.I[work_row], "Site")
        acmb, acmbi = combiner(IndexSet(ci, ci'), tags="Site")
        replaceindex!(acmb, acmbi, cmb)
        AA *= acmb
        HI *= AA * E.I[work_row]
        IH *= E.H[work_row] * AA
    end
    ci  = commonindex(A[row, col], A[row, next_col])
    cmb = findindex(E.I[row], "Site")
    acmb, acmbi = combiner(IndexSet(ci, ci'), tags="Site")
    replaceindex!(acmb, acmbi, cmb)
    op = spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
    op = is_gpu ? cuITensor(op) : op
    HI *= ϕ
    IH *= ϕ
    HI *= op
    IH *= op
    HI *= E.I[row] * acmb
    IH *= E.H[row] * acmb
    @inbounds for work_row in row+1:Ny
        AA = A[work_row, col] * dag(prime(A[work_row, col], "Link"))
        ci = commonindex(A[work_row, col], A[work_row, next_col])
        cmb = findindex(E.I[work_row], "Site")
        acmb, acmbi = combiner(IndexSet(ci, ci'), tags="Site")
        replaceindex!(acmb, acmbi, cmb)
        AA *= acmb 
        HI *= AA * E.I[work_row]
        IH *= E.H[work_row] * AA
    end
    #AAinds = IndexSet(inds(A[row, col]), inds(A[row, col]'))
    AAinds = IndexSet(prime(ϕ))
    @assert hasinds(inds(IH), AAinds)
    @assert hasinds(AAinds, inds(IH))
    return [IH]
end

function buildHIs(A::PEPS, L::Environments, R::Environments, row::Int, col::Int, ϕ::ITensor)
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    col == 1  && return buildHIedge(A, R, row, col, :left, ϕ)
    col == Nx && return buildHIedge(A, L, row, col, :right, ϕ)
    HLI_a = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    HLI_b = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    IHR_a = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    IHR_b = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    HLI   = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    IHR   = is_gpu ? cuITensor(1.0) : ITensor(1.0)
    @inbounds for work_row in 1:row-1
        AA = A[work_row, col] * dag(prime(A[work_row, col], "Link"))
        lci = commonindex(A[work_row, col], A[work_row, col-1])
        rci = commonindex(A[work_row, col], A[work_row, col+1])
        lcmb = findindex(L.I[work_row], "Site")
        rcmb = findindex(R.I[work_row], "Site")
        lacmb, lacmbi = combiner(IndexSet(lci, lci'), tags="Site")
        racmb, racmbi = combiner(IndexSet(rci, rci'), tags="Site")
        replaceindex!(lacmb, lacmbi, lcmb)
        replaceindex!(racmb, racmbi, rcmb)
        AA *= lacmb
        AA *= racmb
        HLI_b *= L.H[work_row] * AA * R.I[work_row]
        IHR_b *= L.I[work_row] * AA * R.H[work_row]
    end
    lci = commonindex(A[row, col], A[row, col-1])
    rci = commonindex(A[row, col], A[row, col+1])
    lcmb = findindex(L.I[row], "Site")
    rcmb = findindex(R.I[row], "Site")
    lacmb, lacmbi = combiner(IndexSet(lci, lci'), tags="Site")
    racmb, racmbi = combiner(IndexSet(rci, rci'), tags="Site")
    replaceindex!(lacmb, lacmbi, lcmb)
    replaceindex!(racmb, racmbi, rcmb)
    HLI  *= L.H[row] * lacmb
    HLI  *= ϕ
    HLI  *= R.I[row] * racmb
    IHR  *= L.I[row] * lacmb 
    IHR  *= ϕ
    IHR  *= R.H[row] * racmb 
    op = spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
    HLI *= op
    IHR *= op
    @inbounds for work_row in reverse(row+1:Ny)
        AA = A[work_row, col] * dag(prime(A[work_row, col], "Link"))
        lci = commonindex(A[work_row, col], A[work_row, col-1])
        rci = commonindex(A[work_row, col], A[work_row, col+1])
        lcmb = findindex(L.I[work_row], "Site")
        rcmb = findindex(R.I[work_row], "Site")
        lacmb, lacmbi = combiner(IndexSet(lci, lci'), tags="Site")
        racmb, racmbi = combiner(IndexSet(rci, rci'), tags="Site")
        replaceindex!(lacmb, lacmbi, lcmb)
        replaceindex!(racmb, racmbi, rcmb)
        AA *= lacmb
        AA *= racmb
        HLI_a *= L.H[work_row] * AA * R.I[work_row]
        IHR_a *= L.I[work_row] * AA * R.H[work_row]
    end
    HLI *= HLI_a
    HLI *= HLI_b
    IHR *= IHR_a
    IHR *= IHR_b
    #AAinds = IndexSet(inds(A[row, col]), inds(A[row, col]'))
    AAinds = IndexSet(prime(ϕ))
    @assert hasinds(inds(IHR), AAinds)
    @assert hasinds(inds(HLI), AAinds)
    @assert hasinds(AAinds, inds(IHR))
    @assert hasinds(AAinds, inds(HLI))
    return [HLI, IHR]
end

function verticalTerms(A::PEPS, L::Environments, R::Environments, AI, AV, H, row::Int, col::Int, ϕ::ITensor)::Vector{ITensor} 
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    vTerms = ITensor[]#fill(ITensor(), length(H))
    #AAinds = IndexSet(inds(A[row, col]), inds(A[row, col]'))
    AAinds = IndexSet(prime(ϕ))
    dummy  = is_gpu ? cuITensor(1.0) : ITensor(1.0) 
    @inbounds for opcode in 1:length(H)
        thisVert = dummy 
        op_row_a = H[opcode].sites[1][1]
        op_row_b = H[opcode].sites[2][1]
        if op_row_b < row || row < op_row_a
            local V, I
            if op_row_a > row
                V = AV[:above][opcode][end - row]
                I = row > 1 ? AI[:below][row - 1] : dummy 
            elseif op_row_b < row
                V = AV[:below][opcode][row - op_row_b]
                I = row < Ny ? AI[:above][end - row] : dummy
            end
            thisVert *= V
            if col > 1
                ci = commonindex(A[row, col], A[row, col-1])
                msi = multiply_side_ident(A[row, col], ci, copy(L.I[row]))
                thisVert *= msi
            end
            thisVert *= ϕ
            if col < Nx
                ci = commonindex(A[row, col], A[row, col+1])
                msi = multiply_side_ident(A[row, col], ci, copy(R.I[row]))
                thisVert *= msi 
            end
            thisVert *= I
            thisVert *= spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
        elseif row == op_row_a
            low_row  = op_row_a - 1
            high_row = op_row_b
            AIL = low_row > 0 ? AI[:below][low_row] : dummy 
            AIH = high_row < Ny ? AI[:above][end - high_row] : dummy 
            thisVert = AIH
            if col > 1
                ci  = commonindex(A[op_row_b, col], A[op_row_b, col-1])
                msi = multiply_side_ident(A[op_row_b, col], ci, copy(L.I[op_row_b]))
                thisVert *= msi 
            end
            thisVert *= ϕ
            if col < Nx
                ci  = commonindex(A[op_row_b, col], A[op_row_b, col+1])
                msi = multiply_side_ident(A[op_row_b, col], ci, copy(R.I[op_row_b]))
                thisVert *= msi
            end
            sA = findindex(A[op_row_a, col], "Site")
            op_a = replaceindex!(copy(H[opcode].ops[1]), H[opcode].site_ind, sA)
            op_a = replaceindex!(op_a, H[opcode].site_ind', sA')
            sB = findindex(A[op_row_b, col], "Site")
            op_b = replaceindex!(copy(H[opcode].ops[2]), H[opcode].site_ind, sB)
            op_b = replaceindex!(op_b, H[opcode].site_ind', sB')
            thisVert *= A[op_row_b, col] * op_b * dag(A[op_row_b, col])'
            if col > 1
                ci  = commonindex(A[op_row_a, col], A[op_row_a, col-1])
                msi = multiply_side_ident(A[op_row_a, col], ci, copy(L.I[op_row_a]))
                thisVert *= msi 
            end
            if col < Nx
                ci  = commonindex(A[op_row_a, col], A[op_row_a, col+1])
                msi = multiply_side_ident(A[op_row_a, col], ci, copy(R.I[op_row_a]))
                thisVert *= msi
            end
            thisVert *= AIL 
            thisVert *= op_a
        elseif row == op_row_b
            low_row  = op_row_a - 1
            high_row = op_row_b
            AIL = low_row > 0 ? AI[:below][low_row] : dummy 
            AIH = high_row < Ny ? AI[:above][end - high_row] : dummy 
            thisVert = AIL
            if col > 1
                ci  = commonindex(A[op_row_a, col], A[op_row_a, col-1])
                msi = multiply_side_ident(A[op_row_a, col], ci, copy(L.I[op_row_a]))
                thisVert *= msi 
            end
            if col < Nx
                ci  = commonindex(A[op_row_a, col], A[op_row_a, col+1])
                msi = multiply_side_ident(A[op_row_a, col], ci, copy(R.I[op_row_a]))
                thisVert *= msi
            end
            sA = findindex(A[op_row_a, col], "Site")
            op_a = replaceindex!(copy(H[opcode].ops[1]), H[opcode].site_ind, sA)
            op_a = replaceindex!(op_a, H[opcode].site_ind', sA')
            sB = findindex(A[op_row_b, col], "Site")
            op_b = replaceindex!(copy(H[opcode].ops[2]), H[opcode].site_ind, sB)
            op_b = replaceindex!(op_b, H[opcode].site_ind', sB')
            thisVert *= A[op_row_a, col] * op_a * dag(A[op_row_a, col])'
            if col > 1
                ci  = commonindex(A[op_row_b, col], A[op_row_b, col-1])
                msi = multiply_side_ident(A[op_row_b, col], ci, copy(L.I[op_row_b]))
                thisVert *= msi 
            end
            thisVert *= ϕ
            if col < Nx
                ci  = commonindex(A[op_row_b, col], A[op_row_b, col+1])
                msi = multiply_side_ident(A[op_row_b, col], ci, copy(R.I[op_row_b]))
                thisVert *= msi
            end
            thisVert *= AIH 
            thisVert *= op_b
        end
        @assert hasinds(inds(thisVert), AAinds) "inds of thisVert and AAinds differ!\n$(inds(thisVert))\n$AAinds\n"
        @assert hasinds(AAinds, inds(thisVert)) "inds of thisVert and AAinds differ!\n$(inds(thisVert))\n$AAinds\n"
        if hasinds(AAinds, inds(thisVert)) && hasinds(inds(thisVert), AAinds)
            push!(vTerms, thisVert)
        end
    end
    return vTerms
end

function fieldTerms(A::PEPS, L::Environments, R::Environments, AI, AF, H, row::Int, col::Int, ϕ::ITensor)::Vector{ITensor} 
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    fTerms = Vector{ITensor}(undef, length(H))
    dummy  = is_gpu ? cuITensor(1.0) : ITensor(1.0) 
    #AAinds = IndexSet(inds(A[row, col]), inds(A[row, col]'))
    AAinds = IndexSet(prime(ϕ))
    @inbounds for opcode in 1:length(H)
        thisField = dummy 
        op_row    = H[opcode].sites[1][1]
        if op_row != row
            local F, I
            if op_row > row
                F = AF[:above][opcode][end - row]
                I = row > 1 ? AI[:below][row - 1] : dummy
            else
                F = AF[:below][opcode][row - 1]
                I = row < Ny ? AI[:above][end - row] : dummy 
            end
            thisField *= F
            if col > 1
                ci = commonindex(A[row, col], A[row, col-1])
                thisField *= multiply_side_ident(A[row, col], ci, copy(L.I[row]))
            end
            thisField *= ϕ
            if col < Nx
                ci = commonindex(A[row, col], A[row, col+1])
                thisField *= multiply_side_ident(A[row, col], ci, copy(R.I[row]))
            end
            thisField *= I
            thisField *= spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
        else
            low_row  = op_row - 1
            high_row = op_row
            AIL = low_row > 0   ? AI[:below][low_row]        : dummy 
            AIH = high_row < Ny ? AI[:above][end - high_row] : dummy 
            thisField = AIL
            if col > 1
                ci  = commonindex(A[row, col], A[row, col-1])
                msi = multiply_side_ident(A[row, col], ci, copy(L.I[row]))
                thisField *= msi 
            end
            thisField *= ϕ
            if col < Nx
                ci  = commonindex(A[row, col], A[row, col+1])
                msi = multiply_side_ident(A[row, col], ci, copy(R.I[row]))
                thisField *= msi
            end
            thisField *= AIH
            sA = findindex(A[row, col], "Site")
            op = copy(H[opcode].ops[1])
            op = replaceindex!(op, H[opcode].site_ind, sA) 
            op = replaceindex!(op, H[opcode].site_ind', sA')
            thisField *= op
        end
        @assert hasinds(inds(thisField), AAinds)
        @assert hasinds(AAinds, inds(thisField))
        fTerms[opcode] = thisField;
    end
    return fTerms
end

function connectLeftTerms(A::PEPS, L::Environments, R::Environments, AI, AL, H, row::Int, col::Int, ϕ::ITensor)::Vector{ITensor} 
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    lTerms = Vector{ITensor}(undef, length(H))
    #AAinds = IndexSet(inds(A[row, col]), inds(A[row, col]'))
    AAinds = IndexSet(prime(ϕ))
    dummy  = is_gpu ? cuITensor(1.0) : ITensor(1.0) 
    @inbounds for opcode in 1:length(H)
        op_row_b = H[opcode].sites[2][1]
        op_b = copy(H[opcode].ops[2])
        as   = findindex(A[op_row_b, col], "Site")
        op_b = replaceindex!(op_b, H[opcode].site_ind, as)
        op_b = replaceindex!(op_b, H[opcode].site_ind', as')
        thisHori = is_gpu ? cuITensor(1.0) : ITensor(1.0)
        if op_row_b != row
            local ancL, I
            if op_row_b > row
                ancL = AL[:above][opcode][end - row]
                I = row > 1 ? AL[:below][opcode][row - 1] : dummy 
            else
                ancL = AL[:below][opcode][row - 1]
                I = row < Ny ? AL[:above][opcode][end - row] : dummy
            end
            thisHori = ancL
            thisHori *= L.InProgress[row, opcode]
            thisHori *= ϕ
            if col < Nx
                ci = commonindex(A[row, col], A[row, col+1])
                thisHori *= multiply_side_ident(A[row, col], ci, copy(R.I[row]))
            end
            thisHori *= I
            thisHori *= spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
        else
            low_row = (op_row_b <= row) ? op_row_b - 1 : row - 1;
            high_row = (op_row_b >= row) ? op_row_b + 1 : row + 1;
            if low_row >= 1
                thisHori *= AL[:below][opcode][low_row]
            end
            if high_row <= Ny
                thisHori *= AL[:above][opcode][end - high_row + 1]
            end
            if col < Nx
                ci = commonindex(A[row, col], A[row, col+1])
                thisHori *= multiply_side_ident(A[row, col], ci, R.I[row])
            end
            uih = uniqueinds(thisHori, L.InProgress[row, opcode])
            uil = uniqueinds(L.InProgress[row, opcode], thisHori)
            thisHori *= L.InProgress[row, opcode]
            thisHori *= ϕ
            thisHori *= op_b
        end
        @assert hasinds(inds(thisHori), AAinds)
        @assert hasinds(AAinds, inds(thisHori))
        lTerms[opcode] = thisHori
    end
    return lTerms
end

function connectRightTerms(A::PEPS, L::Environments, R::Environments, AI, AR, H, row::Int, col::Int, ϕ::ITensor)::Vector{ITensor} 
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    rTerms = Vector{ITensor}(undef, length(H))
    #AAinds = IndexSet(inds(A[row, col]), inds(A[row, col]'))
    AAinds = IndexSet(prime(ϕ))
    dummy  = is_gpu ? cuITensor(1.0) : ITensor(1.0) 
    @inbounds for opcode in 1:length(H)
        op_row_a = H[opcode].sites[1][1]
        op_a = copy(H[opcode].ops[1])
        as   = findindex(A[op_row_a, col], "Site")
        op_a = replaceindex!(op_a, H[opcode].site_ind, as)
        op_a = replaceindex!(op_a, H[opcode].site_ind', as')
        thisHori = is_gpu ? cuITensor(1.0) : ITensor(1.0)
        if op_row_a != row
            local ancR, I
            if op_row_a > row
                ancR = AR[:above][opcode][end - row]
                I = row > 1 ? AR[:below][opcode][row - 1] : dummy 
            else
                ancR = AR[:below][opcode][row - 1]
                I = row < Ny ? AR[:above][opcode][end - row] : dummy 
            end
            thisHori = ancR
            thisHori *= R.InProgress[row, opcode]
            thisHori *= ϕ
            if col > 1
                ci = commonindex(A[row, col], A[row, col-1])
                thisHori *= multiply_side_ident(A[row, col], ci, copy(L.I[row]))
            end
            thisHori *= I
            thisHori *= spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
        else
            low_row = (op_row_a <= row) ? op_row_a - 1 : row - 1;
            high_row = (op_row_a >= row) ? op_row_a + 1 : row + 1;
            if low_row >= 1
                thisHori *= AR[:below][opcode][low_row]
            end
            if high_row <= Ny
                thisHori *= AR[:above][opcode][end - high_row + 1]
            end
            if col > 1
                ci = commonindex(A[row, col], A[row, col-1])
                thisHori *= multiply_side_ident(A[row, col], ci, L.I[row])
            end
            thisHori *= R.InProgress[row, opcode]
            thisHori *= ϕ
            thisHori *= op_a 
        end
        @assert hasinds(inds(thisHori), AAinds)
        @assert hasinds(AAinds, inds(thisHori))
        rTerms[opcode] = thisHori
    end
    return rTerms
end

function buildLocalH(A::PEPS, L::Environments, R::Environments, AncEnvs, H, row::Int, col::Int, ϕ::ITensor; verbose::Bool=false)
    field_H_terms = getDirectional(vcat(H[:, col]...), Field)
    vert_H_terms  = getDirectional(vcat(H[:, col]...), Vertical)
    term_count    = 1 + length(field_H_terms) + length(vert_H_terms)
    Ny, Nx        = size(A)
    @timeit "build Ns" begin
        N   = buildN(A, L, R, AncEnvs[:I], row, col, ϕ)
    end
    den = scalar(collect(N * dag(ϕ)'))
    local left_H_terms, right_H_terms
    if col > 1
        left_H_terms = getDirectional(vcat(H[:, col - 1]...), Horizontal)
        term_count += length(left_H_terms)
    end
    if col < Nx
        right_H_terms = getDirectional(vcat(H[:, col]...), Horizontal)
        term_count += length(right_H_terms)
    end
    if 1 < col < Nx
        term_count += 1
    end
    Hs = Vector{ITensor}(undef, term_count)
    term_counter = 1
    @debug "\t\tBuilding I*H and H*I row $row col $col"
    @timeit "build HIs" begin
        HIs = buildHIs(A, L, R, row, col, ϕ)
        Hs[term_counter:term_counter+length(HIs)-1] = HIs
        term_counter += length(HIs)
        if verbose
            println("--- HI TERMS ---")
            for HI in HIs
                println(scalar(HI * dag(ϕ)'))
            end
        end
    end
    @debug "\t\tBuilding vertical H terms row $row col $col"
    @timeit "build vertical terms" begin
        vTs = verticalTerms(A, L, R, AncEnvs[:I], AncEnvs[:V], vert_H_terms, row, col, ϕ) 
        Hs[term_counter:term_counter+length(vTs) - 1] = vTs
        term_counter += length(vTs)
        if verbose
            println( "--- vT TERMS ---")
            for vT in vTs
                println(scalar(vT * dag(ϕ)'))
            end
        end
    end
    @debug "\t\tBuilding field H terms row $row col $col"
    @timeit "build field terms" begin
        fTs = fieldTerms(A, L, R, AncEnvs[:I], AncEnvs[:F], field_H_terms, row, col, ϕ)
        Hs[term_counter:term_counter+length(fTs) - 1] = fTs[:]
        term_counter += length(fTs)
        if verbose
            println( "--- fT TERMS ---")
            for fT in fTs
                println(scalar(fT * dag(ϕ)'))
            end
        end
    end
    if col > 1
        @debug "\t\tBuilding left H terms row $row col $col"
        @timeit "build left terms" begin
            lTs = connectLeftTerms(A, L, R, AncEnvs[:I], AncEnvs[:L], left_H_terms, row, col, ϕ)
            Hs[term_counter:term_counter+length(lTs) - 1] = lTs[:]
            term_counter += length(lTs)
            if verbose
                println( "--- lT TERMS ---")
                for lT in lTs
                    println(scalar(lT * dag(ϕ)'))
                end
            end
        end
        @debug "\t\tBuilt left terms"
    end
    if col < Nx
        @debug "\t\tBuilding right H terms row $row col $col"
        @timeit "build right terms" begin
            rTs = connectRightTerms(A, L, R, AncEnvs[:I], AncEnvs[:R], right_H_terms, row, col, ϕ)
            Hs[term_counter:term_counter+length(rTs) - 1] = rTs[:]
            term_counter += length(rTs)
            if verbose
                println( "--- rT TERMS ---")
                for rT in rTs
                    println(scalar(rT * dag(ϕ)'))
                end
            end
        end
        @debug "\t\tBuilt right terms"
    end
    return Hs, N
end

function intraColumnGauge(A::PEPS, col::Int; kwargs...)::PEPS
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    @inbounds for row in reverse(2:Ny)
        @debug "\tBeginning intraColumnGauge for col $col row $row"
        cmb_is   = IndexSet(findindex(A[row, col], "Site"))
        if col > 1
            cmb_is = IndexSet(cmb_is, commonindex(A[row, col], A[row, col - 1]))
        end
        if col < Nx
            cmb_is = IndexSet(cmb_is, commonindex(A[row, col], A[row, col + 1]))
        end
        cmb, ci = combiner(cmb_is, tags="CMB")
        Lis     = IndexSet(ci) #cmb_is
        if row < Ny 
            Lis = IndexSet(Lis, commonindex(A[row, col], A[row + 1, col]))
        end
        Ac = A[row, col]*cmb
        U, S, V = svd(Ac, Lis; kwargs...)
        A[row, col]   = U*cmb
        A[row-1, col] *= (S*V)
    end
    return A
end

function simpleUpdate(A::PEPS, col::Int, next_col::Int, H; kwargs...)::PEPS
    do_side::Bool = get(kwargs, :do_side, true)
    τ::Float64    = get(kwargs, :tau, -0.1)
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    @inbounds for row in Iterators.reverse(1:Ny)
        if do_side
            hori_col   = next_col < col ? next_col : col
            nhori_col  = next_col < col ? col : next_col
            si_a       = findindex(A[row, col], "Site")
            si_b       = findindex(A[row, next_col], "Site")
            ci         = commonindex(A[row, col], A[row, next_col])
            min_dim    = dim(ci)
            Ua, Sa, Va = svd(A[row, col], si_a, ci; mindim=min_dim, kwargs...)
            Ub, Sb, Vb = svd(A[row, next_col], si_b, ci; mindim=min_dim, kwargs...)
            Hab_hori   = is_gpu ? cuITensor() : ITensor()
            horiH      = getDirectional(vcat(H[:, hori_col]...), Horizontal)
            horiH      = filter(x->x.sites[1][1] == row, horiH)
            for hH in horiH
                op_a = replaceindex!(copy(hH.ops[1]), hH.site_ind, hori_col == col ? si_a : si_b)
                op_a = replaceindex!(op_a, hH.site_ind', hori_col == col ? si_a' : si_b')
                op_b = replaceindex!(copy(hH.ops[2]), hH.site_ind, hori_col == col ? si_b : si_a)
                op_b = replaceindex!(op_b, hH.site_ind', hori_col == col ? si_b' : si_a')
                Hab_hori = dim(Hab_hori) < 2 ? op_a * op_b : Hab_hori + op_a * op_b
            end
            cmb, ci   = combiner(findinds(Hab_hori, ("",0)), tags="hab,Site")
            Hab_hori *= cmb
            Hab_hori *= cmb'
            Hab_mat   = is_gpu ? matrix(collect(Hab_hori)) : matrix(Hab_hori)
            expiH_mat = exp(τ*Hab_mat)
            expiH     = is_gpu ? cuITensor(vec(expiH_mat), ci, ci') : ITensor(expiH_mat, ci, ci')
            expiH *= cmb
            expiH *= cmb'
             
            bond  = noprime(expiH * Ua * Ub)
            Uf, Sf, Vf = svd(bond, si_a, commonindex(Ua, Sa); vtags="r,Link,r$row,c$hori_col", mindim=min_dim, kwargs...)
            A[row, col] = Sa * Va * Uf * Sf
            A[row, next_col] = Sb * Vb * Vf
        end
        if row < Ny
            si_a       = findindex(A[row, col], "Site")
            si_b       = findindex(A[row+1, col], "Site")
            ci         = commonindex(A[row, col], A[row+1, col])
            min_dim    = dim(ci)
            Ua, Sa, Va = svd(A[row, col], si_a, ci; mindim=min_dim, kwargs...)
            Ub, Sb, Vb = svd(A[row+1, col], si_b, ci; mindim=min_dim, kwargs...)
            Hab_vert   = is_gpu ? cuITensor() : ITensor()
            vertH      = getDirectional(vcat(H[:, col]...), Vertical)
            vertH      = filter(x->x.sites[1][1] == row, vertH)
            for vH in vertH
                op_a = replaceindex!(copy(vH.ops[1]), vH.site_ind, si_a)
                op_a = replaceindex!(op_a, vH.site_ind', si_a')
                op_b = replaceindex!(copy(vH.ops[2]), vH.site_ind, si_b)
                op_b = replaceindex!(op_b, vH.site_ind', si_b')
                Hab_vert = dim(Hab_vert) < 2 ? op_a * op_b : Hab_vert + op_a * op_b
            end
            cmb, ci   = combiner(findinds(Hab_vert, ("",0)), tags="hab,Site")
            Hab_vert *= cmb
            Hab_vert *= cmb'
            Hab_mat   = is_gpu ? matrix(collect(Hab_vert)) : matrix(Hab_vert)
            expiH_mat = exp(τ*Hab_mat)
            expiH     = is_gpu ? cuITensor(vec(expiH_mat), ci, ci') : ITensor(expiH_mat, ci, ci')
            expiH *= cmb
            expiH *= cmb'
            bond  = noprime(expiH * Ua * Ub)
            Uf, Sf, Vf = svd(bond, si_a, commonindex(Ua, Sa); vtags="u,Link,r$row,c$col", mindim=min_dim, kwargs...)
            A[row, col] = Sa * Va * Uf * Sf
            A[row+1, col] = Sb * Vb * Vf
        end
    end
    return A
end

function buildAncs(A::PEPS, L::Environments, R::Environments, H, col::Int)
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    dummy  = is_gpu ? cuITensor(1.0) : ITensor(1.0) 
    @debug "\tMaking ancillary identity terms for col $col"
    Ia = makeAncillaryIs(A, L, R, col)
    #Ib = fill(dummy, Ny)
    Ib = Vector{ITensor}(undef, Ny)
    Is = (above=Ia, below=Ib)

    @debug "\tMaking ancillary vertical terms for col $col"
    vH = getDirectional(vcat(H[:, col]...), Vertical)
    Va = makeAncillaryVs(A, L, R, vH, col)
    Vb = [Vector{ITensor}() for ii in 1:length(Va)]
    Vs = (above=Va, below=Vb)

    @debug "\tMaking ancillary field terms for col $col"
    fH = getDirectional(vcat(H[:, col]...), Field)
    Fa = makeAncillaryFs(A, L, R, fH, col)
    Fb = [Vector{ITensor}() for ii in 1:length(Fa)]
    Fs = (above=Fa, below=Fb)

    Ls = (above=Vector{ITensor}(), below=Vector{ITensor}()) 
    Rs = (above=Vector{ITensor}(), below=Vector{ITensor}()) 
    if col > 1
        @debug "\tMaking ancillary left terms for col $col"
        lH = getDirectional(vcat(H[:, col-1]...), Horizontal)
        La = makeAncillarySide(A, L, R, lH, col, :left)
        Lb = [Vector{ITensor}() for ii in  1:length(La)]
        Ls = (above=La, below=Lb)
    end
    if col < Nx
        @debug "\tMaking ancillary right terms for col $col"
        rH = getDirectional(vcat(H[:, col]...), Horizontal)
        Ra = makeAncillarySide(A, R, L, rH, col, :right)
        Rb = [Vector{ITensor}() for ii in  1:length(Ra)]
        Rs = (above=Ra, below=Rb)
    end
    Ancs = (I=Is, V=Vs, F=Fs, L=Ls, R=Rs)
    return Ancs
end

function updateAncs(A::PEPS, L::Environments, R::Environments, AncEnvs, H, row::Int, col::Int)
    Ny, Nx = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
   
    Is, Vs, Fs, Ls, Rs = AncEnvs
    @debug "\tUpdating ancillary identity terms for col $col row $row"
    Ib = updateAncillaryIs(A, Is[:below], L, R, row, col)
    Is = (above=Is[:above], below=Ib)

    @debug "\tUpdating ancillary vertical terms for col $col row $row"
    vH = getDirectional(vcat(H[:, col]...), Vertical)
    Vb = updateAncillaryVs(A, Vs[:below], Ib, L, R, vH, row, col)  
    Vs = (above=Vs[:above], below=Vb)

    @debug "\tUpdating ancillary field terms for col $col row $row"
    fH = getDirectional(vcat(H[:, col]...), Field)
    Fb = updateAncillaryFs(A, Fs[:below], Ib, L, R, fH, row, col)  
    Fs = (above=Fs[:above], below=Fb)

    if col > 1
        @debug "\tUpdating ancillary left terms for col $col row $row"
        lH = getDirectional(vcat(H[:, col-1]...), Horizontal)
        Lb = updateAncillarySide(A, Ls[:below], Ib, L, R, lH, row, col, :left)
        Ls = (above=Ls[:above], below=Lb)
    end
    if col < Nx
        @debug "\tUpdating ancillary right terms for col $col row $row"
        rH = getDirectional(vcat(H[:, col]...), Horizontal)
        Rb = updateAncillarySide(A, Rs[:below], Ib, R, L, rH, row, col, :right)
        Rs = (above=Rs[:above], below=Rb)
    end
    Ancs = (I=Is, V=Vs, F=Fs, L=Ls, R=Rs)
    return Ancs
end

struct ITensorMap
  A::PEPS
  H::Matrix{Vector{Operator}}
  L::Environments
  R::Environments
  AncEnvs::NamedTuple
  row::Int
  col::Int
end
Base.eltype(M::ITensorMap)  = eltype(M.A)
Base.size(M::ITensorMap)    = dim(M.A[M.row, M.col])
function (M::ITensorMap)(v::ITensor) 
    Hs, N  = buildLocalH(M.A, M.L, M.R, M.AncEnvs, M.H, M.row, M.col, v)
    localH = sum(Hs)
    return noprime(localH)
end

function optimizeLocalH(A::PEPS, L::Environments, R::Environments, AncEnvs, H, row::Int, col::Int; kwargs...)
    Ny, Nx        = size(A)
    is_gpu        = !(data(store(A[1,1])) isa Array)
    field_H_terms = getDirectional(vcat(H[:, col]...), Field)
    vert_H_terms  = getDirectional(vcat(H[:, col]...), Vertical)
    @debug "\tBuilding H for col $col row $row"
    @timeit "build H" begin
        Hs, N = buildLocalH(A, L, R, AncEnvs, H, row, col, A[row, col])
    end
    initial_N = real(scalar(collect(N * dag(A[row, col])')))
    @timeit "sum H terms" begin
        localH = sum(Hs)
    end
    initial_E = real(scalar(collect(deepcopy(localH) * dag(A[row, col])')))
    @info "Initial energy at row $row col $col : $(initial_E/(initial_N*Nx*Ny)) and norm : $initial_N"
    @debug "\tBeginning davidson for col $col row $row"
    mapper   = ITensorMap(A, H, L, R, AncEnvs, row, col)
    λ, new_A = davidson(mapper, A[row, col]; kwargs...)
    new_E    = λ #real(scalar(collect(new_A * localH * dag(new_A)')))
    N        = buildN(A, L, R, AncEnvs[:I], row, col, new_A)
    new_N    = real(scalar(collect(N * dag(new_A)')))
    #=println("Optimized energy at row $row col $col : $(new_E/(new_N*Nx*Ny)) and norm : $new_N")
    if new_E/new_N > initial_E/initial_N
        @info "badness"
        new_A = deepcopy(A[row, col])
        new_E = initial_E 
        new_N = initial_N #real(scalar(collect(N * dag(new_A)')))
    end=#
    @info "Optimized energy at row $row col $col : $(new_E/(new_N*Nx*Ny)) and norm : $new_N"
    @timeit "restore intraColumnGauge" begin
        if row < Ny
            @debug "\tRestoring intraColumnGauge for col $col row $row"
            cmb_is   = IndexSet(findindex(A[row, col], "Site"))
            if col > 1
                cmb_is = IndexSet(cmb_is, commonindex(A[row, col], A[row, col - 1]))
            end
            if col < Nx
                cmb_is = IndexSet(cmb_is, commonindex(A[row, col], A[row, col + 1]))
            end
            cmb, ci = combiner(cmb_is, tags="CMB")
            Lis     = IndexSet(ci) #cmb_is 
            if row > 1
                Lis = IndexSet(Lis, commonindex(A[row, col], A[row - 1, col]))
            end
            old_ci = commonindex(A[row, col], A[row+1, col])
            svdA = new_A*cmb
            Ris = uniqueinds(inds(svdA), Lis) 
            U, S, V = svd(svdA, Ris; kwargs...)
            new_ci = commonindex(V, S)
            replaceindex!(V, new_ci, old_ci)
            A[row, col]    = V*cmb 
            newU = S*U*A[row+1, col]
            replaceindex!(newU, new_ci, old_ci)
            A[row+1, col] = newU
            if row < Ny - 1
                nI    = spinI(findindex(A[row+1, col], "Site"); is_gpu=is_gpu)
                newAA = A[row+1, col] * nI * dag(A[row+1, col])'
                if col > 1
                    ci     = commonindex(A[row+1, col], A[row+1, col-1])
                    newAA *= multiply_side_ident(A[row+1, col], ci, L.I[row+1])
                end
                if col < Nx
                    ci     = commonindex(A[row+1, col], A[row+1, col+1])
                    newAA *= multiply_side_ident(A[row+1, col], ci, R.I[row+1])
                end
                AncEnvs[:I][:above][end - row] = newAA * AncEnvs[:I][:above][end - row - 1]
            end
        else
            A[row, col] = initial_N > 0 ? new_A/√new_N : new_A
        end
    end
    return A, AncEnvs
end

function measureEnergy(A::PEPS, L::Environments, R::Environments, AncEnvs, H, row::Int, col::Int)::Tuple{Float64, Float64}
    Ny, Nx    = size(A)
    Hs, N     = buildLocalH(A, L, R, AncEnvs, H, row, col, A[row, col])
    initial_N = collect(N * dag(A[row, col])')
    localH    = sum(Hs)
    initial_E = collect(localH * dag(A[row, col])')
    return real(scalar(initial_N)), real(scalar(initial_E))/real(scalar(initial_N))
end

function sweepColumn(A::PEPS, L::Environments, R::Environments, H, col::Int; kwargs...)
    Ny, Nx = size(A)
    @debug "Beginning intraColumnGauge for col $col" 
    A = intraColumnGauge(A, col; kwargs...)
    if col == div(Nx,2)# || col == Nx 
        L_s = buildLs(A, H; kwargs...)
        R_s = buildRs(A, H; kwargs...)
        EAncEnvs = buildAncs(A, L_s[col - 1], R_s[col + 1], H, col)
        N, E = measureEnergy(A, L_s[col - 1], R_s[col + 1], EAncEnvs, H, 1, col)
        println("Energy at mid:", E/(Nx*Ny))
    end
    @debug "Beginning buildAncs for col $col" 
    AncEnvs = buildAncs(A, L, R, H, col)
    @inbounds for row in 1:Ny
        if row > 1
            @debug "Beginning updateAncs for col $col" 
            AncEnvs = updateAncs(A, L, R, AncEnvs, H, row-1, col)
        end
        @debug "Beginning optimizing H for col $col" 
        A, AncEnvs = optimizeLocalH(A, L, R, AncEnvs, H, row, col; kwargs...)
    end
    return A
end

function rightwardSweep(A::PEPS, Ls::Vector{Environments}, Rs::Vector{Environments}, H; kwargs...)
    simple_update_cutoff = get(kwargs, :simple_update_cutoff, 4)
    Ny, Nx = size(A)
    dummyI = MPS(Ny, fill(ITensor(1.0), Ny), 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny)) 
    prev_cmb_r = Vector{ITensor}(undef, Ny)
    next_cmb_r = Vector{ITensor}(undef, Ny)
    sweep::Int = get(kwargs, :sweep, 0)
    @inbounds for col in 1:Nx-1
        L = col == 1 ? dummyEnv : Ls[col - 1]
        @debug "Sweeping col $col"
        if sweep >= simple_update_cutoff
            @timeit "sweep" begin
                A = sweepColumn(A, L, Rs[col+1], H, col; kwargs...)
            end
        end
        if sweep < simple_update_cutoff
            # Simple update...
            A = simpleUpdate(A, col, col+1, H; do_side=(col < Nx), kwargs...)
        end
        if sweep >= simple_update_cutoff
            # Gauge
            A = gaugeColumn(A, col, :right; kwargs...)
        end
        if col == 1
            left_H_terms = getDirectional(H[1], Horizontal)
            @timeit "left edge env" begin
                Ls[col] = buildEdgeEnvironment(A, H, left_H_terms, prev_cmb_r, :left, 1; kwargs...)
            end
        else
            @timeit "left next env" begin
                I, H_, IP = buildNextEnvironment(A, Ls[col-1], H, prev_cmb_r, next_cmb_r, :left, col; kwargs...)
                Ls[col] = Environments(I, H_, IP)
            end
            prev_cmb_r = deepcopy(next_cmb_r)
        end
    end
    return A, Ls, Rs
end

function leftwardSweep(A::PEPS, Ls::Vector{Environments}, Rs::Vector{Environments}, H; kwargs...)
    simple_update_cutoff = get(kwargs, :simple_update_cutoff, 4)
    Ny, Nx = size(A)
    dummyI = MPS(Ny, fill(ITensor(1.0), Ny), 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny)) 
    prev_cmb_l = Vector{ITensor}(undef, Ny)
    next_cmb_l = Vector{ITensor}(undef, Ny)
    sweep::Int = get(kwargs, :sweep, 0)
    @inbounds for col in reverse(2:Nx)
        R = col == Nx ? dummyEnv : Rs[col + 1]
        @debug "Sweeping col $col"
        if sweep >= simple_update_cutoff
            @timeit "sweep" begin
                A = sweepColumn(A, Ls[col - 1], R, H, col; kwargs...)
            end
        end
        if sweep < simple_update_cutoff
            # Simple update...
            A = simpleUpdate(A, col, col-1, H; do_side=(col>1), kwargs...)
        end
        if sweep >= simple_update_cutoff
            # Gauge
            A = gaugeColumn(A, col, :left; kwargs...)
        end
        if col == Nx
            right_H_terms = getDirectional(H[Nx - 1], Horizontal)
            @timeit "right edge env" begin
                Rs[col] = buildEdgeEnvironment(A, H, right_H_terms, prev_cmb_l, :right, Nx; kwargs...)
            end
        else
            @timeit "right next env" begin
                I, H_, IP = buildNextEnvironment(A, Rs[col+1], H, prev_cmb_l, next_cmb_l, :right, col; kwargs...)
                Rs[col]   = Environments(I, H_, IP)
            end
            prev_cmb_l = deepcopy(next_cmb_l)
        end
    end
    return A, Ls, Rs
end

function doSweeps(A::PEPS, Ls::Vector{Environments}, Rs::Vector{Environments}, H; mindim::Int=1, maxdim::Int=1, simple_update_cutoff::Int=4, sweep_start::Int=1, sweep_count::Int=10, cutoff::Float64=0.)
    for sweep in sweep_start:sweep_count
        if iseven(sweep)
            println("SWEEP RIGHT $sweep")
            A, Ls, Rs = rightwardSweep(A, Ls, Rs, H; sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
        else
            println("SWEEP LEFT $sweep")
            A, Ls, Rs = leftwardSweep(A, Ls, Rs, H; sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
        end
        flush(stdout)
        if sweep == simple_update_cutoff - 1
            for col in reverse(2:Nx)
                A = gaugeColumn(A, col, :left; mindim=1, maxdim=chi)
            end
            Ls = buildLs(A, H; mindim=1, maxdim=chi)
            Rs = buildRs(A, H; mindim=1, maxdim=chi)
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

