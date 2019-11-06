function measureXmag(A::PEPS, Ls::Vector{Environments}, Rs::Vector{Environments}; kwargs...)
    s = Index(2, "Site,SpinInd")
    X = ITensor(s, s')
    is_gpu = !(data(store(A[1,1])) isa Array)
    X[s(1), s'(2)] = 0.5
    X[s(2), s'(1)] = 0.5
    Ny, Nx = size(A)
    dummyI   = MPS(Ny, fill(ITensor(1.0), Ny), 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny))
    measuredX = zeros(Nx, Ny)
    op = is_gpu ? cuITensor(X) : X
    for col in 1:Nx
        Xs = [Operator([row=>col], [op], s, Field) for row in 1:Ny]
        A  = intraColumnGauge(A, col; kwargs...)
        tR = col == Nx ? dummyEnv : Rs[col+1]
        tL = col == 1  ? dummyEnv : Ls[col-1]
        AI = makeAncillaryIs(A, tL, tR, col)
        AF = makeAncillaryFs(A, tL, tR, Xs, col)
        fT = fieldTerms(A, tL, tR, (above=AI,), (above=AF,), Xs, 1, col) 
        N  = buildN(A, tL, tR, (above=AI,), 1, col)
        for row in 1:Ny
            measuredX[row, col] = scalar(A[1, col] * fT[row] * dag(A[1, col])')/scalar(A[1, col] * N * dag(A[1, col]'))
        end
    end
    return measuredX
end

function measureZmag(A::PEPS, Ls::Vector{Environments}, Rs::Vector{Environments}; kwargs...)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    is_gpu = !(data(store(A[1,1])) isa Array)
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    Nx, Ny = size(A)
    dummyI   = MPS(Ny, fill(ITensor(1.0), Ny), 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny))
    measuredZ = zeros(Nx, Ny)
    op = is_gpu ? cuITensor(Z) : Z 
    for col in 1:Nx
        Zs = [Operator([row=>col], [op], s, Field) for row in 1:Ny]
        A  = intraColumnGauge(A, col; kwargs...)
        tR = col == Nx ? dummyEnv : Rs[col+1]
        tL = col == 1  ? dummyEnv : Ls[col-1]
        AI = makeAncillaryIs(A, tL, tR, col)
        AF = makeAncillaryFs(A, tL, tR, Zs, col)
        fT = fieldTerms(A, tL, tR, (above=AI,), (above=AF,), Zs, 1, col)
        N  = buildN(A, tL, tR, (above=AI,), 1, col)
        for row in 1:Ny
            measuredZ[row, col] = scalar(A[1, col] * fT[row] * dag(A[1, col]'))/scalar(A[1, col] * N * dag(A[1, col]'))
        end
    end
    return measuredZ
end

function measureSmagVertical(A::PEPS, Ls::Vector{Environments}, Rs::Vector{Environments}; kwargs...)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    P = ITensor(s, s')
    M = ITensor(s, s')
    P[s(1), s'(2)] = 1.0
    M[s(2), s'(1)] = 1.0
    Nx, Ny     = size(A)
    dummyI     = MPS(Ny, fill(ITensor(1.0), Ny), 0, Ny+1)
    dummyEnv   = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny))
    measuredSV = zeros(Nx, Ny)
    is_gpu     = !(data(store(A[1,1])) isa Array)
    Z = is_gpu ? cuITensor(Z) : Z
    P = is_gpu ? cuITensor(P) : P
    M = is_gpu ? cuITensor(M) : M
    for col in 1:Nx
        SVs = Operator[]
        for row in 1:Ny-1
            push!(SVs, Operator([row=>col, row+1=>col], [0.5*P, M], s, Vertical))
            push!(SVs, Operator([row=>col, row+1=>col], [0.5*M, P], s, Vertical))
            push!(SVs, Operator([row=>col, row+1=>col], [Z, Z], s, Vertical))
        end
        A = intraColumnGauge(A, col; kwargs...)
        tR = col == Nx ? dummyEnv : Rs[col+1]
        tL = col == 1  ? dummyEnv : Ls[col-1]
        AI = makeAncillaryIs(A, tL, tR, col)
        AV = makeAncillaryVs(A, tL, tR, SVs, col)
        vTs = verticalTerms(A, tL, tR, (above=AI,), (above=AV,), SVs, 1, col) 
        N  = buildN(A, tL, tR, (above=AI,), 1, col)
        nrm = scalar(A[1, col] * N * dag(A[1, col]'))
        for (vi, vT) in enumerate(vTs)
            row = SVs[vi].sites[1][1]
            measuredSV[row, col] += scalar(A[1, col] * vT * dag(A[1, col]'))/nrm
        end
    end
    return measuredSV
end

function measureSmagHorizontal(A::PEPS, Ls::Vector{Environments}, Rs::Vector{Environments}; kwargs...)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    P = ITensor(s, s')
    M = ITensor(s, s')
    P[s(1), s'(2)] = 1.0
    M[s(2), s'(1)] = 1.0
    is_gpu     = !(data(store(A[1,1])) isa Array)
    Z = is_gpu ? cuITensor(Z) : Z
    P = is_gpu ? cuITensor(P) : P
    M = is_gpu ? cuITensor(M) : M
    Nx, Ny = size(A)
    dummyI     = MPS(Ny, fill(ITensor(1.0), Ny), 0, Ny+1)
    dummyEnv   = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny))
    measuredSH = zeros(Nx, Ny)
    for col in 1:Nx-1
        SHs = Operator[]
        for row in 1:Ny
            push!(SHs, Operator([row=>col, row=>col+1], [0.5*P, M], s, Horizontal))
            push!(SHs, Operator([row=>col, row=>col+1], [0.5*M, P], s, Horizontal))
            push!(SHs, Operator([row=>col, row=>col+1], [Z, Z], s, Horizontal))
        end
        A = intraColumnGauge(A, col; kwargs...)
        tR = Rs[col+1]
        tL = col == 1      ? dummyEnv : Ls[col-1]
        AI = makeAncillaryIs(A, tL, tR, col)
        AS = makeAncillarySide(A, tR, tL, SHs, col, :right)
        hTs = connectRightTerms(A, tL, tR, (above=AI,), (above=AS,), SHs, 1, col) 
        N  = buildN(A, tL, tR, (above=AI,), 1, col)
        nrm = scalar(A[1, col] * N * dag(A[1, col]'))
        for (hi, hT) in enumerate(hTs)
            row = SHs[hi].sites[1][1]
            measuredSH[row, col] += scalar(A[1, col] * hT * dag(A[1, col]'))/nrm
        end
    end
    return measuredSH
end
