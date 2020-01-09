include("peps.jl")

function reindexA(A::PEPS, new_A::PEPS)
    Ny, Nx   = size(new_A)
    midpoint = div(Nx, 2)
    old_column = Vector{Int}(undef, Nx)
    old_column[1:midpoint-1]  = 1:midpoint-1
    old_column[midpoint]      = midpoint
    old_column[midpoint+1]    = midpoint - 1
    old_column[midpoint+2:Nx] = midpoint:Nx-2
    new_right_bonds = Matrix{Index}(undef, Ny, Nx)
    new_left_bonds  = Matrix{Index}(undef, Ny, Nx)
    new_up_bonds    = Matrix{Index}(undef, Ny, Nx)
    new_down_bonds  = Matrix{Index}(undef, Ny, Nx)
    for col in 1:Nx, row in 1:Ny
        oldSpin = findindex(new_A[row, col], "Site")
        newSpin = Index(dim(oldSpin), tags(oldSpin))
        replaceindex!(new_A[row, col], oldSpin, newSpin)
        if row != Ny
            oldUp = commonindex(A[row, old_column[col]], A[row+1, old_column[col]])
            newUp = Index(dim(oldUp), "Link,u,c$col,r$row")
            new_up_bonds[row, col] = newUp
            new_down_bonds[row+1, col] = newUp
            replaceindex!(new_A[row, col], oldUp, newUp)
        end
        if row != 1
            oldDown = commonindex(A[row, old_column[col]], A[row-1, old_column[col]])
            newDown = new_down_bonds[row, col] 
            replaceindex!(new_A[row, col], oldDown, newDown)
        end
        if col != Nx
            oldRight = commonindex(A[row, old_column[col]+1], A[row, old_column[col]])
            newRight = Index(dim(oldRight), "Link,r,c$col,r$row")
            new_right_bonds[row, col]  = newRight
            new_left_bonds[row, col+1] = newRight
            replaceindex!(new_A[row, col], oldRight, newRight)
        end
        if col != 1
            oldLeft = commonindex(A[row, old_column[col]-1], A[row, old_column[col]])
            newLeft = new_left_bonds[row, col] 
            replaceindex!(new_A[row, col], oldLeft, newLeft)
        end
    end
    return new_A
end

function InsertTensors(A::PEPS, Rs::Vector{Environments}, Ls::Vector{Environments}, H; kwargs...)
    maxdim      = get(kwargs, :maxdim, 1)
    Ny, Nx      = size(A)
    midpoint    = div(Nx, 2)
    A, Q_f, R_f, finds = gaugeColumnForInsert(A, midpoint, :right; kwargs...)
    for row in 1:Ny
        QRind = findindex(Q_f[row], "QR")
        ci    = commonindex(A[row, midpoint], A[row, midpoint + 1])
        replaceindex!(Q_f[row], QRind, ci)
        A[row, midpoint] = Q_f[row]
    end
    oldH = deepcopy(H)
    Ar = deepcopy(A[:, midpoint + 1])
    Al = deepcopy(A[:, midpoint])
    new_A = PEPS(Nx+2, Ny, hcat(deepcopy(A[:, 1:midpoint]), Ar, Al, deepcopy(A[:, midpoint+1:Nx])))
    new_A = reindexA(A, new_A)
    A     = new_A
    Hr = H[:, midpoint + 1]
    Hl = H[:, midpoint]
    H = hcat(H[:, 1:midpoint], Hr, Hl, H[:, midpoint+1:Nx])
    Nx += 2
    A, Q_l, R_l, linds = gaugeColumnForInsert(A, midpoint + 1, :right; kwargs...)
    A, Q_r, R_r, rinds = gaugeColumnForInsert(A, midpoint + 2, :left; kwargs...)
    A[:, midpoint + 1] = tensors(Q_l)
    A[:, midpoint + 2] = tensors(Q_r)
    Λı = deepcopy(R_l) # Lambdan
    Λȷ = MPO(Ny) # Lambdanmo
    X  = MPO(Ny)
    X_up_inds = [Index(maxdim, "Link,Xup,l=$row") for row in 1:Ny-1]
    midpoint += 1
    for row in 1:Ny
        Λȷ[row] = R_f[row]
        Xinds   = Index[findindex(R_l[row], "QR"); rinds[row]]
        if row > 1
            push!(Xinds, X_up_inds[row-1]) 
        end
        if row < Ny
            push!(Xinds, X_up_inds[row]) 
        end
        X[row]  = cuITensor(dense(diagITensor(1.0, IndexSet(Xinds...))))
        X[row] /= norm(X[row])
        i_ind = findindex(Λı[row], "DM")
        j_ind = findindex(Λȷ[row], "QR")
        replaceindex!(Λȷ[row], j_ind, i_ind)
        replaceindex!(Λȷ[row], finds[row], rinds[row])
    end
    γ = 0.05
    inner_product = 0.1
    iter   = 1
    Λk     = deepcopy(Λȷ) #Lambdanmo_
    Λℓ     = deepcopy(Λȷ) #Lambdanmo__
    Λm     = deepcopy(Λı) #Lambdan_
    # F = ||A*x - b||^2 - ||x||^2
    # Grad F = 2A^T(Ax - b) - 2x
    # grad_const is 2A^T*b
    #foreach(println, inds.(Λk))
    #foreach(println, inds.(Λm))
    Δconst = multMPO(Λk, Λm; kwargs...)
    println("Begin grad descent")
    flush(stdout)
    while abs(inner_product) < 0.95 && iter < 100
        tmp_Δ  = multMPO(X, Λȷ; kwargs...)
        tmp_Δ2 = multMPO(tmp_Δ, Λℓ; kwargs...)
        Δ      = sum([2.0*Δconst, -2.0*X, 2.0*tmp_Δ2])
        X      = sum(X, -γ*Δ; kwargs...)
        tmp_Δ  = multMPO(Λℓ, X; kwargs...)
        inner_product = overlap(tmp_Δ, Λm)
        iter  += 1
    end
    println("Done grad descent")
    flush(stdout)
    XR  = multMPO(X, R_r; kwargs...)
    # must combine inds of QR
    Q_r_cmbs = Vector{ITensor}(undef, Ny)
    for row in 1:Ny
        cmb_inds = IndexSet(uniqueindex(findinds(Q_r[row], "Site"), findinds(XR[row], "Site")), findindex(Q_r[row], "Link,r"))
        Q_r_cmb, ci = combiner(cmb_inds, tags="Qcmb,Site,r$row")
        Q_r_cmbs[row] = Q_r_cmb
        Q_r[row] *= Q_r_cmb
    end
    XRQ = multMPO(XR, Q_r; kwargs...)
    for row in 1:Ny
        XRQ[row] *= Q_r_cmbs[row]
    end
    A[:, midpoint+1] = tensors(XRQ)
    for row in 1:Ny
        qr_ci = commonindex(A[row, midpoint], A[row, midpoint+1])
        new_link = Index(dim(qr_ci), "Link,r,c$midpoint,r$row")
        replaceindex!(A[row, midpoint], qr_ci, new_link)
        replaceindex!(A[row, midpoint+1], qr_ci, new_link)
    end
    return A, H
end

function rightwardSweepToInsert(A::PEPS, Ls::Vector{Environments}, Rs::Vector{Environments}, H; kwargs...)
    simple_update_cutoff = get(kwargs, :simple_update_cutoff, 4)
    Ny, Nx = size(A)
    dummyI = MPS(Ny, fill(ITensor(1.0), Ny), 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny)) 
    prev_cmb_r = Vector{ITensor}(undef, Ny)
    next_cmb_r = Vector{ITensor}(undef, Ny)
    sweep::Int = get(kwargs, :sweep, 0)
    stop_col::Int = get(kwargs, :stop_col, div(Nx, 2))
    sweep_width::Int = get(kwargs, :sweep_width, Nx)
    offset    = mod(Nx, 2)
    midpoint  = div(Nx, 2)
    rightmost = midpoint
    leftmost  = sweep_width == Nx ? 1 : midpoint - div(sweep_width, 2)
    if leftmost > 1 
        for row in 1:Ny
            prev_cmb_r[row] = reconnect(commonindex(A[row, leftmost], A[row, leftmost-1]), Ls[leftmost-1].I[row])
        end
    end
    @inbounds for col in leftmost:rightmost
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

function rightwardSweepFromInsert(A::PEPS, Ls::Vector{Environments}, Rs::Vector{Environments}, H; kwargs...)
    simple_update_cutoff = get(kwargs, :simple_update_cutoff, 4)
    Ny, Nx = size(A)
    dummyI = MPS(Ny, fill(ITensor(1.0), Ny), 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny)) 
    prev_cmb_r = Vector{ITensor}(undef, Ny)
    next_cmb_r = Vector{ITensor}(undef, Ny)
    sweep::Int = get(kwargs, :sweep, 0)
    start_col::Int = get(kwargs, :start_col, div(Nx, 2))
    sweep_width::Int = get(kwargs, :sweep_width, Nx)
    offset    = mod(Nx, 2)
    midpoint  = div(Nx, 2)
    rightmost = midpoint + div(sweep_width, 2) + offset
    leftmost  = midpoint + 1
    @inbounds for col in leftmost:rightmost
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

function rightwardSweepFromInsert(A::PEPS, Ls::Vector{Environments}, Rs::Vector{Environments}, H; kwargs...)
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

function doSweepsInsert(A::PEPS, Ls::Vector{Environments}, Rs::Vector{Environments}, H; mindim::Int=1, maxdim::Int=1, simple_update_cutoff::Int=4, sweep_start::Int=1, sweep_count::Int=10, cutoff::Float64=0., insert_interval::Int=20)
    Ny, Nx      = size(A)
    sweep_width = Nx
    for sweep in sweep_start:sweep_count
        insert_flag = mod(sweep, insert_interval) == 0
        if iseven(sweep)
            if !insert_flag
                println("SWEEP RIGHT $sweep")
                A, Ls, Rs = rightwardSweep(A, Ls, Rs, H; sweep_width=sweep_width, sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
            else
                println("SWEEP RIGHT TO INSERT $sweep")
                A, Ls, Rs = rightwardSweepToInsert(A, Ls, Rs, H; sweep_width=sweep_width, stopcol=div(Nx, 2), sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
            end
        else
            println("SWEEP LEFT $sweep")
            A, Ls, Rs = leftwardSweep(A, Ls, Rs, H; sweep_width=sweep_width, sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
        end
        flush(stdout)
        if sweep == simple_update_cutoff - 1
            for col in reverse(2:Nx)
                A = gaugeColumn(A, col, :left; mindim=1, maxdim=maxdim, cutoff=cutoff)
            end
            Ls = buildLs(A, H; mindim=1, maxdim=maxdim, cutoff=cutoff)
            Rs = buildRs(A, H; mindim=1, maxdim=maxdim, cutoff=cutoff)
        end
        if insert_flag
            A, H = InsertTensors(A, Rs, Ls, H; mindim=mindim, maxdim=maxdim, overlap_cutoff=0.999, cutoff=cutoff)
            Ny, Nx = size(A)
            println(" Insert complete at Nx = $Nx, Ny = $Ny")
            flush(stdout)
            sweep_width = Nx
            Ls = buildLs(A, H; mindim=1, maxdim=maxdim, cutoff=cutoff)
            Rs = buildRs(A, H; mindim=1, maxdim=maxdim, cutoff=cutoff)
            println(" Rebuild of envs complete at Nx = $Nx, Ny = $Ny")
            flush(stdout)
            rightwardSweepFromInsert(A, Ls, Rs, H; sweep_width=sweep_width, startcol=div(Nx, 2), sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=0.999, cutoff=cutoff)
            sweep_width = Nx
            if sweep_width < Nx
                midpoint = div(Nx, 2) 
                barrier = midpoint + div(sweep_width, 2) + 2
                barrier = barrier > Nx ? Nx : barrier
                for col in reverse((midpoint + div(sweep_width, 2) + 1):barrier)
                    A = gaugeColumn(A, col, :left; mindim=1, maxdim=maxdim, cutoff=cutoff)  
                end
                #Ls = buildLs(A, H; mindim=1, maxdim=maxdim, cutoff=cutoff)
                #Rs = buildRs(A, H; mindim=1, maxdim=maxdim, cutoff=cutoff)
            end
        end
        Ls = buildLs(A, H; mindim=1, maxdim=maxdim, cutoff=cutoff)
        Rs = buildRs(A, H; mindim=1, maxdim=maxdim, cutoff=cutoff)
        #=if sweep == sweep_count
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
        end=#
    end
    return A
end
