function prepareRow(A::ITensor, op::ITensor, left_A::ITensor, right_A::ITensor, left_I::ITensor, right_I::ITensor, col::Int, Nx::Int)
    AA = A * op * dag(A')
    if col > 1
        ci  = commonindex(A, left_A)
        #println("before msi: ")
        #@show collect(left_I)
        msi = multiply_side_ident(AA, ci, left_I)
        #println("msi: ")
        #@show collect(msi)
        AA *= msi
    end
    if col < Nx
        ci  = commonindex(A, right_A)
        #println("before msi: ")
        #@show collect(right_I)
        msi = multiply_side_ident(AA, ci, right_I)
        #println("msi: ")
        #@show collect(msi)
        AA *= msi
    end
    return AA
end

function makeAncillaryIs(A::PEPS, L::Environments, R::Environments, col::Int)
    Ny, Nx   = size(A)
    is_gpu   = !(data(store(A[1,1])) isa Array)
    dummy    = is_gpu    ? cuITensor(1.0)  : ITensor(1.0) 
    left_As  = [col > 1  ? A[row, col - 1] : dummy for row in 1:Ny] 
    right_As = [col < Nx ? A[row, col + 1] : dummy for row in 1:Ny]
    col_site_inds = [findindex(x, "Site") for x in A[:, col]]
    ops = map(x -> spinI(x; is_gpu=is_gpu), col_site_inds)
    AAs = [prepareRow(A[row, col], ops[row], left_As[row], right_As[row], L.I[row], R.I[row], col, Nx) for row in 1:Ny]
    Iabove = cumprod(reverse(AAs))
    return Iabove
end

function makeAncillaryIsBelow(A::PEPS, L::Environments, R::Environments, col::Int)
    Ny, Nx   = size(A)
    is_gpu   = !(data(store(A[1,1])) isa Array)
    dummy    = is_gpu    ? cuITensor(1.0)  : ITensor(1.0) 
    left_As  = [col > 1  ? A[row, col - 1] : dummy for row in 1:Ny] 
    right_As = [col < Nx ? A[row, col + 1] : dummy for row in 1:Ny]
    col_site_inds = [findindex(x, "Site") for x in A[:, col]]
    ops = map(x -> spinI(x; is_gpu=is_gpu), col_site_inds)
    AAs = [prepareRow(A[row, col], ops[row], left_As[row], right_As[row], L.I[row], R.I[row], col, Nx) for row in 1:Ny]
    return cumprod(AAs)
end

function updateAncillaryIs(A::PEPS, Ibelow::Vector{ITensor}, L::Environments, R::Environments, row::Int, col::Int )
    Ny, Nx  = size(A)
    is_gpu  = !(data(store(A[1,1])) isa Array)
    dummy   = is_gpu   ? cuITensor(1.0)  : ITensor(1.0) 
    left_A  = col > 1  ? A[row, col - 1] : dummy
    right_A = col < Nx ? A[row, col + 1] : dummy
    op      = spinI(findindex(A[row, col], "Site"); is_gpu=is_gpu)
    AA      = prepareRow(A[row, col], op, left_A, right_A, L.I[row], R.I[row], col, Nx)
    AA     *= row > 1  ? Ibelow[row - 1] : dummy 
    Ibelow[row] = AA
    return Ibelow
end

function makeAncillaryFs(A::PEPS, L::Environments, R::Environments, H, col::Int)
    Ny, Nx   = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    dummy    = is_gpu    ? cuITensor(1.0)  : ITensor(1.0) 
    left_As  = [col > 1  ? A[row, col - 1] : dummy for row in 1:Ny] 
    right_As = [col < Nx ? A[row, col + 1] : dummy for row in 1:Ny]
    col_site_inds = [findindex(x, "Site") for x in A[:, col]]
    Fabove   = fill(Vector{ITensor}(), length(H))
    for opcode in 1:length(H)
        op_row      = H[opcode].sites[1][1]
        ops = map(x -> spinI(x; is_gpu=is_gpu), col_site_inds)
        ops[op_row] = replaceindex!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row])
        ops[op_row] = replaceindex!(copy(ops[op_row]), H[opcode].site_ind', col_site_inds[op_row]')
        ancFs = [prepareRow(A[row, col], ops[row], left_As[row], right_As[row], L.I[row], R.I[row], col, Nx) for row in 1:Ny]
        Fabove[opcode] = cumprod(reverse(ancFs)) 
    end
    return Fabove
end

function updateAncillaryFs(A::PEPS, Fbelow, Ibelow::Vector{ITensor}, L::Environments, R::Environments, H, row::Int, col::Int)
    Ny, Nx   = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    dummy    = is_gpu    ? cuITensor(1.0)  : ITensor(1.0) 
    left_As  = [col > 1  ? A[row, col - 1] : dummy for row in 1:Ny] 
    right_As = [col < Nx ? A[row, col + 1] : dummy for row in 1:Ny]
    col_site_inds = [findindex(x, "Site") for x in A[:, col]]
    for opcode in 1:length(H)
        op_row      = H[opcode].sites[1][1]
        ops         = ITensor[spinI(spin_ind; is_gpu=is_gpu) for spin_ind in col_site_inds] 
        if op_row == row
            ops[op_row] = replaceindex!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row])
            ops[op_row] = replaceindex!(ops[op_row], H[opcode].site_ind', col_site_inds[op_row]')
        end
        ancF = prepareRow(A[row, col], ops[row], left_As[row], right_As[row], L.I[row], R.I[row], col, Nx)
        if length(Fbelow[opcode]) > 0
            push!(Fbelow[opcode], Fbelow[opcode][end]*ancF) 
        else
            push!(Fbelow[opcode], ancF) 
        end
    end
    return Fbelow
end

function makeAncillaryVs(A::PEPS, L::Environments, R::Environments, H, col::Int)
    Ny, Nx   = size(A)
    is_gpu   = !(data(store(A[1,1])) isa Array)
    dummy    = is_gpu    ? cuITensor(1.0)  : ITensor(1.0) 
    left_As  = [col > 1  ? A[row, col - 1] : dummy for row in 1:Ny] 
    right_As = [col < Nx ? A[row, col + 1] : dummy for row in 1:Ny]
    col_site_inds = [findindex(x, "Site") for x in A[:, col]]
    Vabove   = fill(Vector{ITensor}(), length(H))
    for opcode in 1:length(H)
        op_row_a      = H[opcode].sites[1][1]
        op_row_b      = H[opcode].sites[2][1]
        ops           = map(x->spinI(x; is_gpu=is_gpu), col_site_inds)
        ops[op_row_a] = replaceindex!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row_a])
        ops[op_row_a] = replaceindex!(ops[op_row_a], H[opcode].site_ind', col_site_inds[op_row_a]')
        ops[op_row_b] = replaceindex!(copy(H[opcode].ops[2]), H[opcode].site_ind, col_site_inds[op_row_b])
        ops[op_row_b] = replaceindex!(ops[op_row_b], H[opcode].site_ind', col_site_inds[op_row_b]')
        ancVs         = [prepareRow(A[row, col], ops[row], left_As[row], right_As[row], L.I[row], R.I[row], col, Nx) for row in 1:Ny]
        Vabove[opcode] = cumprod(reverse(ancVs))#[1:op_row_a+1] 
    end
    return Vabove
end

function makeAncillaryVsBelow(A::PEPS, L::Environments, R::Environments, H, col::Int)
    Ny, Nx   = size(A)
    is_gpu   = !(data(store(A[1,1])) isa Array)
    dummy    = is_gpu    ? cuITensor(1.0)  : ITensor(1.0) 
    left_As  = [col > 1  ? A[row, col - 1] : dummy for row in 1:Ny] 
    right_As = [col < Nx ? A[row, col + 1] : dummy for row in 1:Ny]
    col_site_inds = [findindex(x, "Site") for x in A[:, col]]
    Vbelow   = fill(Vector{ITensor}(), length(H))
    for opcode in 1:length(H)
        op_row_a      = H[opcode].sites[1][1]
        op_row_b      = H[opcode].sites[2][1]
        ops           = map(x->spinI(x; is_gpu=is_gpu), col_site_inds)
        ops[op_row_a] = replaceindex!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row_a])
        ops[op_row_a] = replaceindex!(ops[op_row_a], H[opcode].site_ind', col_site_inds[op_row_a]')
        ops[op_row_b] = replaceindex!(copy(H[opcode].ops[2]), H[opcode].site_ind, col_site_inds[op_row_b])
        ops[op_row_b] = replaceindex!(ops[op_row_b], H[opcode].site_ind', col_site_inds[op_row_b]')
        ancVs = [prepareRow(A[row, col], ops[row], left_As[row], right_As[row], L.I[row], R.I[row], col, Nx) for row in 1:Ny]
        Vbelow[opcode] = cumprod(ancVs)#[1:op_row_a+1] 
    end
    return Vbelow
end

function updateAncillaryVs(A::PEPS, Vbelow, Ibelow::Vector{ITensor}, L::Environments, R::Environments, H, row::Int, col::Int)
    Ny, Nx   = size(A)
    is_gpu   = !(data(store(A[1,1])) isa Array)
    dummy    = is_gpu    ? cuITensor(1.0)  : ITensor(1.0) 
    left_As  = [col > 1  ? A[row, col - 1] : dummy for row in 1:Ny] 
    right_As = [col < Nx ? A[row, col + 1] : dummy for row in 1:Ny]
    col_site_inds = [findindex(x, "Site") for x in A[:, col]]
    for opcode in 1:length(H)
        op_row_a      = H[opcode].sites[1][1]
        op_row_b      = H[opcode].sites[2][1]
        ops           = map(x->spinI(x; is_gpu=is_gpu), col_site_inds)
        ops[op_row_a] = replaceindex!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row_a])
        ops[op_row_a] = replaceindex!(ops[op_row_a], H[opcode].site_ind', col_site_inds[op_row_a]')
        ops[op_row_b] = replaceindex!(copy(H[opcode].ops[2]), H[opcode].site_ind, col_site_inds[op_row_b])
        ops[op_row_b] = replaceindex!(ops[op_row_b], H[opcode].site_ind', col_site_inds[op_row_b]')
        if op_row_b < row
            AA  = prepareRow(A[row, col], ops[row], left_As[row], right_As[row], L.I[row], R.I[row], col, Nx)
            push!(Vbelow[opcode], Vbelow[opcode][end] * AA)
        elseif op_row_b == row
            Ib  = op_row_a > 1 ? Ibelow[op_row_a - 1] : dummy
            AA  = prepareRow(A[op_row_a, col], ops[op_row_a], left_As[op_row_a], right_As[op_row_a], L.I[op_row_a], R.I[op_row_a], col, Nx)
            AA *= prepareRow(A[op_row_b, col], ops[op_row_b], left_As[op_row_b], right_As[op_row_b], L.I[op_row_b], R.I[op_row_b], col, Nx)
            push!(Vbelow[opcode], AA*Ib)
        end
    end
    return Vbelow
end

function makeAncillarySide(A::PEPS, EnvIP::Environments, EnvIdent::Environments, H, col::Int, side::Symbol)
    Ny, Nx   = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    col_site_inds = [findindex(x, "Site") for x in A[:, col]]
    Sabove   = fill(Vector{ITensor}(), length(H))
    next_col = side == :left ? col + 1 : col - 1
    for opcode in 1:length(H)
        op_row      = side == :left ? H[opcode].sites[2][1] : H[opcode].sites[1][1] 
        ops         = map(x->spinI(x; is_gpu=is_gpu), col_site_inds)
        ops         = is_gpu ? map(cuITensor, ops) : ops
        this_op     = side == :left ? H[opcode].ops[2] : H[opcode].ops[1]
        ops[op_row] = replaceindex!(copy(this_op), H[opcode].site_ind, col_site_inds[op_row])
        ops[op_row] = replaceindex!(ops[op_row], H[opcode].site_ind', col_site_inds[op_row]')
        AAs         = [A[row, col] * ops[row] * dag(A[row, col])' * EnvIP.InProgress[row, opcode] for row in 1:Ny]
        if (col > 1 && side == :right) || (col < Nx && side == :left)
            cis = [commonindex(A[row, col], A[row, next_col]) for row in 1:Ny]
            msi = [multiply_side_ident(AAs[row], cis[row], EnvIdent.I[row]) for row in 1:Ny]
            AAs = AAs .* msi
        end
        Sabove[opcode] = cumprod(reverse(AAs))
    end
    return Sabove
end

function makeAncillarySideBelow(A::PEPS, EnvIP::Environments, EnvIdent::Environments, H, col::Int, side::Symbol)
    Ny, Nx   = size(A)
    is_gpu   = !(data(store(A[1,1])) isa Array)
    col_site_inds = [findindex(x, "Site") for x in A[:, col]]
    Sbelow   = fill(Vector{ITensor}(), length(H))
    next_col = side == :left ? col + 1 : col - 1
    for opcode in 1:length(H)
        op_row      = side == :left ? H[opcode].sites[2][1] : H[opcode].sites[1][1] 
        ops         = map(x->spinI(x; is_gpu=is_gpu), col_site_inds)
        ops         = is_gpu ? map(cuITensor, ops) : ops
        this_op     = side == :left ? H[opcode].ops[2] : H[opcode].ops[1]
        ops[op_row] = replaceindex!(copy(this_op), H[opcode].site_ind, col_site_inds[op_row])
        ops[op_row] = replaceindex!(ops[op_row], H[opcode].site_ind', col_site_inds[op_row]')
        AAs         = [A[row, col] * ops[row] * dag(A[row, col])' * EnvIP.InProgress[row, opcode] for row in 1:Ny]
        if (col > 1 && side == :right) || (col < Nx && side == :left)
            cis = [commonindex(A[row, col], A[row, next_col]) for row in 1:Ny]
            msi = [multiply_side_ident(AAs[row], cis[row], EnvIdent.I[row]) for row in 1:Ny]
            AAs = AAs .* msi
        end
        Sbelow[opcode] = cumprod(AAs)
    end
    return Sbelow
end

function updateAncillarySide(A::PEPS, Sbelow, Ibelow::Vector{ITensor}, EnvIP::Environments, EnvIdent::Environments, H, row::Int, col::Int, side::Symbol)
    Ny, Nx   = size(A)
    is_gpu = !(data(store(A[1,1])) isa Array)
    col_site_inds = [findindex(x, "Site") for x in A[:, col]]
    next_col = side == :left ? col + 1 : col - 1
    prev_col = side == :left ? col + 1 : col - 1
    for opcode in 1:length(H)
        op_row        = side == :left ? H[opcode].sites[2][1] : H[opcode].sites[1][1] 
        ops           = ITensor[spinI(spin_ind; is_gpu=is_gpu) for spin_ind in col_site_inds] 
        ops           = is_gpu ? map(cuITensor, ops) : ops
        this_op       = side == :left ? H[opcode].ops[2] : H[opcode].ops[1]
        ops[op_row]   = replaceindex!(copy(this_op), H[opcode].site_ind, col_site_inds[op_row])
        ops[op_row]   = replaceindex!(ops[op_row], H[opcode].site_ind', col_site_inds[op_row]')
        AA            = A[row, col] * ops[row] * dag(A[row, col]')
        if (col > 1 && side == :right) || (col < Nx && side == :left)
            ci  = commonindex(A[row, col], A[row, prev_col])
            msi = multiply_side_ident(AA, ci, EnvIdent.I[row])
            AA  = AA * msi
        end
        AA *= EnvIP.InProgress[row, opcode]
        if row > 1
            AAinds = inds(AA)
            Sbinds = inds(Sbelow[opcode][row-1])
            for aI in AAinds, sI in Sbinds
                if hastags(aI, tags(sI)) && !hasindex(AAinds, sI)
                    Sbelow[opcode][row-1] = replaceindex!(Sbelow[opcode][row-1], sI, aI)
                end
            end
        end
        thisS = row >= 2 ? Sbelow[opcode][row-1] * AA : AA
        push!(Sbelow[opcode], thisS)
    end
    return Sbelow
end
