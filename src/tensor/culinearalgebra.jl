
#
# Linear Algebra of order 2 Tensors
#
# Even though CuDenseTensor{_,2} is strided
# and passable to BLAS/LAPACK, it cannot
# be made <: StridedArray

function Base.:*(T1::Tensor{ElT1,2,StoreT1,IndsT1},
                 T2::Tensor{ElT2,2,StoreT2,IndsT2}) where
                                        {ElT1,StoreT1<:CuDense,IndsT1,
                                        ElT2,StoreT2<:CuDense,IndsT2}
  println("hi")
  RM    = matrix(T1)*matrix(T2)
  indsR = IndsT1(ind(T1,1),ind(T2,2))
  pT    = promote_type(ElT1,ElT2)
  return Tensor(Dense(vec(RM)),indsR)
end

function LinearAlgebra.exp(T::CuDenseTensor{ElT,2}) where {ElT,IndsT}
  expTM = exp(matrix(T))
  return Tensor(Dense(vec(expTM)),inds(T))
end

function expHermitian(T::CuDenseTensor{ElT,2}) where {ElT,IndsT}
  # exp(::Hermitian/Symmetric) returns Hermitian/Symmetric,
  # so extract the parent matrix
  expTM = parent(exp(Hermitian(matrix(T))))
  return Tensor(Dense(vec(expTM)),inds(T))
end

# svd of an order-2 tensor
function LinearAlgebra.svd(T::CuDenseTensor{ElT,2,IndsT}; kwargs...) where {ElT,IndsT}
  maxdim::Int = get(kwargs,:maxdim,minimum(dims(T)))
  mindim::Int = get(kwargs,:mindim,1)
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
  doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)
  fastSVD::Bool = get(kwargs,:fastSVD,false)
  MU,MS,MV = CUSOLVER.svd(array(T))
  #conj!(MV)
  P = MS.^2
  truncerr, docut, P = truncate!(P;mindim=mindim,
              maxdim=maxdim,
              cutoff=cutoff,
              absoluteCutoff=absoluteCutoff,
              doRelCutoff=doRelCutoff)
  spec = Spectrum(P,truncerr)
  dS = length(P)
  if dS < length(MS)
    MU = MU[:,1:dS]
    MS = MS[1:dS]
    MV = MV[:,1:dS]
  end

  # Make the new indices to go onto U and V
  u = eltype(IndsT)(dS)
  v = eltype(IndsT)(dS)
  Uinds = IndsT((ind(T,1),u))
  Sinds = IndsT((u,v))
  Vinds = IndsT((ind(T,2),v))
  local Udata, Sdata, Vdata 
  unified = CUDAdrv.is_managed(data(store(T)).ptr)
  if unified
      ubuf  = CUDAdrv.Mem.alloc(CUDAdrv.Mem.UnifiedBuffer, length(MU) * sizeof(ElT))
      Udata = unsafe_wrap(CuVector{ElT}, convert(CuPtr{ElT}, ubuf), length(MU); own=true)
      sbuf  = CUDAdrv.Mem.alloc(CUDAdrv.Mem.UnifiedBuffer, dS*dS * sizeof(ElT))
      Sdata = unsafe_wrap(CuVector{ElT}, convert(CuPtr{ElT}, sbuf), dS*dS; own=true)
      vbuf  = CUDAdrv.Mem.alloc(CUDAdrv.Mem.UnifiedBuffer, length(MV) * sizeof(ElT))
      Vdata = unsafe_wrap(CuVector{ElT}, convert(CuPtr{ElT}, vbuf), length(MV); own=true)
      fill!(Sdata, zero(ElT))
  else
      Udata = CuArrays.zeros(ElT, length(MU)) 
      Sdata = CuArrays.zeros(ElT, dS * dS)
      Vdata = CuArrays.zeros(ElT, length(MV)) 
  end
  copyto!(Udata, vec(MU))
  copyto!(Vdata, vec(MV))
  dsi = diagind(reshape(Sdata, dS, dS), 0)
  Sdata[dsi] = MS
  U = Tensor(Dense(Udata),Uinds)
  S = Tensor(Dense(Sdata),Sinds)
  V = Tensor(Dense(Vdata),Vinds)
  return U,S,V,spec
end

function eigenHermitian(T::CuDenseTensor{ElT,2,IndsT};
                        kwargs...) where {ElT,IndsT}
  ispossemidef::Bool = get(kwargs,:ispossemidef,false)
  maxdim::Int = get(kwargs,:maxdim,minimum(dims(T)))
  mindim::Int = get(kwargs,:mindim,1)
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
  doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)
  synchronize()
  local DM, UM 
  if ElT <: Complex
    DM, UM = CUSOLVER.heevd!('V', 'U', matrix(T))
  else
    DM, UM = CUSOLVER.syevd!('V', 'U', matrix(T))
  end
  DM_ = reverse(DM)
  truncerr, docut, DM = truncate!(DM_;maxdim=maxdim, cutoff=cutoff, absoluteCutoff=absoluteCutoff, doRelCutoff=doRelCutoff)
  spec = Spectrum(DM,truncerr)
  dD = length(DM)
  dV = reverse(UM, dims=2)
  if dD < size(dV,2)
      dV = CuMatrix(dV[:,1:dD])
  end
  # Make the new indices to go onto U and V
  u = eltype(IndsT)(dD)
  v = eltype(IndsT)(dD)
  Uinds = IndsT((ind(T,1),u))
  Dinds = IndsT((u,v))
  local dV_
  unified = CUDAdrv.is_managed(data(store(T)).ptr)
  if unified
      buf = CUDAdrv.Mem.alloc(CUDAdrv.Mem.UnifiedBuffer, length(dV) * sizeof(ElT))
      dV_ = unsafe_wrap(CuVector{ElT}, convert(CuPtr{ElT}, buf), length(dV); own=true)
  else
      dV_ = CuArrays.zeros(ElT, length(dV))
  end
  copyto!(dV_, vec(dV))
  U = Tensor(Dense(dV_),Uinds)
  D = Tensor(Diag(real.(DM)),Dinds)
  return U,D,spec
end

function LinearAlgebra.qr(T::CuDenseTensor{ElT,2,IndsT}) where {ElT,IndsT}
  QM,RM = qr(matrix(T))
  # Make the new indices to go onto Q and R
  q,r = inds(T)
  q = dim(q) < dim(r) ? sim(q) : sim(r)
  Qinds = IndsT((ind(T,1),q))
  Rinds = IndsT((q,ind(T,2)))
  QM = CuMatrix(QM)
  unified = CUDAdrv.is_managed(data(store(T)).ptr)
  local Q_, R_
  if unified
      qbuf = CUDAdrv.Mem.alloc(CUDAdrv.Mem.UnifiedBuffer, length(QM) * sizeof(ElT))
      Q_   = unsafe_wrap(CuVector{ElT}, convert(CuPtr{ElT}, qbuf), length(QM); own=true)
      rbuf = CUDAdrv.Mem.alloc(CUDAdrv.Mem.UnifiedBuffer, length(RM) * sizeof(ElT))
      R_   = unsafe_wrap(CuVector{ElT}, convert(CuPtr{ElT}, rbuf), length(RM); own=true)
  else
      Q_ = CuArrays.zeros(ElT, length(QM))
      R_ = CuArrays.zeros(ElT, length(RM))
  end
  copyto!(Q_, vec(QM))
  copyto!(R_, vec(RM))
  Q = Tensor(Dense(Q_),Qinds)
  R = Tensor(Dense(R_),Rinds)
  return Q,R
end

function polar(T::CuDenseTensor{ElT,2,IndsT}) where {ElT,IndsT}
  QM,RM = polar(matrix(T))
  dim = size(QM,2)
  # Make the new indices to go onto Q and R
  q = eltype(IndsT)(dim)
  # TODO: use push/pushfirst instead of a constructor
  # call here
  Qinds = IndsT((ind(T,1),q))
  Rinds = IndsT((q,ind(T,2)))
  Q = Tensor(Dense(vec(QM)),Qinds)
  R = Tensor(Dense(vec(RM)),Rinds)
  return Q,R
end

