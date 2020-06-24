function cuITensor(::Type{T},inds::IndexSet) where {T<:Number}
    return ITensor(Dense{float(T)}(CuArrays.zeros(float(T),dim(inds))), inds)
end
cuITensor(::Type{T},inds::Index...) where {T<:Number} = cuITensor(T,IndexSet(inds...))

cuITensor(is::IndexSet)   = cuITensor(Float64,is)
cuITensor(inds::Index...) = cuITensor(IndexSet(inds...))

cuITensor() = ITensor()
function cuITensor(x::S, inds::IndexSet{N}) where {S<:Number, N}
    dat = CuVector{float(S)}(undef, dim(inds))
    fill!(dat, float(x))
    ITensor(Dense{S}(dat), inds)
end
cuITensor(x::S, inds::Index...) where {S<:Number} = cuITensor(x,IndexSet(inds...))

#TODO: check that the size of the Array matches the Index dimensions
function cuITensor(A::Array{S},inds::IndexSet) where {S<:Number}
    return ITensor(Dense(CuArray{S}(A)), inds)
end
function cuITensor(A::CuArray{S},inds::IndexSet) where {S<:Number}
    return ITensor(Dense(A), inds)
end
cuITensor(A::Array{S},   inds::Index...) where {S<:Number} = cuITensor(A,IndexSet(inds...))
cuITensor(A::CuArray{S}, inds::Index...) where {S<:Number} = cuITensor(A,IndexSet(inds...))
cuITensor(A::ITensor) = store(tensor(A)) isa ITensors.Empty ? cuITensor(A.inds) : cuITensor(data(store(tensor(A))), A.inds)

function Base.collect(A::ITensor)
    typeof(A.store.data) <: CuArray && return ITensor(collect(A.store), A.inds)    
    return A
end

function randomCuITensor(::Type{S},inds::IndexSet) where {S<:Real}
  T = cuITensor(S,inds)
  randn!(T)
  return T
end
function randomCuITensor(::Type{S},inds::IndexSet) where {S<:Complex}
  Tr = cuITensor(real(S),inds)
  randn!(Tr)
  Ti = cuITensor(real(S),inds)
  randn!(Ti)
  return complex(Tr) + im.*complex(Ti)
end
randomCuITensor(::Type{S},inds::Index...) where {S<:Number} = randomCuITensor(S,IndexSet(inds...))
randomCuITensor(inds::IndexSet) = randomCuITensor(Float64,inds)
randomCuITensor(inds::Index...) = randomCuITensor(Float64,IndexSet(inds...))

CuArray(T::ITensor) = CuArray(tensor(T))

CuArray(T::ITensor,ninds::Index...) = storage_convert(CuArray,store(T),inds(T),IndexSet(ninds))

CuArrays.CuMatrix(A::ITensor) = CuArray(A)

function CuVector(A::ITensor)
  if ndims(A) != 1
    throw(DimensionMismatch("Vector() expected a 1-index ITensor"))
  end
  return CuArray(A)
end

function CuArray(T::ITensor{N},is::Vararg{Index,N}) where {N}
  perm = getperm(inds(T),is)
  return CuArray(permutedims(tensor(T),perm))
end

function CuMatrix(T::ITensor{N},i1::Index,i2::Index) where {N}
  N≠2 && throw(DimensionMismatch("ITensor must be order 2 to convert to a Matrix"))
  return CuArray(T,i1,i2)
end

