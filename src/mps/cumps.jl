function cuMPS(psi::MPS)
    phi = copy(psi)
    for site in 1:length(psi)
        phi.data[site] = cuITensor(psi.data[site])
    end
    return phi
end

cu(ψ::MPS) = cuMPS(ψ)

cuMPS() = MPS() 
function cuMPS(sites) # random MPS
  N = length(sites)
  v = Vector{ITensor}(undef, N)
  l = [Index(1, "Link,l=$ii") for ii=1:N-1]
  for ii in eachindex(sites)
    s = sites[ii]
    if ii == 1
      v[ii] = cuITensor(l[ii], s)
    elseif ii == N
      v[ii] = cuITensor(l[ii-1], s)
    else
      v[ii] = cuITensor(l[ii-1],s,l[ii])
    end
  end
  return MPS(v,0,N+1)
end

function randomCuMPS(sites)
  M = cuMPS(sites)
  for i in eachindex(sites)
    randn!(M[i])
    normalize!(M[i])
  end
  M.llim = 1
  M.rlim = length(M)
  return M
end

function productCuMPS(::Type{T}, ivals::Vector{<:IndexVal}) where {T<:Number}
  N     = length(ivals)
  As    = Vector{ITensor}(undef,N)
  links = Vector{Index}(undef,N)
  for n=1:N
    s = ITensors.ind(ivals[n])
    links[n] = Index(1,"Link,l=$n")
    if n == 1
      A = ITensor(T, s, links[n])
      A[ivals[n],links[n](1)] = 1.0
    elseif n == N
      A = ITensor(T, links[n-1], s)
      A[links[n-1](1),ivals[n]] = 1.0
    else
      A = ITensor(T, links[n-1], s, links[n])
      A[links[n-1](1),ivals[n],links[n](1)] = 1.0
    end
    As[n] = cuITensor(A)
  end
  return MPS(As,0,2)
end
productCuMPS(ivals::Vector{<:IndexVal}) = productCuMPS(Float64, ivals)

function productCuMPS(::Type{T}, sites,
                      states) where {T<:Number}
  if length(sites) != length(states)
    throw(DimensionMismatch("Number of sites and and initial states don't match"))
  end
  ivals = [state(sites[n],states[n]) for n=1:length(sites)]
  return productCuMPS(T, ivals)
end

productCuMPS(sites, states) = productCuMPS(Float64, sites, states)
