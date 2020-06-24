module ITensorsGPU

using CuArrays, CUDAdrv
using CuArrays.CUTENSOR
using CuArrays.CUBLAS
using CuArrays.CUSOLVER
using LinearAlgebra
using Random
using TimerOutputs
using StaticArrays
using ITensors
using Strided
import CuArrays: CuArray, CuMatrix, CuVector
import CuArrays.CUTENSOR: cutensorContractionPlan_t, cutensorAlgo_t

using CUDAdrv
import CUDAdrv.Mem: pin
#=
const devs = Ref{Vector{CUDAdrv.CuDevice}}()
const dev_rows = Ref{Int}(0)
const dev_cols = Ref{Int}(0)
function __init__()
  voltas    = filter(dev->occursin("V100", CUDAdrv.name(dev)), collect(CUDAdrv.devices()))
  pascals    = filter(dev->occursin("P100", CUDAdrv.name(dev)), collect(CUDAdrv.devices()))
  devs[] = voltas[1:1]
  #devs[] = pascals[1:2]
  CUBLASMG.cublasMgDeviceSelect(CUBLASMG.mg_handle(), length(devs[]), devs[])
  dev_rows[] = 1
  dev_cols[] = 1
end
=#
import ITensors: randn!, compute_contraction_labels,
                 plussers, eigen, similar_type, tensor,
                 scale!, unioninds, array, matrix, vector,
                 polar, tensors, truncate!, leftlim, rightlim
import ITensors.NDTensors: ContractionProperties, contract!!, _contract!!, _contract!, contract!, contract,
                         contraction_output, UniformDiagTensor, CombinerTensor, contraction_output_type,
                         UniformDiag, Diag, DiagTensor, NonuniformDiag, NonuniformDiagTensor, zero_contraction_output,
                         outer!, outer!!, is_trivial_permutation, ind, permutedims!!, Dense, DenseTensor, Combiner,
                         Tensor, data, permute, getperm, compute_contraction_properties!, Atrans, Btrans, Ctrans
import Base.*, Base.permutedims!
include("tensor/cudense.jl")
include("tensor/dense.jl")
include("tensor/culinearalgebra.jl")
include("tensor/cutruncate.jl")
include("tensor/cucombiner.jl")
include("tensor/cudiag.jl")
include("cuitensor.jl")
include("mps/cumps.jl")
include("mps/cumpo.jl")

#const ContractionPlans = Dict{String, Tuple{cutensorAlgo_t, cutensorContractionPlan_t}}()
const ContractionPlans = Dict{String, cutensorAlgo_t}()

export cuITensor,
       randomCuITensor,
       cuMPS,
       randomCuMPS,
       productCuMPS,
       randomCuMPO,
       cuMPO
end #module
