module ITensorsGPU

using CuArrays
using CuArrays.CUTENSOR
using CuArrays.CUBLAS
using CuArrays.CUSOLVER
using LinearAlgebra
using Random
using TimerOutputs
using StaticArrays
using ITensors
import CuArrays: CuArray, CuMatrix, CuVector
import CuArrays.CUTENSOR: cutensorContractionPlan_t, cutensorAlgo_t
import ITensors: randn!, compute_contraction_labels,
                 plussers, eigen, similar_type,
                 scale!, getperm, unioninds, array, matrix, vector,
                 polar, tensors, truncate!
import ITensors.NDTensors: ContractionProperties, contract!!, _contract!!, _contract!, contract!, contract,
                         contraction_output, UniformDiagTensor, CombinerTensor, contraction_output_type,
                         UniformDiag, Diag, DiagTensor, NonuniformDiag, NonuniformDiagTensor, zero_contraction_output,
                         outer!, outer!!, is_trivial_permutation, ind, permutedims!!, Dense, DenseTensor, Combiner,
                         Tensor
import Base.*
include("tensor/cudense.jl")
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
