Base.promote_rule(::Type{<:Combiner},StorageT::Type{<:CuDense}) = StorageT

function contraction_output(T1::TensorT1,
                            T2::TensorT2,
                            indsR::IndsR) where {TensorT1<:CombinerTensor,
                                                 TensorT2<:CuDenseTensor,
                                                 IndsR}
  TensorR = contraction_output_type(TensorT1,TensorT2,IndsR)
  unified = CUDAdrv.is_managed(data(store(T2)).ptr)
  return similar(TensorR,indsR; unified=unified)
end
