Base.promote_rule(::Type{<:Combiner},StorageT::Type{<:CuDense}) = StorageT
#=function contraction_output_type(TensorT1::Type{<:CombinerTensor},
                                 TensorT2::Type{<:CuDenseTensor},
                                 indsR)
  return similar_type(promote_type(TensorT1,TensorT2),indsR)
end

function contraction_output_type(TensorT1::Type{<:CuDenseTensor},
                                 TensorT2::Type{<:CombinerTensor},
                                 indsR)
  return contraction_output_type(TensorT2,TensorT1,indsR)
end

function contraction_output(TensorT1::Type{<:CombinerTensor},
                            TensorT2::Type{<:CuDenseTensor},
                            indsR)
  return similar(contraction_output_type(TensorT1,TensorT2,indsR),indsR)
end

function contraction_output(TensorT1::Type{<:CuDenseTensor},
                            TensorT2::Type{<:CombinerTensor},
                            indsR)
  return contraction_output(TensorT2,TensorT1,indsR)
end=#
