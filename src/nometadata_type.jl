#: A concrete subtype of AbstractMetaDataArray

export AbstractTemplateArray, TemplateArray, CuTemplateArray, raw_data, meta_data, join_metadata, metadata_array


# Abstract type

abstract type AbstractTemplateArray{T,N}<:AbstractMetaDataArray{Nothing,T,N} end


# Concrete types

struct TemplateArray{T,N}<:AbstractTemplateArray{T,N}
    array::Array{T,N}
end

raw_data(x::TemplateArray) = x.array
meta_data(::TemplateArray) = nothing
join_metadata(::TemplateArray{T,N},::TemplateArray{T,N}) where {T,N} = nothing
metadata_array(x::Array{T,N}, ::Nothing) where {T,N} = TemplateArray{T,N}(x)

struct CuTemplateArray{T,N}<:AbstractTemplateArray{T,N}
    array::CuArray{T,N}
end

raw_data(x::CuTemplateArray) = x.array
meta_data(::CuTemplateArray) = nothing
join_metadata(::CuTemplateArray{T,N},::CuTemplateArray{T,N}) where {T,N} = nothing
metadata_array(x::CuArray{T,N}, ::Nothing) where {T,N} = CuTemplateArray{T,N}(x)


# Utils

TemplateArray(array::AbstractArray{T,N}) where {T,N} = metadata_array(array, nothing)
Base.similar(x::AbstractTemplateArray{T,N}) where {T,N} = metadata_array(similar(raw_data(x)), nothing)


# CUDA utils

Flux.gpu(x::TemplateArray{T,N}) where {T,N} = CuTemplateArray{T,N}(gpu(raw_data(x)))
Flux.cpu(x::CuTemplateArray{T,N}) where {T,N} = TemplateArray{T,N}(cpu(raw_data(x)))
