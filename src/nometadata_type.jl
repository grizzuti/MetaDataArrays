#: A concrete subtype of AbstractMetaDataArray

export AbstractNoMetaDataArray, NoMetaDataArray, CuNoMetaDataArray, raw_data, meta_data, join_metadata, metadata_array


# Abstract type

abstract type AbstractNoMetaDataArray{T,N}<:AbstractMetaDataArray{Nothing,T,N} end


# Concrete types

struct NoMetaDataArray{T,N}<:AbstractNoMetaDataArray{T,N}
    array::Array{T,N}
end

raw_data(x::NoMetaDataArray) = x.array
meta_data(::NoMetaDataArray) = nothing
join_metadata(::NoMetaDataArray{T,N},::NoMetaDataArray{T,N}) where {T,N} = nothing
metadata_array(x::Array{T,N}, ::Nothing) where {T,N} = NoMetaDataArray{T,N}(x)

struct CuNoMetaDataArray{T,N}<:AbstractNoMetaDataArray{T,N}
    array::CuArray{T,N}
end

raw_data(x::CuNoMetaDataArray) = x.array
meta_data(::CuNoMetaDataArray) = nothing
join_metadata(::CuNoMetaDataArray{T,N},::CuNoMetaDataArray{T,N}) where {T,N} = nothing
metadata_array(x::CuArray{T,N}, ::Nothing) where {T,N} = CuNoMetaDataArray{T,N}(x)


# Utils

NoMetaDataArray(array::AbstractArray{T,N}) where {T,N} = metadata_array(array, nothing)
Base.similar(x::AbstractNoMetaDataArray{T,N}) where {T,N} = metadata_array(similar(raw_data(x)), nothing)


# CUDA utils

Flux.gpu(x::NoMetaDataArray{T,N}) where {T,N} = CuNoMetaDataArray{T,N}(gpu(raw_data(x)))
Flux.cpu(x::CuNoMetaDataArray{T,N}) where {T,N} = NoMetaDataArray{T,N}(cpu(raw_data(x)))
