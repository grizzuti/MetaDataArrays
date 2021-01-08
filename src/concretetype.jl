#: A concrete subtype of AbstractMetaDataArray

export MetaDataArray, CuMetaDataArray, raw_data, meta_data, join_metadata, metadata_array, NoMetaDataArray


"""
Concrete meta-data type
"""
struct MetaDataArray{MDT,T,N}<:AbstractMetaDataArray{MDT,T,N}
    array::Array{T,N}
    meta_data::MDT
end

raw_data(x::MetaDataArray) = x.array
meta_data(x::MetaDataArray) = x.meta_data
join_metadata(::MetaDataArray{Nothing,T,N}, ::MetaDataArray{Nothing,T,N}) where {T,N} = nothing # only defined here for MetaDataArray{Nothing,T,N}; specify this function for your custom type
metadata_array(array::Array{T,N}, meta_data::MDT) where {MDT,T,N} = MetaDataArray{MDT,T,N}(array, meta_data)

struct CuMetaDataArray{MDT,T,N}<:AbstractMetaDataArray{MDT,T,N}
    array::CuArray{T,N}
    meta_data::MDT
end

raw_data(x::CuMetaDataArray) = x.array
meta_data(x::CuMetaDataArray) = x.meta_data
join_metadata(::CuMetaDataArray{Nothing,T,N},::CuMetaDataArray{Nothing,T,N}) where {T,N} = nothing
metadata_array(array::CuArray{T,N}, meta_data::MDT) where {MDT,T,N} = CuMetaDataArray{MDT,T,N}(array, meta_data)


# Utils

NoMetaDataArray(array::AbstractArray{T,N}) where {T,N} = metadata_array(array, nothing)
Base.similar(x::AbstractMetaDataArray{Nothing,T,N}) where {T,N} = metadata_array(similar(raw_data(x)), nothing)


# CUDA utils

Flux.gpu(x::AbstractMetaDataArray{Nothing,T,N}) where {T,N} = metadata_array(gpu(raw_data(x)), nothing)
Flux.cpu(x::AbstractMetaDataArray{Nothing,T,N}) where {T,N} = metadata_array(cpu(raw_data(x)), nothing)
