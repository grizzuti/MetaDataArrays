#: A concrete subtype of AbstractMetaDataArray

export MetaDataArray, CuMetaDataArray, raw_data, meta_data, join_metadata, NoMetaDataArray


"""
Concrete meta-data type
"""
struct MetaDataArray{ADT,MDT,T,N}<:AbstractMetaDataArray{ADT,MDT,T,N}
    array::ADT
    meta_data::MDT
end

raw_data(x::MetaDataArray) = x.array
meta_data(x::MetaDataArray) = x.meta_data
join_metadata(::Nothing, ::Nothing) = nothing


# Utils

NoMetaDataArray(array::Array{T,N}) where {T,N} = MetaDataArray{Array{T,N},Nothing,T,N}(array, nothing)
NoMetaDataArray(array::CuArray{T,N}) where {T,N} = MetaDataArray{CuArray{T,N},Nothing,T,N}(array, nothing)