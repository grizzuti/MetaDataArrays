#: Abstract type: AbstractMetaDataArray


export AbstractMetaDataArray


# Abstract type

"""
AbstractMetaDataArray is simply a wrapper for arrays + metadata.
Expected behavior for AbstractMetaDataArray:
    - raw_data(x::AbstractMetaDataArray{MDT,T,N})::some_type<:AbstractArray{T,N} -> Return raw data as an abstract array for which basic linear algebra has been defined
    - meta_data(x::AbstractMetaDataArray{MDT,T,N})::MDT
    - join_metadata(x::AbstractMetaDataArray{MDT,T,N}, y::AbstractMetaDataArray{MDT,T,N})::MDT
    - metadata_array(raw_data::some_type, meta_data::MDT)<:AbstractMetaDataArray{MDT,T,N}
"""
abstract type AbstractMetaDataArray{MDT,T<:Number,N}<:AbstractArray{T,N} end


# Size/dimension

Base.size(x::AbstractMetaDataArray) = size(raw_data(x))
Base.ndims(::AbstractMetaDataArray{MDT,T,N}) where {MDT,T,N} = N


# Copying

Base.copy!(x::TF, y::TF) where {TF<:AbstractMetaDataArray} = (copy!(meta_data(x), meta_data(y)); copy!(raw_data(x), raw_data(y)); return x)
Base.copy(x::AbstractMetaDataArray) = metadata_array(copy(meta_data(x)), copy(raw_data(x)))


# Indexing

Base.IndexStyle(::Type{<:AbstractMetaDataArray}) = IndexLinear()
Base.getindex(x::AbstractMetaDataArray, i::Int64) = getindex(raw_data(x), i)
Base.setindex!(x::AbstractMetaDataArray{MDT,T,N}, val::T, i::Int64) where {MDT,T,N} = setindex!(raw_data(x), val, i)


# Show

Base.show(io::IO, x::AbstractMetaDataArray) = show(io, raw_data(x))
Base.show(io::IO, mime::MIME"text/plain", x::AbstractMetaDataArray) = show(io, mime, raw_data(x))


# Linear algebra

## x1+x2
+(x1::TF, x2::TF) where {TF<:AbstractMetaDataArray} = metadata_array(raw_data(x1)+raw_data(x2), join_metadata(x1,x2))

## x1.+c, c.+x1
Base.broadcasted(::typeof(+), x::AbstractMetaDataArray, c::Number) = metadata_array(raw_data(x).+c, meta_data(x))
Base.broadcasted(::typeof(+), c::Number, x::AbstractMetaDataArray) = x.+c

## x1-x2
-(x1::TF, x2::TF) where {TF<:AbstractMetaDataArray} = metadata_array(raw_data(x1)-raw_data(x2), join_metadata(x1,x2))

## -x
-(x::AbstractMetaDataArray) = metadata_array(-raw_data(x), meta_data(x))

## x.-c, c.-x
Base.broadcasted(::typeof(-), x::AbstractMetaDataArray, c::Number) = metadata_array(raw_data(x).-c, meta_data(x))
Base.broadcasted(::typeof(-), c::Number, x::AbstractMetaDataArray) = x.-c

## x1.*x2
Base.broadcasted(::typeof(*), x1::TF, x2::TF) where {TF<:AbstractMetaDataArray} = metadata_array(raw_data(x1).*raw_data(x2), join_metadata(x1,x2))

## c*x, x*c
*(c::Number, x::AbstractMetaDataArray) = metadata_array(c*raw_data(x), meta_data(x))
*(x::AbstractMetaDataArray, c::Number) = c*x
LinearAlgebra.lmul!(c::Number, x::AbstractMetaDataArray) = (lmul!(c, raw_data(x)); return x)
LinearAlgebra.rmul!(x::AbstractMetaDataArray, c::Number) = (rmul!(raw_data(x), c); return x)

## c.*x, x.*c
Base.broadcasted(::typeof(*), c::Number, x::AbstractMetaDataArray) = metadata_array(c.*raw_data(x), meta_data(x))
Base.broadcasted(::typeof(*), x::AbstractMetaDataArray, c::Number) = c.*x

## x1./x2
Base.broadcasted(::typeof(/), x1::TF, x2::TF) where {TF<:AbstractMetaDataArray} = metadata_array(raw_data(x1)./raw_data(x2), join_metadata(x1,x2))

## x/c
/(x::AbstractMetaDataArray, c::Number) = metadata_array(raw_data(x)/c, meta_data(x))
LinearAlgebra.ldiv!(c::Number, x::AbstractMetaDataArray) = (ldiv!(c, raw_data(x)); return x)
LinearAlgebra.rdiv!(x::AbstractMetaDataArray, c::Number) = (rdiv!(raw_data(x), c); return x)

## x./c, c./x
Base.broadcasted(::typeof(/), x::AbstractMetaDataArray, c::Number) = metadata_array(raw_data(x)./c, meta_data(x))
Base.broadcasted(::typeof(/), c::Number, x::AbstractMetaDataArray) = metadata_array(c./raw_data(x), meta_data(x))

## conj(x)
Base.conj(x::AbstractMetaDataArray) = metadata_array(conj(raw_data(x)), meta_data(x))

## dot(x1,x2)
LinearAlgebra.dot(x1::TF, x2::TF) where {TF<:AbstractMetaDataArray} = dot(raw_data(x1), raw_data(x2))

## norm(x)
LinearAlgebra.norm(x::AbstractMetaDataArray, p::Real) = norm(raw_data(x), p)

## abs.(x)
Base.broadcasted(::typeof(abs), x::AbstractMetaDataArray) = metadata_array(abs.(raw_data(x)), meta_data(x))

## fill!
Base.fill!(x::AbstractMetaDataArray{MDT,T,N}, c::T) where {MDT,T,N} = (fill!(raw_data(x), c); return x)

## sum(x)
Base.sum(x::AbstractMetaDataArray; dims) = metadata_array(sum(raw_data(x); dims=dims), meta_data(x))


# Utils

function Base.isapprox(x::TF, y::TF; rtol::Real=sqrt(eps()), atol::Real=0) where {TF<:AbstractMetaDataArray}
    join_metadata(x,y)
    return isapprox(raw_data(x), raw_data(y); rtol=rtol, atol=atol)
end