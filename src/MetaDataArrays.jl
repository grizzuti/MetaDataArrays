module MetaDataArrays

using LinearAlgebra, CUDA, Flux
import Base: +,-,*,/

include("abstracttype.jl")
include("concretetype.jl")

end
