module NORiImplementation

using NORiOceanParameterization.Utils: load_nn_weights, load_baseclosure_params
using NORiOceanParameterization.DataWrangling: construct_zeromeanunitvariance_scaling

export
    NORiBaseVerticalDiffusivity, NORiNNFluxClosure,
    NORiClosureTuple, NORiBaseClosureOnly, NORiClosureWithNN

include("NORiBaseVerticalDiffusivity.jl")
include("NORiNNFluxClosure.jl")
include("closure_builder.jl")

end # module
