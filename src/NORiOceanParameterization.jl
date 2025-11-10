module NORiOceanParameterization

export
    find_min, find_max,
    calculate_Ri, calculate_Ri!,
    Dᶜ, Dᶠ, Dᶜ!, Dᶠ!, D²ᶜ, Iᶠ,
    ZeroMeanUnitVarianceScaling, MinMaxScaling, DiffusivityScaling,
    LESData, LESDatasets,
    ODEParam, ODEParams,
    write_scaling_params, construct_zeromeanunitvariance_scaling,
    get_SOBLLES_data_path,
    local_Ri_ν, local_Ri_κ,
    nonlocal_Ri_ν, nonlocal_Ri_κ,
    predict_boundary_flux, predict_boundary_flux!,
    compute_density_contribution

include("utils.jl")
include("Operators/Operators.jl")
include("DataWrangling/DataWrangling.jl")
include("Closures/Closures.jl")
include("ODEOperations/ODEOperations.jl")
include("TrainingOperations/TrainingOperations.jl")

using .Operators
using .DataWrangling
using .Closures
using .ODEOperations
using .TrainingOperations

end
