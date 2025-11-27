module NORiOceanParameterization

export
    find_min, find_max,
    calculate_Ri, calculate_Ri!,
    Dᶜ, Dᶠ, Dᶜ!, Dᶠ!, D²ᶜ, Iᶠ,
    get_SOBLLES_data_path, get_LES_3D_fields_data_path, 
    get_losses_data_path, get_ODE_inference_data_path, 
    get_oceananigans_column_data_path, get_long_integration_data_path,
    get_timestep_data_path, get_doublegyre_data_path,
    ZeroMeanUnitVarianceScaling, MinMaxScaling, DiffusivityScaling,
    LESData, LESDatasets,
    ODEParam, ODEParams,
    write_scaling_params, construct_zeromeanunitvariance_scaling,
    get_nn_datasets, get_baseclosure_datasets,
    local_Ri_ν, local_Ri_κ,
    nonlocal_Ri_ν, nonlocal_Ri_κ,
    predict_boundary_flux, predict_boundary_flux!,
    predict_diffusivities, predict_diffusivities!,
    predict_diffusive_flux, predict_diffusive_boundary_flux_dimensional,
    compute_density_contribution,
    animate_data, plot_loss,
    BaseClosureMode, NNMode,
    solve_NDE, diagnose_fields,
    individual_loss, loss, compute_loss_prefactor_density_contribution, loss_multipleics,
    predict_residual_flux, predict_residual_flux_dimensional,
    NORiBaseVerticalDiffusivity, NORiNNFluxClosure,
    NORiClosureTuple, NORiBaseClosureOnly, NORiClosureWithNN

include("Operators/Operators.jl")
include("Utils/Utils.jl")
include("DataWrangling/DataWrangling.jl")
include("Closures/Closures.jl")
include("ODEOperations/ODEOperations.jl")
include("TrainingOperations/TrainingOperations.jl")
include("Plotting/Plotting.jl")
include("Implementation/Implementation.jl")

using .Operators
using .DataWrangling
using .Closures
using .ODEOperations
using .TrainingOperations
using .Utils
using .Plotting
using .Implementation

end
