module DataWrangling

export 
    get_SOBLLES_data_path, get_LES_3D_fields_data_path, 
    get_losses_data_path, get_ODE_inference_data_path, 
    get_oceananigans_column_data_path, get_long_integration_data_path,
    get_timestep_data_path, get_doublegyre_data_path,
    ZeroMeanUnitVarianceScaling, MinMaxScaling, DiffusivityScaling,
    LESData, LESDatasets,
    ODEParam, ODEParams,
    write_scaling_params, construct_zeromeanunitvariance_scaling,
    get_nn_datasets, get_baseclosure_datasets

include("register_datadep.jl")
include("feature_scaling.jl")
include("training_data_processing.jl")
include("coarse_graining.jl")
include("model_params.jl")
include("training_data_extraction.jl")

end