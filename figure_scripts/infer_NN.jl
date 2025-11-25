using NORiOceanParameterization
using NORiOceanParameterization.TrainingOperations
using Oceananigans
using JLD2
using SeawaterPolynomials.TEOS10
using ComponentArrays
using Lux

mode = NNMode()
coarse_size = 32

SOBLLES_path = joinpath(get_SOBLLES_data_path(), "SOBLLES_jld2")
data_path = joinpath(SOBLLES_path, "data")
nn_split_path = joinpath(SOBLLES_path, "nn_dataset_split.json")
dataset_split = get_nn_datasets(nn_split_path)

training_field_datasets = [FieldDataset(joinpath(data_path, sim, "instantaneous_timeseries.jld2"), backend=OnDisk()) for sim in dataset_split[:training]]
validation_field_datasets = [FieldDataset(joinpath(data_path, sim, "instantaneous_timeseries.jld2"), backend=OnDisk()) for sim in dataset_split[:validation]]

training_timeframes = [25:length(data["ubar"].times) for data in training_field_datasets]
validation_timeframes = [25:length(data["ubar"].times) for data in validation_field_datasets]

ps_baseclosure = jldopen(joinpath(pwd(), "calibrated_parameters", "baseclosure_final.jld2"), "r")["u"]
nn_model_path = joinpath(pwd(), "calibrated_parameters", "NNclosure_weights.jld2")
ps, sts, scaling_params, wT_model, wS_model = jldopen(nn_model_path, "r") do file
    ps = file["u"]
    sts = file["sts"]
    scaling_params = file["scaling"]
    wT_model = file["model"].wT
    wS_model = file["model"].wS
    return ps, sts, scaling_params, wT_model, wS_model
end

NNs = (wT=wT_model, wS=wS_model)
scaling = construct_zeromeanunitvariance_scaling(scaling_params)

NN_grid_points = 5
grid_point_below_kappa = 5
grid_point_above_kappa = 10
Riᶜ = ps_baseclosure.Riᶜ
window_split_size = NN_grid_points ÷ 2

training_dataset = LESDatasets(training_field_datasets, scaling, training_timeframes)
validation_dataset = LESDatasets(validation_field_datasets, scaling, validation_timeframes)

training_params = ODEParams(training_dataset, scaling)
validation_params = ODEParams(validation_dataset, scaling)

x₀s_training = [(; u = data.profile.u.scaled[:, 1],
                   v = data.profile.v.scaled[:, 1],
                   T = data.profile.T.scaled[:, 1],
                   S = data.profile.S.scaled[:, 1]) for data in training_dataset.data]

x₀s_validation = [(; u = data.profile.u.scaled[:, 1],
                     v = data.profile.v.scaled[:, 1],
                     T = data.profile.T.scaled[:, 1],
                     S = data.profile.S.scaled[:, 1]) for data in validation_dataset.data]

training_results = [diagnose_fields(mode, ps, param, x₀, ps_baseclosure,
                                    sts, NNs, data, length(timeframe),
                                    grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, 2)
                                    for (param, x₀, data, timeframe) in zip(training_params, x₀s_training, training_dataset.data, training_timeframes)]

validation_results = [diagnose_fields(mode, ps, param, x₀, ps_baseclosure,
                                      sts, NNs, data, length(timeframe),
                                      grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, 2)
                                      for (param, x₀, data, timeframe) in zip(validation_params, x₀s_validation, validation_dataset.data, validation_timeframes)]

SAVE_PATH = joinpath(pwd(), "figure_data")
mkpath(SAVE_PATH)
filepath = joinpath(SAVE_PATH, "NN_inference_results.jld2")
jldopen(filepath, "w") do file
    for (i, case) in enumerate(dataset_split[:training])
        file["training/$(case)"] = training_results[i]
    end

    for (i, case) in enumerate(dataset_split[:validation])
        file["validation/$(case)"] = validation_results[i]
    end
end