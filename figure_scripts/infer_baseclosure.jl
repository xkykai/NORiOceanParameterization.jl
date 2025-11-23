using NORiOceanParameterization
using NORiOceanParameterization.TrainingOperations
using Oceananigans
using JLD2
using SeawaterPolynomials.TEOS10
using ComponentArrays

mode = BaseClosureMode()

SOBLLES_path = joinpath(get_SOBLLES_data_path(), "SOBLLES_jld2")
data_path = joinpath(SOBLLES_path, "data")
cases = readdir(data_path)
field_datasets = [FieldDataset(joinpath(data_path, sim, "instantaneous_timeseries.jld2"), backend=OnDisk()) for sim in cases]

coarse_size = 32
full_timeframes = [25:length(data["ubar"].times) for data in field_datasets]
dataset = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)
scaling = dataset.scaling

x₀s = [(; u=data.profile.u.scaled[:, 1], v=data.profile.v.scaled[:, 1], T=data.profile.T.scaled[:, 1], S=data.profile.S.scaled[:, 1]) for data in dataset.data]
params = ODEParams(dataset, scaling, abs_f=true)
ps = jldopen(joinpath(pwd(), "calibrated_parameters", "baseclosure_final.jld2"), "r")["u"]

results = [diagnose_fields(mode, ps, param, x₀, data) for (param, x₀, data) in zip(params, x₀s, dataset.data)]

SAVE_PATH = joinpath(pwd(), "figure_data")
mkpath(SAVE_PATH)
filepath = joinpath(SAVE_PATH, "baseclosure_inference_results.jld2")
jldopen(filepath, "w") do file
    for (i, case) in enumerate(cases)
        file[case] = results[i]
    end
end