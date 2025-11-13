using NORiOceanParameterization
using NORiOceanParameterization.TrainingOperations
using LinearAlgebra
using ComponentArrays, Random
using Printf
using Oceananigans
using JLD2
using SeawaterPolynomials.TEOS10
using CairoMakie
import Dates
using Statistics
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

mode = BaseClosureMode()
training_data_subset = :training_convection

SOBLLES_path = joinpath(get_SOBLLES_data_path(), "SOBLLES_jld2")
data_path = joinpath(SOBLLES_path, "data")
baseclosure_split_path = joinpath(SOBLLES_path, "baseclosure_dataset_split.json")

baseclosure_dataset_split = get_baseclosure_datasets(baseclosure_split_path)

field_datasets = [FieldDataset(joinpath(data_path, sim, "instantaneous_timeseries.jld2"), backend=OnDisk()) for sim in baseclosure_dataset_split[training_data_subset]]

timeframes = [25:10:length(data["ubar"].times) for data in field_datasets]
fulltimeframes = [25:length(data["ubar"].times) for data in field_datasets]
coarse_size = 32
training_data = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, timeframes, coarse_size)
scaling = training_data.scaling

truths = [(;    u = data.profile.u.scaled,
                v = data.profile.v.scaled,
                T = data.profile.T.scaled,
                S = data.profile.S.scaled,
                ρ = data.profile.ρ.scaled,
             ∂u∂z = data.profile.∂u∂z.scaled,
             ∂v∂z = data.profile.∂v∂z.scaled,
             ∂T∂z = data.profile.∂T∂z.scaled,
             ∂S∂z = data.profile.∂S∂z.scaled,
             ∂ρ∂z = data.profile.∂ρ∂z.scaled) for data in training_data.data]

x₀s = [(; u = data.profile.u.scaled[:, 1],
          v = data.profile.v.scaled[:, 1],
          T = data.profile.T.scaled[:, 1],
          S = data.profile.S.scaled[:, 1]) for data in training_data.data]

params = ODEParams(training_data)

rng = Random.default_rng(123)

# Load the optimal shear parameters from the shear training results
PS_PRIOR_DIR = joinpath(pwd(), "calibrated_parameters", "baseclosure_shear.jld2")
ps = jldopen(PS_PRIOR_DIR, "r")["u"]

ps_prior = ps
ps_fixed = ComponentArray(ν_shear=ps.ν_shear, Riᶜ=ps.Riᶜ, Pr_shear=ps.Pr_shear)

ind_losses = [individual_loss(mode, ps_prior, truth, param, x₀) for (truth, x₀, param) in zip(truths, x₀s, params)]
loss_prefactors = compute_loss_prefactor_density_contribution.(Ref(mode), ind_losses, compute_density_contribution.(training_data.data))

prior_loss = loss_multipleics(mode, ps_prior, truths, params, x₀s, loss_prefactors)

prior_ν_conv = constrained_gaussian("ν_conv", ps_prior.ν_conv, ps_prior.ν_conv/5, -Inf, Inf)
prior_Pr_conv = constrained_gaussian("Pr_conv", ps_prior.Pr_conv, ps_prior.Pr_conv/5, -Inf, Inf)
prior_ΔRi = constrained_gaussian("ΔRi", ps_prior.ΔRi, ps_prior.ΔRi/5, -Inf, Inf)

priors = combine_distributions([prior_ν_conv, prior_Pr_conv, prior_ΔRi])
target = [0.]

# N_ensemble = 400
# N_iterations = 200
N_ensemble = 2
N_iterations = 2
Γ = prior_loss / 1e6 * I

ps_eki = EKP.construct_initial_ensemble(rng, priors, N_ensemble)
ensemble_kalman_process = EKP.EnsembleKalmanProcess(ps_eki, target, Γ, Inversion();
                                                    rng = rng,
                                                    failure_handler_method = SampleSuccGauss(),
                                                    scheduler = DataMisfitController(on_terminate="continue"))
G_ens = zeros(1, N_ensemble)

losses = []

wall_clock = [time_ns()]
for i in 1:N_iterations
    global ps_eki .= get_ϕ_final(priors, ensemble_kalman_process)
    Threads.@threads for j in 1:N_ensemble
        ps_ensemble = ComponentArray(ν_conv=ps_eki[1, j], ν_shear=ps_fixed.ν_shear, Riᶜ=ps_fixed.Riᶜ, Pr_conv=ps_eki[2, j], Pr_shear=ps_fixed.Pr_shear, ΔRi=ps_eki[3, j])
        G_ens[j] = loss_multipleics(mode, ps_ensemble, truths, params, x₀s, loss_prefactors)
    end
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
    @printf("%s, Δt %s, iter %d/%d, loss average %6.10e, ν_conv %6.5e, Pr_conv %6.5e, ΔRi %6.5e\n",
            Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), i, N_iterations, mean(G_ens),
            mean(ps_eki[1, :]), mean(ps_eki[2, :]), mean(ps_eki[3, :]))
    push!(losses, mean(G_ens))
    wall_clock[1] = time_ns()
end

losses = Array{Float64}(losses)
losses = (; total=losses)
final_ensemble = get_ϕ_final(priors, ensemble_kalman_process)
ensemble_mean = vec(mean(final_ensemble, dims=2))
ensemble_losses = [loss_multipleics(mode, ComponentArray(ν_conv=p[1], ν_shear=ps_fixed.ν_shear, Riᶜ=ps_fixed.Riᶜ, Pr_conv=p[2], Pr_shear=ps_fixed.Pr_shear, ΔRi=p[3]),
                                    truths, params, x₀s, loss_prefactors) for p in eachcol(final_ensemble)]

ensemble_min = final_ensemble[:, argmin(ensemble_losses)]

ps_final_min = ComponentArray(; ν_conv=ensemble_min[1], ν_shear=ps_fixed.ν_shear, Riᶜ=ps_fixed.Riᶜ, Pr_conv=ensemble_min[2], Pr_shear=ps_fixed.Pr_shear, ΔRi=ensemble_min[3])

ensemble_mean_loss = loss_multipleics(mode, ComponentArray(ν_conv=ensemble_mean[1], ν_shear=ps_fixed.ν_shear, Riᶜ=ps_fixed.Riᶜ, Pr_conv=ensemble_mean[2], Pr_shear=ps_fixed.Pr_shear, ΔRi=ensemble_mean[3]),
                                      truths, params, x₀s, loss_prefactors)

ps_final_mean = ComponentArray(; ν_conv=ensemble_mean[1], ν_shear=ps_fixed.ν_shear, Riᶜ=ps_fixed.Riᶜ, Pr_conv=ensemble_mean[2], Pr_shear=ps_fixed.Pr_shear, ΔRi=ensemble_mean[3])

SAVE_DIR = joinpath(pwd(), "training_results", "localbaseclosure_convection")
RESULTS_DIR = joinpath(SAVE_DIR, "calibrated_parameters")
PLOTS_DIR = joinpath(SAVE_DIR, "animations")

mkpath(SAVE_DIR)
mkpath(RESULTS_DIR)
mkpath(PLOTS_DIR)

jldsave(joinpath(RESULTS_DIR, "training_results_mean.jld2"), u=ps_final_mean)
jldsave(joinpath(RESULTS_DIR, "training_results_min.jld2"), u=ps_final_min)

plot_loss(mode, losses, SAVE_DIR)

training_data_plot = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, fulltimeframes, coarse_size)
params_plot = ODEParams(training_data_plot, scaling)

for i in eachindex(params)
    @info "Animating simulation $(i)..."
    sols, fluxes, diffusivities = diagnose_fields(mode, ps_final_mean, params_plot[i], x₀s[i], training_data_plot.data[i])
    animate_data(mode, training_data_plot.data[i], diffusivities, sols, fluxes, diffusivities, i, PLOTS_DIR; suffix="_mean")
end

for i in eachindex(params)
    @info "Animating simulation $(i)..."
    sols, fluxes, diffusivities = diagnose_fields(mode, ps_final_min, params_plot[i], x₀s[i], training_data_plot.data[i])
    animate_data(mode, training_data_plot.data[i], diffusivities, sols, fluxes, diffusivities, i, PLOTS_DIR; suffix="_min")
end
