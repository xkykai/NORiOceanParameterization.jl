using NORiOceanParameterization
using LinearAlgebra
using Lux, ComponentArrays, Random
using Printf
using Enzyme
using Oceananigans
using JLD2
using SeawaterPolynomials.TEOS10
using CairoMakie
using Optimisers
import Dates
using Statistics

const hidden_layer_size = 128
const N_hidden_layer = 3

const activation = relu
const activation_str = "relu"

const grid_point_below_kappa = 5
const grid_point_above_kappa = 10
seed = 20

const NN_grid_points = 5

# Create output directory
FILE_DIR = joinpath(@__DIR__, "..", "training_results", "NDE")
mkpath(FILE_DIR)

WEIGHTS_DIR = joinpath(@__DIR__, "..", "model_weights")
mkpath(WEIGHTS_DIR)

# Load base closure parameters
BASECLOSURE_FILE_DIR = joinpath(@__DIR__, "..", "calibrated_parameters", "baseclosure_final.jld2")
ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

const coarse_size = 32
const Riᶜ = ps_baseclosure.Riᶜ

# Load datasets
SOBLLES_path = joinpath(get_SOBLLES_data_path(), "SOBLLES_jld2")
data_path = joinpath(SOBLLES_path, "data")
nn_split_path = joinpath(SOBLLES_path, "nn_dataset_split.json")
dataset_split = get_nn_datasets(nn_split_path)

# Load scaling data to obtain scaling (uses all timeframes for consistent normalization)
scaling_field_datasets = [FieldDataset(joinpath(data_path, sim, "instantaneous_timeseries.jld2"), backend=OnDisk())
                          for sim in dataset_split[:training]]
scaling_full_timeframes = [25:length(data["ubar"].times) for data in scaling_field_datasets]
# Load training data for plotting (full timeframes)
training_data_plot = LESDatasets(scaling_field_datasets, ZeroMeanUnitVarianceScaling, scaling_full_timeframes, coarse_size)
scaling = training_data_plot.scaling

# Load actual training data (subsampled every 10 timesteps for efficiency)
training_field_datasets = [FieldDataset(joinpath(data_path, sim, "instantaneous_timeseries.jld2"), backend=OnDisk())
                           for sim in dataset_split[:training]]
timeframes = [25:10:length(data["ubar"].times) for data in training_field_datasets]
training_data = LESDatasets(training_field_datasets, scaling, timeframes, coarse_size)

# Load validation data (same subsampling for training; full resolution for final plots)
validation_field_datasets = [FieldDataset(joinpath(data_path, sim, "instantaneous_timeseries.jld2"), backend=OnDisk())
                             for sim in dataset_split[:validation]]
timeframes_validation = [25:10:length(data["ubar"].times) for data in validation_field_datasets]
full_timeframes_validation = [25:length(data["ubar"].times) for data in validation_field_datasets]
validation_data = LESDatasets(validation_field_datasets, scaling, timeframes_validation, coarse_size)
# Load validation data for plotting (full timeframes)
validation_data_plot = LESDatasets(validation_field_datasets, scaling, full_timeframes_validation, coarse_size)

# Ground truth profiles for loss computation (only T, S, ρ and gradients; no u, v for NDE training)
truths = [(;    T = data.profile.T.scaled,
                S = data.profile.S.scaled,
                ρ = data.profile.ρ.scaled,
             ∂T∂z = data.profile.∂T∂z.scaled,
             ∂S∂z = data.profile.∂S∂z.scaled,
             ∂ρ∂z = data.profile.∂ρ∂z.scaled) for data in training_data.data]

# Validation truths include u, v for completeness but only T, S, ρ are used in NN loss
truths_validation = [(; u = data.profile.u.scaled,
                        v = data.profile.v.scaled,
                        T = data.profile.T.scaled,
                        S = data.profile.S.scaled,
                        ρ = data.profile.ρ.scaled,
                     ∂u∂z = data.profile.∂u∂z.scaled,
                     ∂v∂z = data.profile.∂v∂z.scaled,
                     ∂T∂z = data.profile.∂T∂z.scaled,
                     ∂S∂z = data.profile.∂S∂z.scaled,
                     ∂ρ∂z = data.profile.∂ρ∂z.scaled) for data in validation_data.data]

# ODE parameters (grid operators, boundary conditions, physical constants)
params = ODEParams(training_data, scaling)
params_plot = ODEParams(training_data_plot, scaling)
params_validation = ODEParams(validation_data, scaling)
params_validation_plot = ODEParams(validation_data_plot, scaling)

rng = Random.default_rng(seed)

# Setup neural networks (separate NNs for temperature and salinity residual fluxes)
const window_split_size = NN_grid_points ÷ 2  # Half-width of sliding window for local inputs

# NN architecture: inputs = [arctan(Ri), ∂T∂z, ∂S∂z, ∂ρ∂z] at NN_grid_points locations + 1 boundary flux scalar
# Total input size = 4 * NN_grid_points + 1
NN_layers = vcat(Dense(4NN_grid_points + 1, hidden_layer_size, activation),
                 [Dense(hidden_layer_size, hidden_layer_size, activation) for _ in 1:N_hidden_layer-1]...,
                 Dense(hidden_layer_size, 1))  # Output: single residual flux value

wT_NN = Chain(NN_layers...)  # NN for temperature residual flux
wS_NN = Chain(NN_layers...)  # NN for salinity residual flux

ps_wT, st_wT = Lux.setup(rng, wT_NN)
ps_wS, st_wS = Lux.setup(rng, wS_NN)

ps_wT = ps_wT |> ComponentArray .|> Float64
ps_wS = ps_wS |> ComponentArray .|> Float64

# Initialize with Glorot uniform distribution
ps_wT .= glorot_uniform(rng, Float64, length(ps_wT))
ps_wS .= glorot_uniform(rng, Float64, length(ps_wS))

ps = ComponentArray(; wT=ps_wT, wS=ps_wS)  # Combined parameters
NNs = (wT=wT_NN, wS=wS_NN)  # NN models
sts = (wT=st_wT, wS=st_wS)  # NN states

# Initial conditions for NDE solver (first timestep of each simulation)
x₀s = [(; u = data.profile.u.scaled[:, 1],
          v = data.profile.v.scaled[:, 1],
          T = data.profile.T.scaled[:, 1],
          S = data.profile.S.scaled[:, 1]) for data in training_data.data]

x₀s_plot = [(; u = data.profile.u.scaled[:, 1],
               v = data.profile.v.scaled[:, 1],
               T = data.profile.T.scaled[:, 1],
               S = data.profile.S.scaled[:, 1]) for data in training_data_plot.data]

x₀s_validation = [(; u = data.profile.u.scaled[:, 1],
                     v = data.profile.v.scaled[:, 1],
                     T = data.profile.T.scaled[:, 1],
                     S = data.profile.S.scaled[:, 1]) for data in validation_data.data]

x₀s_validation_plot = [(; u = data.profile.u.scaled[:, 1],
                          v = data.profile.v.scaled[:, 1],
                          T = data.profile.T.scaled[:, 1],
                          S = data.profile.S.scaled[:, 1]) for data in validation_data_plot.data]

scaling_params = write_scaling_params(scaling)

# Training mode for NN
mode = NNMode()

"""
    train_NDE_multipleics(...)

Training loop for Neural Differential Equation with Enzyme autodiff.

Trains NN residual flux corrections on top of base closure using multiple initial conditions.
Uses Enzyme for automatic differentiation through the entire forward solve.
"""
function train_NDE_multipleics(ps, params, ps_baseclosure, sts, NNs, truths, x₀s, truths_validation, x₀s_validation,
                               params_validation, validation_data, training_data, timeframes, scaling_params,
                               grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size;
                               sim_index=[1], epoch=1, maxiter=2, rule=Optimisers.Adam())
    # Setup optimizer and store state at minimum loss
    opt_state = Optimisers.setup(rule, ps)
    opt_statemin = deepcopy(opt_state)
    opt_statemin_validation = deepcopy(opt_state)

    l_min = Inf
    l_min_validation = Inf

    iter_min = 1
    iter_min_validation = 1

    ps_min = deepcopy(ps)
    ps_min_validation = deepcopy(ps)

    dps = deepcopy(ps) .= 0
    wall_clock = [time_ns()]

    losses = zeros(maxiter)
    losses_validation = zeros(maxiter)

    mean_loss = 0
    mean_loss_validation = 0

    Nts_validation = [length(data.times) for data in validation_data.data]

    # Compute individual loss components for adaptive weighting
    ind_losses = [individual_loss(mode, ps, truth, param, x₀, ps_baseclosure, sts, NNs, length(timeframes),
                                  grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size)
                  for (truth, x₀, param) in zip(truths[sim_index], x₀s[sim_index], params[sim_index])]

    ind_losses_validation = [individual_loss(mode, ps, truth, param, x₀, ps_baseclosure, sts, NNs, Nt,
                                            grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size)
                            for (truth, x₀, param, Nt) in zip(truths_validation, x₀s_validation, params_validation, Nts_validation)]

    # Adaptive loss weighting based on T/S density contributions
    loss_prefactors = compute_loss_prefactor_density_contribution.(Ref(mode), ind_losses, 
                                                                   compute_density_contribution.(training_data.data))
    loss_prefactors_validation = compute_loss_prefactor_density_contribution.(Ref(mode), ind_losses_validation, 
                                                                              compute_density_contribution.(validation_data.data))

    jldopen(joinpath(WEIGHTS_DIR, "model_weights_round$(epoch)_end$(timeframes[end]).jld2"), "w") do file
        file["0"] = ps
        file["scaling"] = scaling_params
        file["model"] = NNs
        file["sts"] = sts
    end

    jldopen(joinpath(FILE_DIR, "model_loss_training_round$(epoch)_end$(timeframes[end]).jld2"), "w")
    jldopen(joinpath(FILE_DIR, "model_loss_validation_round$(epoch)_end$(timeframes[end]).jld2"), "w")

    for iter in 1:maxiter
        # Compute training loss and gradients via Enzyme autodiff
        # DuplicatedNoNeed: differentiate w.r.t. first arg, store gradient in second arg
        # Const: do not differentiate w.r.t. this argument
        _, l = autodiff(Enzyme.ReverseWithPrimal,
                        loss_multipleics,
                        Active,  # Return both value and primal
                        Const(mode),
                        DuplicatedNoNeed(ps, dps),  # Gradient goes into dps
                        DuplicatedNoNeed(truths[sim_index], deepcopy(truths[sim_index])),
                        DuplicatedNoNeed(params[sim_index], deepcopy(params[sim_index])),
                        DuplicatedNoNeed(x₀s[sim_index], deepcopy(x₀s[sim_index])),
                        DuplicatedNoNeed(ps_baseclosure, deepcopy(ps_baseclosure)),
                        Const(sts),
                        Const(NNs),
                        DuplicatedNoNeed(loss_prefactors[sim_index], deepcopy(loss_prefactors[sim_index])),
                        Const(length(timeframes)),
                        Const(grid_point_below_kappa),
                        Const(grid_point_above_kappa),
                        Const(Riᶜ),
                        Const(window_split_size))

        # Compute validation loss (no gradient needed)
        l_validation = loss_multipleics(mode, ps, truths_validation, params_validation, x₀s_validation, ps_baseclosure, 
                                        sts, NNs, loss_prefactors_validation, Nts_validation,
                                        grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size)

        # Update parameters using optimizer
        opt_state, ps = Optimisers.update!(opt_state, ps, dps)

        losses[iter] = l
        losses_validation[iter] = l_validation

        if iter == 1
            l_min = l
            l_min_validation = l_validation
        end

        @printf("%s, Δt %s, round %d, iter %d/%d, (l_t, l_v) (%6.10e, %6.10e), (lmin_t, l_min_v) (%6.5e, %6.5e)\n",
                Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), epoch, iter, 
                maxiter, l, l_validation, l_min, l_min_validation)

        dps .= 0
        mean_loss = l
        mean_loss_validation = l_validation

        # Track best parameters based on training and validation loss
        if mean_loss < l_min
            l_min = mean_loss
            iter_min = iter
            opt_statemin = deepcopy(opt_state)
            ps_min .= ps
        end

        if mean_loss_validation < l_min_validation
            l_min_validation = mean_loss_validation
            iter_min_validation = iter
            opt_statemin_validation = deepcopy(opt_state)
            ps_min_validation .= ps
        end

        # Save model weights and losses at each iteration
        jldopen(joinpath(WEIGHTS_DIR, "model_weights_round$(epoch)_end$(timeframes[end]).jld2"), "a") do file
            file["$(iter)"] = ps
        end

        jldopen(joinpath(FILE_DIR, "model_loss_training_round$(epoch)_end$(timeframes[end]).jld2"), "a+") do file
            file["$(iter)"] = l
        end

        jldopen(joinpath(FILE_DIR, "model_loss_validation_round$(epoch)_end$(timeframes[end]).jld2"), "a+") do file
            file["$(iter)"] = l_validation
        end

        wall_clock = [time_ns()]
    end

    return (ps_min, ps_min_validation, (; total=losses, total_validation=losses_validation), 
            opt_statemin, opt_statemin_validation, iter_min, iter_min_validation)
end

# Curriculum learning: gradually increase training horizon and decrease learning rate
optimizers = [Optimisers.Adam(3e-4), Optimisers.Adam(3e-5), Optimisers.Adam(1e-5)]
maxiters = [2000, 2000, 2000]  # Iterations per stage
end_epochs = cumsum(maxiters)  # Cumulative epoch numbers
training_timeframes = [timeframes[1][1:10], timeframes[1][1:15], timeframes[1][1:27]]  # Progressive horizon extension
sim_indices = 1:length(training_field_datasets)  # Train on all simulations

sols = nothing
for (i, (epoch, optimizer, maxiter, training_timeframe)) in enumerate(zip(end_epochs, optimizers, maxiters, training_timeframes))
    global ps = ps
    global sols = sols

    # Train NDE with current learning rate and time horizon
    ps, ps_validation, losses, opt_state, opt_state_validation, iter_min, iter_min_validation = train_NDE_multipleics(
        ps, params, ps_baseclosure, sts, NNs,
        truths, x₀s, truths_validation, x₀s_validation, params_validation, validation_data,
        training_data, training_timeframe, scaling_params,
        grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size;
        sim_index=sim_indices, epoch=i, maxiter=maxiter, rule=optimizer)

    # Save training results (best parameters, optimizer state, losses)
    jldsave(joinpath(FILE_DIR, "training_results_epoch$(epoch)_end$(training_timeframe[end]).jld2");
            u_train = ps, state_train = opt_state, iter_min = iter_min,
            u_validation = ps_validation, state_validation = opt_state_validation, iter_min_validation = iter_min_validation,
            losses = losses, scaling=scaling_params, model=NNs, sts=sts)

    plot_timeframe = training_timeframe[1]:training_timeframe[end]
    complete_timeframe = 25:length(validation_field_datasets[1]["ubar"].times)

    # Generate animations for training data
    for (i, data) in enumerate(training_data_plot.data)
        sol = diagnose_fields(mode, ps, params_plot[i], x₀s_plot[i], ps_baseclosure, sts, NNs, data, length(plot_timeframe),
                             grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, 2)
        animate_data(mode, data, sol.sols_dimensional, sol.fluxes, sol.diffusivities, 
                     sol.sols_dimensional_noNN, sol.fluxes_noNN, sol.diffusivities_noNN, 
                     i, FILE_DIR, length(plot_timeframe); suffix="epoch$(epoch)_end$(training_timeframe[end])")
    end

    # Generate animations for validation data (full time horizon for generalization assessment)
    for (i, data) in enumerate(validation_data_plot.data)
        sol = diagnose_fields(mode, ps, params_validation_plot[i], x₀s_validation_plot[i], ps_baseclosure, 
                              sts, NNs, data, length(complete_timeframe),
                              grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, 2)
        animate_data(mode, data, sol.sols_dimensional, sol.fluxes, sol.diffusivities, 
                     sol.sols_dimensional_noNN, sol.fluxes_noNN, sol.diffusivities_noNN, 
                     i, FILE_DIR, length(plot_timeframe); suffix="epoch$(epoch)_end$(training_timeframe[end])", prefix="validation")
    end

    # Plot training and validation loss curves
    plot_loss(mode, losses, FILE_DIR; suffix="epoch$(epoch)_end$(training_timeframe[end])")
end
