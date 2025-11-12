using LinearAlgebra
using SeawaterPolynomials.TEOS10
using SeawaterPolynomials
using NORiOceanParameterization.Utils: calculate_Ri
using NORiOceanParameterization.ODEOperations: predict_diffusivities, predict_diffusivities!, predict_boundary_flux!, predict_diffusive_boundary_flux_dimensional
using Statistics

"""
    predict_residual_flux(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, Ri, T_top, S_top, p, params, sts, NNs, grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size)

Predict residual neural network fluxes for T and S based on local gradients and Richardson number.

Uses a sliding window approach to construct NN inputs from neighboring grid points.
Only applies NN corrections in the boundary layer zone determined by Richardson number.

# Arguments
- `∂T∂z_hat`, `∂S∂z_hat`, `∂ρ∂z_hat`: Scaled vertical gradients
- `Ri`: Richardson number profile
- `T_top`, `S_top`: Top boundary values for thermal expansion/haline contraction calculation
- `p`: Neural network parameters (wT and wS networks)
- `params`: ODE parameters containing scalings and grid info
- `sts`: Neural network states
- `NNs`: Neural network models (wT and wS)
- `grid_point_below_kappa`: Grid points below convective kappa to turn off NN
- `grid_point_above_kappa`: Grid points above convective kappa to turn off NN
- `Riᶜ`: Critical Richardson number threshold
- `window_split_size`: Half-width of sliding window for NN inputs

# Returns
Tuple of residual fluxes `(wT, wS)` with NN corrections applied only in boundary layer zone
"""
function predict_residual_flux(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, Ri, T_top, S_top, p, params, sts, NNs,
                                grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size)
    eos = TEOS10EquationOfState()
    α = SeawaterPolynomials.thermal_expansion(T_top, S_top, 0, eos)
    β = SeawaterPolynomials.haline_contraction(T_top, S_top, 0, eos)
    wT = params.wT.unscaled.top
    wS = params.wS.unscaled.top
    coarse_size = params.coarse_size
    top_index = coarse_size + 1

    arctan_Ri = atan.(Ri)

    background_κ_index = findfirst(Ri[2:end] .< Riᶜ)
    nonbackground_κ_index = background_κ_index + 1
    last_index = ifelse(nonbackground_κ_index == top_index, top_index-1, min(background_κ_index + grid_point_above_kappa, coarse_size))
    first_index = ifelse(nonbackground_κ_index == top_index, top_index, max(background_κ_index - grid_point_below_kappa + 1, 2))

    wb_top_scaled = params.scaling.wb(params.g * (α * wT - β * wS))
    common_variables = wb_top_scaled

    wT_residual = zeros(coarse_size+1)
    wS_residual = zeros(coarse_size+1)

    for i in first_index:last_index
        i_min = i - window_split_size
        i_max = i + window_split_size

        if i_max >= top_index
            n_repeat = i_max - coarse_size
            arctan_Ri_i = vcat(arctan_Ri[i_min:coarse_size], repeat(arctan_Ri[coarse_size:coarse_size], n_repeat))
            ∂T∂z_hat_i = vcat(∂T∂z_hat[i_min:coarse_size], repeat(∂T∂z_hat[coarse_size:coarse_size], n_repeat))
            ∂S∂z_hat_i = vcat(∂S∂z_hat[i_min:coarse_size], repeat(∂S∂z_hat[coarse_size:coarse_size], n_repeat))
            ∂ρ∂z_hat_i = vcat(∂ρ∂z_hat[i_min:coarse_size], repeat(∂ρ∂z_hat[coarse_size:coarse_size], n_repeat))
            x = vcat(arctan_Ri_i, ∂T∂z_hat_i, ∂S∂z_hat_i, ∂ρ∂z_hat_i, common_variables)
        elseif i_min <= 1
            n_repeat = 2 - i_min
            arctan_Ri_i = vcat(repeat(arctan_Ri[2:2], n_repeat), arctan_Ri[2:i_max])
            ∂T∂z_hat_i = vcat(repeat(∂T∂z_hat[2:2], n_repeat), ∂T∂z_hat[2:i_max])
            ∂S∂z_hat_i = vcat(repeat(∂S∂z_hat[2:2], n_repeat), ∂S∂z_hat[2:i_max])
            ∂ρ∂z_hat_i = vcat(repeat(∂ρ∂z_hat[2:2], n_repeat), ∂ρ∂z_hat[2:i_max])
            x = vcat(arctan_Ri_i, ∂T∂z_hat_i, ∂S∂z_hat_i, ∂ρ∂z_hat_i, common_variables)
        else
            x = vcat(arctan_Ri[i_min:i_max], ∂T∂z_hat[i_min:i_max], ∂S∂z_hat[i_min:i_max], ∂ρ∂z_hat[i_min:i_max], common_variables)
        end

        wT_residual[i] = first(NNs.wT(x, p.wT, sts.wT))[1]
        wS_residual[i] = first(NNs.wS(x, p.wS, sts.wS))[1]
    end

    return wT_residual, wS_residual
end

"""
    predict_residual_flux_dimensional(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, Ri, T_top, S_top, p, params, sts, NNs, grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size)

Convert residual NN fluxes from scaled to dimensional form, removing mean offset.

# Returns
Dimensional residual fluxes `(wT, wS)` with zero bottom boundary condition
"""
function predict_residual_flux_dimensional(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, Ri, T_top, S_top, p, params, sts, NNs,
                                           grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size)
    wT_hat, wS_hat = predict_residual_flux(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, Ri, T_top, S_top, p, params, sts, NNs,
                                           grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size)

    wT = inv(params.scaling.wT).(wT_hat)
    wS = inv(params.scaling.wS).(wS_hat)

    wT = wT .- wT[1]
    wS = wS .- wS[1]

    return wT, wS
end

"""
    solve_NDE(::NNMode, ps, params, x₀, ps_baseclosure, sts, NNs, Nt, grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, timestep_multiple=10)

Solve neural differential equation forward in time using base closure + neural network correction.

Combines a base Richardson number closure with NN-predicted residual fluxes for T and S.
The NN corrections are only applied in the boundary layer zone determined by the Richardson number profile.

# Arguments
- `::NNMode`: Training mode indicator
- `ps`: Neural network parameters (wT and wS)
- `params`: ODE parameters containing grid, operators, and scaling
- `x₀`: Initial conditions (u, v, T, S)
- `ps_baseclosure`: Base closure parameters (ν_conv, ν_shear, Riᶜ, ΔRi, Pr_conv, Pr_shear)
- `sts`: Neural network states
- `NNs`: Neural network models (wT and wS)
- `Nt`: Number of output timesteps
- `grid_point_below_kappa`: Grid points below convective kappa to turn off NN
- `grid_point_above_kappa`: Grid points above convective kappa to turn off NN
- `Riᶜ`: Critical Richardson number threshold
- `window_split_size`: Half-width of sliding window for NN inputs
- `timestep_multiple=10`: Temporal refinement factor for solver accuracy stability

# Returns
NamedTuple with solution profiles: `(u, v, T, S, ρ)` at original timesteps
"""
function solve_NDE(ps, params, x₀, ps_baseclosure, sts, NNs, Nt,
                   grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, timestep_multiple=10)
    eos = TEOS10EquationOfState()
    coarse_size = params.coarse_size
    timestep = params.scaled_time[2] - params.scaled_time[1]
    Δt = timestep / timestep_multiple
    Nt_solve = (Nt - 1) * timestep_multiple + 1
    Dᶜ_hat = params.Dᶜ_hat
    Dᶠ_hat = params.Dᶠ_hat
    Dᶠ = params.Dᶠ

    scaling = params.scaling
    τ, H = params.τ, params.H
    f = params.f

    u_hat = deepcopy(x₀.u)
    v_hat = deepcopy(x₀.v)
    T_hat = deepcopy(x₀.T)
    S_hat = deepcopy(x₀.S)
    ρ_hat = zeros(coarse_size)

    ∂u∂z_hat = zeros(coarse_size+1)
    ∂v∂z_hat = zeros(coarse_size+1)
    ∂T∂z_hat = zeros(coarse_size+1)
    ∂S∂z_hat = zeros(coarse_size+1)
    ∂ρ∂z_hat = zeros(coarse_size+1)

    u = zeros(coarse_size)
    v = zeros(coarse_size)
    T = zeros(coarse_size)
    S = zeros(coarse_size)
    ρ = zeros(coarse_size)

    u_RHS = zeros(coarse_size)
    v_RHS = zeros(coarse_size)
    T_RHS = zeros(coarse_size)
    S_RHS = zeros(coarse_size)

    wT_residual = zeros(coarse_size+1)
    wS_residual = zeros(coarse_size+1)

    uw_boundary = zeros(coarse_size+1)
    vw_boundary = zeros(coarse_size+1)
    wT_boundary = zeros(coarse_size+1)
    wS_boundary = zeros(coarse_size+1)

    νs = zeros(coarse_size+1)
    κs = zeros(coarse_size+1)

    Ris = zeros(coarse_size+1)

    sol_u = zeros(coarse_size, Nt_solve)
    sol_v = zeros(coarse_size, Nt_solve)
    sol_T = zeros(coarse_size, Nt_solve)
    sol_S = zeros(coarse_size, Nt_solve)
    sol_ρ = zeros(coarse_size, Nt_solve)

    sol_u[:, 1] .= u_hat
    sol_v[:, 1] .= v_hat
    sol_T[:, 1] .= T_hat
    sol_S[:, 1] .= S_hat

    LHS_uv = zeros(coarse_size, coarse_size)
    LHS_TS = zeros(coarse_size, coarse_size)

    for i in 2:Nt_solve
        u .= inv(scaling.u).(u_hat)
        v .= inv(scaling.v).(v_hat)
        T .= inv(scaling.T).(T_hat)
        S .= inv(scaling.S).(S_hat)

        ρ .= TEOS10.ρ.(T, S, 0, Ref(eos))
        ρ_hat .= scaling.ρ.(ρ)
        sol_ρ[:, i-1] .= ρ_hat

        ∂u∂z_hat .= scaling.∂u∂z.(Dᶠ * u)
        ∂v∂z_hat .= scaling.∂v∂z.(Dᶠ * v)
        ∂T∂z_hat .= scaling.∂T∂z.(Dᶠ * T)
        ∂S∂z_hat .= scaling.∂S∂z.(Dᶠ * S)
        ∂ρ∂z_hat .= scaling.∂ρ∂z.(Dᶠ * ρ)

        Ris .= calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf))
        predict_diffusivities!(νs, κs, Ris, ps_baseclosure)

        LHS_uv .= Tridiagonal(Dᶜ_hat * (-νs .* Dᶠ_hat))
        LHS_TS .= Tridiagonal(Dᶜ_hat * (-κs .* Dᶠ_hat))

        LHS_uv .*= -τ / H^2
        LHS_TS .*= -τ / H^2

        T_top = T[end]
        S_top = S[end]

        wT_residual, wS_residual = predict_residual_flux(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, Ris, T_top, S_top, ps, params, sts, NNs,
                                                          grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size)
        predict_boundary_flux!(uw_boundary, vw_boundary, wT_boundary, wS_boundary, params)

        u_RHS .= - τ / H * scaling.uw.σ / scaling.u.σ .* (Dᶜ_hat * (uw_boundary)) .+ f * τ ./ scaling.u.σ .* v
        v_RHS .= - τ / H * scaling.vw.σ / scaling.v.σ .* (Dᶜ_hat * (vw_boundary)) .- f * τ ./ scaling.v.σ .* u
        T_RHS .= - τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary .+ wT_residual))
        S_RHS .= - τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary .+ wS_residual))

        u_hat .= (I - Δt .* LHS_uv) \ (u_hat .+ Δt .* u_RHS)
        v_hat .= (I - Δt .* LHS_uv) \ (v_hat .+ Δt .* v_RHS)
        T_hat .= (I - Δt .* LHS_TS) \ (T_hat .+ Δt .* T_RHS)
        S_hat .= (I - Δt .* LHS_TS) \ (S_hat .+ Δt .* S_RHS)

        sol_u[:, i] .= u_hat
        sol_v[:, i] .= v_hat
        sol_T[:, i] .= T_hat
        sol_S[:, i] .= S_hat
    end

    sol_ρ[:, end] .= scaling.ρ.(TEOS10.ρ.(inv(scaling.T).(T_hat), inv(scaling.S).(S_hat), 0, Ref(eos)))

    return (; u=sol_u[:, 1:timestep_multiple:end], v=sol_v[:, 1:timestep_multiple:end], T=sol_T[:, 1:timestep_multiple:end], S=sol_S[:, 1:timestep_multiple:end], ρ=sol_ρ[:, 1:timestep_multiple:end])
end

"""
    individual_loss(mode::NNMode, ps, truth, params, x₀, ps_baseclosure, sts, NNs, Nt, grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, tstart=1, timestep_multiple=10)

Compute individual loss components for profiles and their gradients using NN-corrected fluxes.

# Arguments
- `mode::NNMode`: Training mode
- `ps`: NN parameters
- `truth`: Ground truth data (T, S, ρ, ∂T∂z, ∂S∂z, ∂ρ∂z)
- `params`: ODE parameters
- `x₀`: Initial conditions
- `ps_baseclosure`: Base closure parameters
- `sts`: NN states
- `NNs`: NN models
- `Nt`: Number of timesteps
- `grid_point_below_kappa`: Grid points below convective kappa
- `grid_point_above_kappa`: Grid points above convective kappa
- `Riᶜ`: Critical Richardson number
- `window_split_size`: NN input window half-width
- `tstart=1`: Starting time index for loss calculation
- `timestep_multiple=10`: Temporal refinement factor

# Returns
NamedTuple with MSE losses for T, S, ρ and their gradients
"""
function individual_loss(ps, truth, params, x₀, ps_baseclosure, sts, NNs, Nt,
                         grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, tstart=1, timestep_multiple=10)
    Dᶠ = params.Dᶠ
    scaling = params.scaling

    # u and v losses are not included in the neural network training
    # This is because the inertial oscillations in u and v significantly deteriorates training quality
    _, _, sol_T, sol_S, sol_ρ = solve_NDE(ps, params, x₀, ps_baseclosure, sts, NNs, Nt,
                                           grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, timestep_multiple)

    T_loss = mean((sol_T .- truth.T[:, tstart:tstart+Nt-1]).^2)
    S_loss = mean((sol_S .- truth.S[:, tstart:tstart+Nt-1]).^2)
    ρ_loss = mean((sol_ρ .- truth.ρ[:, tstart:tstart+Nt-1]).^2)

    T = inv(scaling.T).(sol_T)
    S = inv(scaling.S).(sol_S)
    ρ = inv(scaling.ρ).(sol_ρ)

    ∂T∂z = scaling.∂T∂z.(Dᶠ * T)
    ∂S∂z = scaling.∂S∂z.(Dᶠ * S)
    ∂ρ∂z = scaling.∂ρ∂z.(Dᶠ * ρ)

    # Exclude last 4 grid points for tracer gradients as these are different due to imposition of boundary conditions in the LES
    # e.g. for some tracer ϕ:
    # -κ * ∂ϕ∂z = J_ϕ at boundaries for LES, which is not true in the NDE solver
    ∂T∂z_loss = mean((∂T∂z[1:end-4,:] .- truth.∂T∂z[1:end-4, tstart:tstart+Nt-1]).^2)
    ∂S∂z_loss = mean((∂S∂z[1:end-4,:] .- truth.∂S∂z[1:end-4, tstart:tstart+Nt-1]).^2)
    ∂ρ∂z_loss = mean((∂ρ∂z[1:end-4,:] .- truth.∂ρ∂z[1:end-4, tstart:tstart+Nt-1]).^2)

    return (; T=T_loss, S=S_loss, ρ=ρ_loss, ∂T∂z=∂T∂z_loss, ∂S∂z=∂S∂z_loss, ∂ρ∂z=∂ρ∂z_loss)
end

"""
    loss(mode::NNMode, ps, truth, params, x₀, ps_baseclosure, sts, NNs, Nt, grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, tstart=1, timestep_multiple=10, losses_prefactor=(; T=1, S=1, ρ=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))

Compute weighted total loss across tracer fields using NN-corrected fluxes.

# Arguments
- `mode::NNMode`: Training mode
- `ps`: NN parameters
- `truth`: Ground truth data
- `params`: ODE parameters
- `x₀`: Initial conditions
- `ps_baseclosure`: Base closure parameters
- `sts`: NN states
- `NNs`: NN models
- `Nt`: Number of timesteps
- `grid_point_below_kappa`: Grid points below convective kappa
- `grid_point_above_kappa`: Grid points above convective kappa
- `Riᶜ`: Critical Richardson number
- `window_split_size`: NN input window half-width
- `tstart=1`: Starting time index
- `timestep_multiple=10`: Temporal refinement factor
- `losses_prefactor`: Weights for each loss component (default: all 1)

# Returns
Scalar weighted sum of all individual losses
"""
function loss(ps, truth, params, x₀, ps_baseclosure, sts, NNs, Nt,
              grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, tstart=1, timestep_multiple=10,
              losses_prefactor=(; T=1, S=1, ρ=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))
    losses = individual_loss(ps, truth, params, x₀, ps_baseclosure, sts, NNs, Nt,
                             grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, tstart, timestep_multiple)
    return sum(values(losses) .* values(losses_prefactor))
end

# loss(args...) = loss(NNMode(), args...)

"""
    compute_loss_prefactor_density_contribution(::NNMode, individual_loss, contribution, S_scaling=1.0)

Compute adaptive loss weights based on T/S density contributions for NN training.

Balances temperature and salinity losses according to their physical density contributions.
Only considers tracers (no momentum fields for NDE training).

# Arguments
- `::NNMode`: Training mode
- `individual_loss`: Individual loss components
- `contribution`: Density contributions from T and S (from `compute_density_contribution`)
- `S_scaling=1.0`: Additional scaling factor for salinity losses

# Returns
NamedTuple of loss prefactors for T, S, ρ and their gradients
"""
function compute_loss_prefactor_density_contribution(individual_loss, contribution, S_scaling=1.0)
    T_loss, S_loss, ρ_loss, ∂T∂z_loss, ∂S∂z_loss, ∂ρ∂z_loss = values(individual_loss)

    T_contribution = max(contribution.T, 1e-2)
    S_contribution = max(contribution.S, 1e-2)

    # Density contribution weighting: profile vs density losses
    # These weights balance density loss vs tracer losses
    density_profile_weighting = 0.1
    tracer_profile_weighting = 0.9
    density_weighting_ratio = density_profile_weighting / tracer_profile_weighting

    total_contribution = T_contribution + S_contribution
    T_prefactor = total_contribution / T_contribution
    S_prefactor = total_contribution / S_contribution

    TS_loss = T_prefactor * T_loss + S_prefactor * S_loss

    ρ_prefactor = TS_loss / ρ_loss * density_weighting_ratio
    ∂T∂z_prefactor = T_prefactor
    ∂S∂z_prefactor = S_prefactor

    ∂TS∂z_loss = ∂T∂z_loss + ∂S∂z_loss
    ∂ρ∂z_prefactor = ∂TS∂z_loss / ∂ρ∂z_loss * density_weighting_ratio

    profile_loss = T_prefactor * T_loss + S_prefactor * S_loss + ρ_prefactor * ρ_loss
    gradient_loss = ∂T∂z_prefactor * ∂T∂z_loss + ∂S∂z_prefactor * ∂S∂z_loss + ∂ρ∂z_prefactor * ∂ρ∂z_loss

    gradient_prefactor = profile_loss / gradient_loss

    ∂ρ∂z_prefactor *= gradient_prefactor
    ∂T∂z_prefactor *= gradient_prefactor
    ∂S∂z_prefactor *= gradient_prefactor

    S_prefactor *= S_scaling
    ∂S∂z_prefactor *= S_scaling

    return (T=T_prefactor, S=S_prefactor, ρ=ρ_prefactor, ∂T∂z=∂T∂z_prefactor, ∂S∂z=∂S∂z_prefactor, ∂ρ∂z=∂ρ∂z_prefactor)
end

"""
    loss_multipleics(mode::NNMode, ps, truths, params, x₀s, ps_baseclosure, sts, NNs, losses_prefactors, Nt::Number, grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, tstart=1, timestep_multiple=10)

Compute mean loss across multiple initial conditions using NN-corrected fluxes (single Nt).

# Arguments
- `mode::NNMode`: Training mode
- `ps`: NN parameters (shared across all ICs)
- `truths`: Vector of ground truth datasets
- `params`: Vector of ODE parameters
- `x₀s`: Vector of initial conditions
- `ps_baseclosure`: Base closure parameters
- `sts`: NN states
- `NNs`: NN models
- `losses_prefactors`: Vector of loss weight tuples
- `Nt::Number`: Number of timesteps (same for all ICs)
- `grid_point_below_kappa`: Grid points below convective kappa
- `grid_point_above_kappa`: Grid points above convective kappa
- `Riᶜ`: Critical Richardson number
- `window_split_size`: NN input window half-width
- `tstart=1`: Starting time index
- `timestep_multiple=10`: Temporal refinement factor

# Returns
Mean loss across all initial conditions
"""
function loss_multipleics(ps, truths, params, x₀s, ps_baseclosure, sts, NNs, losses_prefactors, Nt::Number,
                          grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, tstart=1, timestep_multiple=10)
    losses = [loss(ps, truth, param, x₀, ps_baseclosure, sts, NNs, Nt,
                   grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, tstart, timestep_multiple, loss_prefactor)
              for (truth, x₀, param, loss_prefactor) in zip(truths, x₀s, params, losses_prefactors)]
    return mean(losses)
end

"""
    loss_multipleics(mode::NNMode, ps, truths, params, x₀s, ps_baseclosure, sts, NNs, losses_prefactors, Nts::Vector, grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, tstart=1, timestep_multiple=10)

Compute mean loss across multiple initial conditions using NN-corrected fluxes (different Nt per IC).

Allows each initial condition to have a different number of timesteps for loss calculation.

# Arguments
Same as single-Nt version but with:
- `Nts::Vector`: Vector of timesteps (one per IC)

# Returns
Mean loss across all initial conditions
"""
function loss_multipleics(ps, truths, params, x₀s, ps_baseclosure, sts, NNs, losses_prefactors, Nts::Vector,
                          grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, tstart=1, timestep_multiple=10)
    losses = [loss(ps, truth, param, x₀, ps_baseclosure, sts, NNs, Nt,
                   grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, tstart, timestep_multiple, loss_prefactor)
              for (truth, x₀, param, loss_prefactor, Nt) in zip(truths, x₀s, params, losses_prefactors, Nts)]
    return mean(losses)
end

# loss_multipleics(args...) = loss_multipleics(NNMode(), args...)

"""
    diagnose_fields(mode::NNMode, ps, params, x₀, ps_baseclosure, sts, NNs, train_data_plot, Nt, grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, timestep_multiple=2)

Compute diagnostic fields for analysis and visualization using NN-corrected fluxes.

Solves NDE with both NN corrections and base closure only, computes Richardson numbers,
diffusivities, and total fluxes for comparison.

# Arguments
- `mode::NNMode`: Training mode
- `ps`: NN parameters
- `params`: ODE parameters
- `x₀`: Initial conditions
- `ps_baseclosure`: Base closure parameters
- `sts`: NN states
- `NNs`: NN models
- `train_data_plot`: Training data for Richardson number comparison
- `Nt`: Number of timesteps
- `grid_point_below_kappa`: Grid points below convective kappa
- `grid_point_above_kappa`: Grid points above convective kappa
- `Riᶜ`: Critical Richardson number
- `window_split_size`: NN input window half-width
- `timestep_multiple=2`: Temporal refinement factor

# Returns
NamedTuple with:
- `sols_dimensional`: Dimensional solution profiles with NN (u, v, T, S, ρ)
- `sols_dimensional_noNN`: Dimensional solution profiles base closure only
- `fluxes`: Total fluxes with NN (uw, vw, wT, wS) including residual and diffusive_boundary components
- `fluxes_noNN`: Total fluxes base closure only
- `diffusivities`: Diagnosed ν, κ, Ri with NN, and truth Ri
- `diffusivities_noNN`: Diagnosed ν, κ, Ri base closure only
"""
function diagnose_fields(ps, params, x₀, ps_baseclosure, sts, NNs, train_data_plot, Nt,
                         grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, timestep_multiple=2)
    sols = solve_NDE(ps, params, x₀, ps_baseclosure, sts, NNs, Nt,
                     grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, timestep_multiple)

    ps_noNN = deepcopy(ps) .= 0
    sols_noNN = solve_NDE(ps_noNN, params, x₀, ps_baseclosure, sts, NNs, Nt,
                          grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size, timestep_multiple)

    coarse_size = params.coarse_size
    Dᶠ = params.Dᶠ
    scaling = params.scaling

    us = inv(scaling.u).(sols.u)
    vs = inv(scaling.v).(sols.v)
    Ts = inv(scaling.T).(sols.T)
    Ss = inv(scaling.S).(sols.S)
    ρs = inv(scaling.ρ).(sols.ρ)

    T_tops = Ts[end, :]
    S_tops = Ss[end, :]

    us_noNN = inv(scaling.u).(sols_noNN.u)
    vs_noNN = inv(scaling.v).(sols_noNN.v)
    Ts_noNN = inv(scaling.T).(sols_noNN.T)
    Ss_noNN = inv(scaling.S).(sols_noNN.S)
    ρs_noNN = inv(scaling.ρ).(sols_noNN.ρ)

    ∂T∂z_hats = hcat([params.scaling.∂T∂z.(params.Dᶠ * T) for T in eachcol(Ts)]...)
    ∂S∂z_hats = hcat([params.scaling.∂S∂z.(params.Dᶠ * S) for S in eachcol(Ss)]...)
    ∂ρ∂z_hats = hcat([params.scaling.∂ρ∂z.(params.Dᶠ * ρ) for ρ in eachcol(ρs)]...)

    eos = TEOS10EquationOfState()
    Ris_truth = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf))
                      for (u, v, ρ) in zip(eachcol(train_data_plot.profile.u.unscaled), eachcol(train_data_plot.profile.v.unscaled), eachcol(train_data_plot.profile.ρ.unscaled))]...)
    Ris = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf))
                      for (u, v, ρ) in zip(eachcol(us), eachcol(vs), eachcol(ρs))]...)
    Ris_noNN = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf))
                      for (u, v, ρ) in zip(eachcol(us_noNN), eachcol(vs_noNN), eachcol(ρs_noNN))]...)

    νs, κs = predict_diffusivities(Ris, ps_baseclosure)
    νs_noNN, κs_noNN = predict_diffusivities(Ris_noNN, ps_baseclosure)

    uw_residuals = zeros(coarse_size+1, size(Ts, 2))
    vw_residuals = zeros(coarse_size+1, size(Ts, 2))
    wT_residuals = zeros(coarse_size+1, size(Ts, 2))
    wS_residuals = zeros(coarse_size+1, size(Ts, 2))

    uw_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    vw_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    wT_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    wS_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))

    uw_diffusive_boundarys_noNN = zeros(coarse_size+1, size(Ts, 2))
    vw_diffusive_boundarys_noNN = zeros(coarse_size+1, size(Ts, 2))
    wT_diffusive_boundarys_noNN = zeros(coarse_size+1, size(Ts, 2))
    wS_diffusive_boundarys_noNN = zeros(coarse_size+1, size(Ts, 2))

    for i in 1:size(wT_residuals, 2)
        wT_residuals[:, i], wS_residuals[:, i] = predict_residual_flux_dimensional(∂T∂z_hats[:, i], ∂S∂z_hats[:, i], ∂ρ∂z_hats[:, i], Ris[:, i], T_tops[i], S_tops[i], ps, params, sts, NNs,
                                                                                     grid_point_below_kappa, grid_point_above_kappa, Riᶜ, window_split_size)
        uw_diffusive_boundarys[:, i], vw_diffusive_boundarys[:, i], wT_diffusive_boundarys[:, i], wS_diffusive_boundarys[:, i] = predict_diffusive_boundary_flux_dimensional(Ris[:, i], sols.u[:, i], sols.v[:, i], sols.T[:, i], sols.S[:, i], ps_baseclosure, params)

        uw_diffusive_boundarys_noNN[:, i], vw_diffusive_boundarys_noNN[:, i], wT_diffusive_boundarys_noNN[:, i], wS_diffusive_boundarys_noNN[:, i] = predict_diffusive_boundary_flux_dimensional(Ris_truth[:, i], sols_noNN.u[:, i], sols_noNN.v[:, i], sols_noNN.T[:, i], sols_noNN.S[:, i], ps_baseclosure, params)
    end

    uw_totals = uw_residuals .+ uw_diffusive_boundarys
    vw_totals = vw_residuals .+ vw_diffusive_boundarys
    wT_totals = wT_residuals .+ wT_diffusive_boundarys
    wS_totals = wS_residuals .+ wS_diffusive_boundarys

    fluxes = (; uw = (; diffusive_boundary=uw_diffusive_boundarys, residual=uw_residuals, total=uw_totals),
                vw = (; diffusive_boundary=vw_diffusive_boundarys, residual=vw_residuals, total=vw_totals),
                wT = (; diffusive_boundary=wT_diffusive_boundarys, residual=wT_residuals, total=wT_totals),
                wS = (; diffusive_boundary=wS_diffusive_boundarys, residual=wS_residuals, total=wS_totals))

    fluxes_noNN = (; uw = (; total=uw_diffusive_boundarys_noNN),
                     vw = (; total=vw_diffusive_boundarys_noNN),
                     wT = (; total=wT_diffusive_boundarys_noNN),
                     wS = (; total=wS_diffusive_boundarys_noNN))

    diffusivities = (; ν=νs, κ=κs, Ri=Ris, Ri_truth=Ris_truth)
    diffusivities_noNN = (; ν=νs_noNN, κ=κs_noNN, Ri=Ris_noNN)

    sols_dimensional = (; u=us, v=vs, T=Ts, S=Ss, ρ=ρs)
    sols_dimensional_noNN = (; u=us_noNN, v=vs_noNN, T=Ts_noNN, S=Ss_noNN, ρ=ρs_noNN)

    return (; sols_dimensional, sols_dimensional_noNN, fluxes, fluxes_noNN, diffusivities, diffusivities_noNN)
end