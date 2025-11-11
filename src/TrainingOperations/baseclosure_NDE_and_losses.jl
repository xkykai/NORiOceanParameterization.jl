using LinearAlgebra
using SeawaterPolynomials.TEOS10
using SeawaterPolynomials
using NORiOceanParameterization.Utils: calculate_Ri
using NORiOceanParameterization.ODEOperations: predict_diffusivities, predict_diffusivities!, predict_boundary_flux!, predict_diffusive_boundary_flux_dimensional
using Statistics

"""
    solve_NDE(::BaseClosureMode, ps, params, x₀, timestep_multiple=10)

Solve neural differential equation forward in time using base closure parameterization.

# Arguments
- `::BaseClosureMode`: Training mode indicator
- `ps`: Closure parameters (ν_conv, ν_shear, Riᶜ, ΔRi, Pr_conv, Pr_shear)
- `params`: ODE parameters containing grid, operators, and scaling
- `x₀`: Initial conditions (u, v, T, S)
- `timestep_multiple=10`: Temporal refinement factor for solver accuracy stability

# Returns
NamedTuple with solution profiles: `(u, v, T, S, ρ)` at original timesteps
"""
function solve_NDE(::BaseClosureMode, ps, params, x₀, timestep_multiple=10)
    eos = TEOS10EquationOfState()
    coarse_size = params.coarse_size
    Δt = (params.scaled_time[2] - params.scaled_time[1]) / timestep_multiple
    Nt_solve = (params.N_timesteps - 1) * timestep_multiple + 1
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

    u = zeros(coarse_size)
    v = zeros(coarse_size)
    T = zeros(coarse_size)
    S = zeros(coarse_size)
    ρ = zeros(coarse_size)
    
    u_RHS = zeros(coarse_size)
    v_RHS = zeros(coarse_size)
    T_RHS = zeros(coarse_size)
    S_RHS = zeros(coarse_size)

    sol_u = zeros(coarse_size, Nt_solve)
    sol_v = zeros(coarse_size, Nt_solve)
    sol_T = zeros(coarse_size, Nt_solve)
    sol_S = zeros(coarse_size, Nt_solve)
    sol_ρ = zeros(coarse_size, Nt_solve)

    uw_boundary = zeros(coarse_size+1)
    vw_boundary = zeros(coarse_size+1)
    wT_boundary = zeros(coarse_size+1)
    wS_boundary = zeros(coarse_size+1)

    νs = zeros(coarse_size+1)
    κs = zeros(coarse_size+1)

    Ris = zeros(coarse_size+1)

    sol_u[:, 1] .= u_hat
    sol_v[:, 1] .= v_hat
    sol_T[:, 1] .= T_hat
    sol_S[:, 1] .= S_hat

    ν_LHS = Tridiagonal(zeros(coarse_size, coarse_size))
    κ_LHS = Tridiagonal(zeros(coarse_size, coarse_size))

    for i in 2:Nt_solve
        u .= inv(scaling.u).(u_hat)
        v .= inv(scaling.v).(v_hat)
        T .= inv(scaling.T).(T_hat)
        S .= inv(scaling.S).(S_hat)

        ρ .= TEOS10.ρ.(T, S, 0, Ref(eos))
        ρ_hat .= scaling.ρ.(ρ)
        sol_ρ[:, i-1] .= ρ_hat

        Ris .= calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf))
        predict_diffusivities!(νs, κs, Ris, ps)

        Dν = Dᶜ_hat * (-νs .* Dᶠ_hat)
        Dκ = Dᶜ_hat * (-κs .* Dᶠ_hat)

        predict_boundary_flux!(uw_boundary, vw_boundary, wT_boundary, wS_boundary, params)

        ν_LHS .= Tridiagonal(-τ / H^2 .* Dν)
        κ_LHS .= Tridiagonal(-τ / H^2 .* Dκ)

        u_RHS .= - τ / H * scaling.uw.σ / scaling.u.σ .* (Dᶜ_hat * (uw_boundary)) .+ f * τ ./ scaling.u.σ .* v
        v_RHS .= - τ / H * scaling.vw.σ / scaling.v.σ .* (Dᶜ_hat * (vw_boundary)) .- f * τ ./ scaling.v.σ .* u
        T_RHS .= - τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary))
        S_RHS .= - τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary))

        u_hat .= (I - Δt .* ν_LHS) \ (u_hat .+ Δt .* u_RHS)
        v_hat .= (I - Δt .* ν_LHS) \ (v_hat .+ Δt .* v_RHS)
        T_hat .= (I - Δt .* κ_LHS) \ (T_hat .+ Δt .* T_RHS)
        S_hat .= (I - Δt .* κ_LHS) \ (S_hat .+ Δt .* S_RHS)

        sol_u[:, i] .= u_hat
        sol_v[:, i] .= v_hat
        sol_T[:, i] .= T_hat
        sol_S[:, i] .= S_hat
    end

    sol_ρ[:, end] .= scaling.ρ.(TEOS10.ρ.(inv(scaling.T).(T_hat), inv(scaling.S).(S_hat), 0, Ref(eos)))

    return (; u=sol_u[:, 1:timestep_multiple:end], v=sol_v[:, 1:timestep_multiple:end], T=sol_T[:, 1:timestep_multiple:end], S=sol_S[:, 1:timestep_multiple:end], ρ=sol_ρ[:, 1:timestep_multiple:end])
end

"""
    individual_loss(mode::BaseClosureMode, ps, truth, params, x₀)

Compute individual loss components for profiles and their gradients.

# Arguments
- `mode::BaseClosureMode`: Training mode
- `ps`: Closure parameters
- `truth`: Ground truth data (u, v, T, S, ρ, ∂u∂z, ∂v∂z, ∂T∂z, ∂S∂z, ∂ρ∂z)
- `params`: ODE parameters
- `x₀`: Initial conditions

# Returns
NamedTuple with MSE losses for each field and gradient
"""
function individual_loss(mode::BaseClosureMode, ps, truth, params, x₀)
    Dᶠ = params.Dᶠ
    scaling = params.scaling
    sol_u, sol_v, sol_T, sol_S, sol_ρ = solve_NDE(mode, ps, params, x₀)

    u_loss = mean((sol_u .- truth.u).^2)
    v_loss = mean((sol_v .- truth.v).^2)
    T_loss = mean((sol_T .- truth.T).^2)
    S_loss = mean((sol_S .- truth.S).^2)
    ρ_loss = mean((sol_ρ .- truth.ρ).^2)

    u = inv(scaling.u).(sol_u)
    v = inv(scaling.v).(sol_v)
    T = inv(scaling.T).(sol_T)
    S = inv(scaling.S).(sol_S)
    ρ = inv(scaling.ρ).(sol_ρ)

    ∂u∂z = scaling.∂u∂z.(Dᶠ * u)
    ∂v∂z = scaling.∂v∂z.(Dᶠ * v)
    ∂T∂z = scaling.∂T∂z.(Dᶠ * T)
    ∂S∂z = scaling.∂S∂z.(Dᶠ * S)
    ∂ρ∂z = scaling.∂ρ∂z.(Dᶠ * ρ)

    ∂u∂z_loss = mean((∂u∂z .- truth.∂u∂z).^2)
    ∂v∂z_loss = mean((∂v∂z .- truth.∂v∂z).^2)
    ∂T∂z_loss = mean((∂T∂z .- truth.∂T∂z)[1:end-3, :].^2)
    ∂S∂z_loss = mean((∂S∂z .- truth.∂S∂z)[1:end-3, :].^2)
    ∂ρ∂z_loss = mean((∂ρ∂z .- truth.∂ρ∂z)[1:end-3, :].^2)

    return (; u=u_loss, v=v_loss, T=T_loss, S=S_loss, ρ=ρ_loss, ∂u∂z=∂u∂z_loss, ∂v∂z=∂v∂z_loss, ∂T∂z=∂T∂z_loss, ∂S∂z=∂S∂z_loss, ∂ρ∂z=∂ρ∂z_loss)
end

"""
    loss(mode::BaseClosureMode, ps, truth, params, x₀, losses_prefactor)

Compute weighted total loss across all fields.

# Arguments
- `mode::BaseClosureMode`: Training mode
- `ps`: Closure parameters
- `truth`: Ground truth data
- `params`: ODE parameters
- `x₀`: Initial conditions
- `losses_prefactor`: Weights for each loss component (default: all 1)

# Returns
Scalar weighted sum of all individual losses
"""
function loss(mode::BaseClosureMode, ps, truth, params, x₀, losses_prefactor=(; u=1, v=1, T=1, S=1, ρ=1, ∂u∂z=1, ∂v∂z=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))
    losses = individual_loss(mode, ps, truth, params, x₀)
    return sum(values(losses) .* values(losses_prefactor))
end

"""
    compute_loss_prefactor_density_contribution(::BaseClosureMode, individual_loss, contribution, S_scaling=1.0, momentum_ratio=0.25)

Compute adaptive loss weights based on T/S density contributions and relative magnitudes.

Balances tracer (T, S) vs momentum (u, v) losses and profile vs gradient losses
according to their physical density contributions.

# Arguments
- `::BaseClosureMode`: Training mode
- `individual_loss`: Individual loss components
- `contribution`: Density contributions from T and S (from `compute_density_contribution`)

# Returns
NamedTuple of loss prefactors for each field
"""
function compute_loss_prefactor_density_contribution(::BaseClosureMode, individual_loss, contribution)
    u_loss, v_loss, T_loss, S_loss, ρ_loss, ∂u∂z_loss, ∂v∂z_loss, ∂T∂z_loss, ∂S∂z_loss, ∂ρ∂z_loss = values(individual_loss)

    # Density contribution weighting: profile vs density losses
    # These weights balance density loss (10%) vs tracer losses (90%)
    density_profile_weighting = 0.1
    tracer_profile_weighting = 0.9
    density_weighting_ratio = density_profile_weighting / tracer_profile_weighting

    total_contribution = contribution.T + contribution.S
    T_prefactor = total_contribution / contribution.T
    S_prefactor = total_contribution / contribution.S

    TS_loss = T_prefactor * T_loss + S_prefactor * S_loss

    ρ_prefactor = TS_loss / ρ_loss * density_weighting_ratio

    uv_loss = u_loss + v_loss

    if uv_loss > eps(eltype(uv_loss))
        uv_prefactor = TS_loss / uv_loss
    else
        uv_prefactor = TS_loss * 1000
    end

    ∂T∂z_prefactor = T_prefactor
    ∂S∂z_prefactor = S_prefactor

    ∂TS∂z_loss = ∂T∂z_loss + ∂S∂z_loss

    ∂ρ∂z_prefactor = ∂TS∂z_loss / ∂ρ∂z_loss * density_weighting_ratio

    ∂uv∂z_loss = ∂u∂z_loss + ∂v∂z_loss

    if ∂uv∂z_loss > eps(eltype(∂uv∂z_loss))
        ∂uv∂z_prefactor = ∂TS∂z_loss / ∂uv∂z_loss
    else
        ∂uv∂z_prefactor = ∂TS∂z_loss * 1000
    end

    profile_loss = uv_prefactor * u_loss + uv_prefactor * v_loss + T_prefactor * T_loss + S_prefactor * S_loss + ρ_prefactor * ρ_loss
    gradient_loss = ∂uv∂z_prefactor * ∂u∂z_loss + ∂uv∂z_prefactor * ∂v∂z_loss + ∂T∂z_prefactor * ∂T∂z_loss + ∂S∂z_prefactor * ∂S∂z_loss + ∂ρ∂z_prefactor * ∂ρ∂z_loss

    gradient_prefactor = profile_loss / gradient_loss

    ∂ρ∂z_prefactor *= gradient_prefactor
    ∂T∂z_prefactor *= gradient_prefactor
    ∂S∂z_prefactor *= gradient_prefactor
    ∂uv∂z_prefactor *= gradient_prefactor

    return (u=uv_prefactor, v=uv_prefactor, T=T_prefactor, S=S_prefactor, ρ=ρ_prefactor, ∂u∂z=∂uv∂z_prefactor, ∂v∂z=∂uv∂z_prefactor, ∂T∂z=∂T∂z_prefactor, ∂S∂z=∂S∂z_prefactor, ∂ρ∂z=∂ρ∂z_prefactor)
end

"""
    loss_multipleics(mode::BaseClosureMode, ps, truths, params, x₀s, losses_prefactors)

Compute mean loss across multiple initial conditions.

# Arguments
- `mode::BaseClosureMode`: Training mode
- `ps`: Closure parameters (shared across all ICs)
- `truths`: Vector of ground truth datasets
- `params`: Vector of ODE parameters
- `x₀s`: Vector of initial conditions
- `losses_prefactors`: Vector of loss weight tuples

# Returns
Mean loss across all initial conditions
"""
function loss_multipleics(mode::BaseClosureMode, ps, truths, params, x₀s, losses_prefactors)
    losses = [loss(mode, ps, truth, param, x₀, loss_prefactor) for (truth, x₀, param, loss_prefactor) in zip(truths, x₀s, params, losses_prefactors)]
    return mean(losses)
end

"""
    diagnose_fields(mode::BaseClosureMode, ps, params, x₀, train_data_plot, timestep_multiple=2)

Compute diagnostic fields for analysis and visualization.

Solves NDE and computes Richardson numbers, diffusivities, and fluxes.

# Arguments
- `mode::BaseClosureMode`: Training mode
- `ps`: Closure parameters
- `params`: ODE parameters
- `x₀`: Initial conditions
- `train_data_plot`: Training data for Richardson number comparison
- `timestep_multiple=2`: Temporal refinement factor

# Returns
NamedTuple with:
- `sols_dimensional`: Dimensional solution profiles (u, v, T, S, ρ)
- `fluxes`: Total fluxes (uw, vw, wT, wS)
- `diffusivities`: Diagnosed ν, κ, Ri, and truth Ri
"""
function diagnose_fields(mode::BaseClosureMode, ps, params, x₀, train_data_plot, timestep_multiple=2)
    sols = solve_NDE(mode, ps, params, x₀, timestep_multiple)

    coarse_size = params.coarse_size
    Dᶠ = params.Dᶠ
    scaling = params.scaling

    us = inv(scaling.u).(sols.u)
    vs = inv(scaling.v).(sols.v)
    Ts = inv(scaling.T).(sols.T)
    Ss = inv(scaling.S).(sols.S)
    ρs = inv(scaling.ρ).(sols.ρ)

    eos = TEOS10EquationOfState()
    Ris_truth = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) for (u, v, ρ) in zip(eachcol(train_data_plot.profile.u.unscaled), eachcol(train_data_plot.profile.v.unscaled), eachcol(train_data_plot.profile.ρ.unscaled))]...)
    Ris = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) for (u, v, ρ) in zip(eachcol(us), eachcol(vs), eachcol(ρs))]...)
    
    νs, κs = predict_diffusivities(Ris, ps)

    uw_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    vw_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    wT_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    wS_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))

    for i in 1:size(wT_diffusive_boundarys, 2)
        uw_diffusive_boundarys[:, i], vw_diffusive_boundarys[:, i], wT_diffusive_boundarys[:, i], wS_diffusive_boundarys[:, i] = predict_diffusive_boundary_flux_dimensional(Ris[:, i], sols.u[:, i], sols.v[:, i], sols.T[:, i], sols.S[:, i], ps, params)        
    end

    uw_totals = uw_diffusive_boundarys
    vw_totals = vw_diffusive_boundarys
    wT_totals = wT_diffusive_boundarys
    wS_totals = wS_diffusive_boundarys

    fluxes = (; uw = (; total=uw_totals),
                vw = (; total=vw_totals),
                wT = (; total=wT_totals), 
                wS = (; total=wS_totals))

    diffusivities = (; ν=νs, κ=κs, Ri=Ris, Ri_truth=Ris_truth)

    sols_dimensional = (; u=us, v=vs, T=Ts, S=Ss, ρ=ρs)
    return (; sols_dimensional, fluxes, diffusivities)
end