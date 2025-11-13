using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: znodes
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Utils: prettysummary
using SeawaterPolynomials
using SeawaterPolynomials: TEOS10
using Printf
using Statistics
using JLD2

# Import NORi closures from Implementation module
using NORiOceanParameterization.Implementation

#####
##### Load validation data
#####
const Δz = 8  # meters
const Nz = 64
const Lz = Δz * Nz

const dTdz = 1.2e-2
const dSdz = 7e-4
const T_surface = 13
const S_surface = 36

# Forcing parameters
const f₀ = -1e-4
const uw_top = -2e-4
const wT_top = 2.5e-4
const wS_top = -3e-5

# Simulation parameters
const Δt = 5minutes
const stop_time = 8days

@info "Validation parameters:"
@info "  Grid: Nz = $Nz, Lz = $Lz m"
@info "  Coriolis: f₀ = $f₀ s⁻¹"
@info "  Surface fluxes: uw = $uw_top, wT = $wT_top, wS = $wS_top"
@info "  Time: Δt = $Δt s, stop_time = $stop_time s"

#####
##### Architecture and Grid Setup
#####

model_architecture = CPU()

grid = RectilinearGrid(model_architecture,
    topology = (Flat, Flat, Bounded),
    size = Nz,
    halo = 3,
    z = (-Lz, 0))

@info "Built a grid: $grid"

#####
##### Boundary Conditions
#####

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(uw_top))
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(wT_top), bottom=GradientBoundaryCondition(dTdz))
S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(wS_top), bottom=GradientBoundaryCondition(dSdz))

#####
##### Coriolis
#####

coriolis = FPlane(f = f₀)

#####
##### Initial Conditions
#####

# Linear stratification matching validation data
T_initial(z) = dTdz * z + T_surface
S_initial(z) = dSdz * z + S_surface

# Add small noise for numerical stability
noise(z) = rand() * exp(z / 8)
T_initial_noisy(z) = T_initial(z) + 1e-6 * noise(z)
S_initial_noisy(z) = S_initial(z) + 1e-6 * noise(z)

#####
##### Closure Setup - Using NORi Implementation Module
#####

@info "Setting up NORi closures..."

# This is the recommended approach - the helper function ensures correct ordering
closures = NORiClosureWithNN(arch=model_architecture)

# Alternative: If you only want the base closure without NN
# closure = NORiBaseClosureOnly()

# Alternative: Mix with other Oceananigans closures
# using Oceananigans.TurbulenceClosures
# scalar_diff = ScalarDiffusivity(ν=1e-2, κ=1e-2)  # Horizontal diffusion
# closure = NORiClosureTuple(:base, :nn, scalar_diff, arch=model_architecture)

@info "Closures configured:"
if closures isa Tuple
    for (i, closure) in enumerate(closures)
        @info "  [$i] $(closure)"
    end
else
    @info "  Single closure: $(closures)"
end

#####
##### Model Building
#####

@info "Building model with NORi closures..."

model = HydrostaticFreeSurfaceModel(
    grid = grid,
    free_surface = ImplicitFreeSurface(),
    momentum_advection = WENO(grid = grid),
    tracer_advection = WENO(grid = grid),
    buoyancy = SeawaterBuoyancy(equation_of_state = TEOS10.TEOS10EquationOfState()),
    coriolis = coriolis,
    closure = closures,  # NORi closures in correct order
    tracers = (:T, :S),
    boundary_conditions = (; T = T_bcs, S = S_bcs, u = u_bcs),
)

@info "Built $model"

#####
##### Set Initial Conditions
#####

set!(model, T = T_initial_noisy, S = S_initial_noisy)
update_state!(model)

#####
##### Simulation Setup
#####

simulation = Simulation(model, Δt = Δt, stop_time = stop_time)

#####
##### Progress Callback
#####

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(T): %6.3e, max(S): %6.3e, next Δt: %s\n",
        100 * (sim.model.clock.time / sim.stop_time),
        sim.model.clock.iteration,
        prettytime(sim.model.clock.time),
        prettytime(1e-9 * (time_ns() - wall_clock[1])),
        maximum(abs, sim.model.tracers.T),
        maximum(abs, sim.model.tracers.S),
        prettytime(sim.Δt))

    wall_clock[1] = time_ns()
    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(20))

#####
##### Diagnostics
#####

u, v, w = model.velocities
T, S = model.tracers.T, model.tracers.S

# Access diffusivity fields based on correct closure ordering
# closures[1] = NORiBaseVerticalDiffusivity (base closure)
# closures[2] = NORiNNFluxClosure (NN closure)

if closures isa Tuple && length(closures) >= 2
    # With both base and NN closures
    Ri = model.diffusivity_fields[1].Ri     # Richardson number from base closure
    ν = model.diffusivity_fields[1].κᵘ      # Viscosity from base closure
    κ = model.diffusivity_fields[1].κᶜ      # Diffusivity from base closure

    wT_residual = model.diffusivity_fields[2].wT  # Temperature flux from NN
    wS_residual = model.diffusivity_fields[2].wS  # Salinity flux from NN

    @info "Diagnostics: Base closure (Ri, ν, κ) + NN closure (wT_residual, wS_residual)"
else
    # Only base closure (no NN)
    Ri = model.diffusivity_fields.Ri
    ν = model.diffusivity_fields.κᵘ
    κ = model.diffusivity_fields.κᶜ

    # No residual fluxes without NN
    wT_residual = nothing
    wS_residual = nothing

    @info "Diagnostics: Base closure only (Ri, ν, κ)"
end

# Compute horizontal averages (for 1D, this just extracts the single column)
ubar = Field(Average(u, dims = (1, 2)))
vbar = Field(Average(v, dims = (1, 2)))
Tbar = Field(Average(T, dims = (1, 2)))
Sbar = Field(Average(S, dims = (1, 2)))

# Build outputs NamedTuple based on available fields
if wT_residual !== nothing
    averaged_outputs = (; ubar, vbar, Tbar, Sbar, wT_residual, wS_residual, ν, κ, Ri)
else
    averaged_outputs = (; ubar, vbar, Tbar, Sbar, ν, κ, Ri)
end

#####
##### Output Writer
#####

output_filename = "nori_column_model_instantaneous_fields"

simulation.output_writers[:jld2] = JLD2OutputWriter(model, averaged_outputs,
    filename = joinpath(pwd(), output_filename),
    schedule = TimeInterval(30minutes),
    overwrite_existing = true)

@info "Output will be saved to: $(output_filename).jld2"

#####
##### Run Simulation
#####

@info "Running the simulation..."

try
    run!(simulation, pickup = false)
catch err
    @error "Simulation failed with error:"
    showerror(stdout, err)
    Base.show_backtrace(stdout, catch_backtrace())
end

#####
##### Visualization (Optional - requires CairoMakie)
#####

# Uncomment the following section to generate comparison plots

# using CairoMakie

# @info "Loading output data for visualization..."

# ubar_data = FieldTimeSeries(joinpath(pwd(), "$(output_filename).jld2"), "ubar")
# vbar_data = FieldTimeSeries(joinpath(pwd(), "$(output_filename).jld2"), "vbar")
# Tbar_data = FieldTimeSeries(joinpath(pwd(), "$(output_filename).jld2"), "Tbar")
# Sbar_data = FieldTimeSeries(joinpath(pwd(), "$(output_filename).jld2"), "Sbar")
# ν_data = FieldTimeSeries(joinpath(pwd(), "$(output_filename).jld2"), "ν")
# κ_data = FieldTimeSeries(joinpath(pwd(), "$(output_filename).jld2"), "κ")
# Ri_data = FieldTimeSeries(joinpath(pwd(), "$(output_filename).jld2"), "Ri")

# if haskey(averaged_outputs, :wT_residual)
#     wT_residual_data = FieldTimeSeries(joinpath(pwd(), "$(output_filename).jld2"), "wT_residual")
#     wS_residual_data = FieldTimeSeries(joinpath(pwd(), "$(output_filename).jld2"), "wS_residual")
# end

# zC = znodes(Tbar_data.grid, Center())
# zF = znodes(Tbar_data.grid, Face())

# Nt = length(Tbar_data.times)

# @info "Creating visualization..."

# fig = Figure(size = (2500, 1000), fontsize=20)

# # Setup axes
# axu = Axis(fig[1, 1], xlabel = "u (m s⁻¹)", ylabel = "z (m)")
# axv = Axis(fig[1, 2], xlabel = "v (m s⁻¹)", ylabel = "z (m)")
# axT = Axis(fig[1, 3], xlabel = "T (°C)", ylabel = "z (m)")
# axS = Axis(fig[1, 4], xlabel = "S (g kg⁻¹)", ylabel = "z (m)")
# axRi = Axis(fig[1, 5], xlabel = "arctan(Ri)", ylabel = "z (m)")
# axν = Axis(fig[2, 3], xlabel = "ν (m² s⁻¹)", ylabel = "z (m)", xscale = log10)
# axκ = Axis(fig[2, 4], xlabel = "κ (m² s⁻¹)", ylabel = "z (m)", xscale = log10)

# if haskey(averaged_outputs, :wT_residual)
#     axwT_residual = Axis(fig[2, 1], xlabel = "wT residual", ylabel = "z (m)")
#     axwS_residual = Axis(fig[2, 2], xlabel = "wS residual", ylabel = "z (m)")
# end

# n = Observable(1)

# # Observables for time-varying data
# ubarₙ = @lift interior(ubar_data[$n], 1, 1, :)
# vbarₙ = @lift interior(vbar_data[$n], 1, 1, :)
# Tbarₙ = @lift interior(Tbar_data[$n], 1, 1, :)
# Sbarₙ = @lift interior(Sbar_data[$n], 1, 1, :)
# νₙ = @lift interior(ν_data[$n], 1, 1, 2:Nz+1)
# κₙ = @lift interior(κ_data[$n], 1, 1, 2:Nz+1)
# Riₙ = @lift atan.(interior(Ri_data[$n], 1, 1, 2:Nz+1))

# if haskey(averaged_outputs, :wT_residual)
#     wT_residualₙ = @lift interior(wT_residual_data[$n], 1, 1, :)
#     wS_residualₙ = @lift interior(wS_residual_data[$n], 1, 1, :)
# end

# title_str = @lift "Time: $(round(Tbar_data.times[$n] / 86400, digits=3)) days"

# # Plot NORi results
# lines!(axu, ubarₙ, zC, label = "NORi", color = :blue)
# lines!(axv, vbarₙ, zC, label = "NORi", color = :blue)
# lines!(axT, Tbarₙ, zC, label = "NORi", color = :blue)
# lines!(axS, Sbarₙ, zC, label = "NORi", color = :blue)
# lines!(axRi, Riₙ, zF[2:Nz+1], label = "NORi", color = :blue)
# lines!(axν, νₙ, zF[2:Nz+1], label = "NORi", color = :blue)
# lines!(axκ, κₙ, zF[2:Nz+1], label = "NORi", color = :blue)

# if haskey(averaged_outputs, :wT_residual)
#     lines!(axwT_residual, wT_residualₙ, zF, label = "NORi", color = :blue)
#     lines!(axwS_residual, wS_residualₙ, zF, label = "NORi", color = :blue)
# end

# # Add legends and title
# Label(fig[0, :], title_str, tellwidth = false)

# # Link y-axes
# if haskey(averaged_outputs, :wT_residual)
#     linkyaxes!(axu, axv, axT, axS, axwT_residual, axwS_residual, axν, axκ, axRi)
# else
#     linkyaxes!(axu, axv, axT, axS, axν, axκ, axRi)
# end

# ulim = (minimum(interior(ubar_data)), maximum(interior(ubar_data)))
# vlim = (minimum(interior(vbar_data)), maximum(interior(vbar_data)))
# Rilim = (-2, 2)

# if haskey(averaged_outputs, :wT_residual)
#     wTlim = (minimum(interior(wT_residual_data)), maximum(interior(wT_residual_data)))
#     wSlim = (minimum(interior(wS_residual_data)), maximum(interior(wS_residual_data)))
# end

# νlim = (1e-6, 10)
# κlim = (1e-6, 10)

# xlims!(axu, ulim)
# xlims!(axv, vlim)
# xlims!(axRi, Rilim)
# xlims!(axν, νlim)
# xlims!(axκ, κlim)
# if haskey(averaged_outputs, :wT_residual)
#     xlims!(axwT_residual, wTlim)
#     xlims!(axwS_residual, wSlim)
# end

# @info "Displaying visualization..."
# display(fig)

# # Optional: Record animation
# CairoMakie.record(fig, joinpath(pwd(), "nori_column_model.mp4"), 1:Nt, framerate=30) do nn
#     @info "Frame $nn / $Nt"
#     n[] = nn
# end

# @info "Script completed!"
