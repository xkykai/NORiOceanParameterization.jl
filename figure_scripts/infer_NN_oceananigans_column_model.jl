using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: update_state!
using Printf
using SeawaterPolynomials
using SeawaterPolynomials: TEOS10
using NORiOceanParameterization
using NORiOceanParameterization.Implementation

#####
##### Setup
#####

model_architecture = CPU()

# Load LES training data
SOBLLES_path = joinpath(get_SOBLLES_data_path(), "SOBLLES_jld2")
data_path = joinpath(SOBLLES_path, "data")
cases = readdir(data_path)
field_datasets = [FieldDataset(joinpath(data_path, sim, "instantaneous_timeseries.jld2"), backend=OnDisk()) for sim in cases]

# Grid configuration
const Nz = 32
const Lz = 256

grid = RectilinearGrid(model_architecture,
                       topology = (Flat, Flat, Bounded),
                       size = Nz,
                       halo = 3,
                       z = (-Lz, 0))

OUTPUT_DIR = joinpath(pwd(), "figure_data", "NN_inference_results")

#####
##### Run NN model simulation
#####

"""
    run_inference_NN(data, grid, OUTPUT_DIR, prefix)

Run column model simulation using NORi closure with neural network.
Combines base closure (Richardson number dependent diffusivity) with NN-predicted residual fluxes.
Matches initial conditions and forcing from LES data.
"""
function run_inference_NN(data, grid, OUTPUT_DIR, prefix)
    # Extract metadata from LES data
    dTdz = data.metadata["temperature_gradient"]
    dSdz = data.metadata["salinity_gradient"]
    T_surface = data.metadata["surface_temperature"]
    S_surface = data.metadata["surface_salinity"]
    f₀ = data.metadata["coriolis_parameter"]

    # Boundary conditions matching LES forcing
    T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(data.metadata["temperature_flux"]))
    S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(data.metadata["salinity_flux"]))
    u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(data.metadata["momentum_flux"]))

    coriolis = FPlane(f=f₀)

    # Linear initial stratification
    T_initial(z) = dTdz * z + T_surface
    S_initial(z) = dSdz * z + S_surface

    # NORi closure: base closure + NN correction
    closures = NORiClosureWithNN(arch=grid.architecture)

    model = HydrostaticFreeSurfaceModel(
        grid = grid,
        free_surface = ImplicitFreeSurface(),
        momentum_advection = WENO(grid = grid),
        tracer_advection = WENO(grid = grid),
        buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
        coriolis = coriolis,
        closure = closures,
        tracers = (:T, :S),
        boundary_conditions = (; T = T_bcs, S = S_bcs, u = u_bcs),
    )

    # Add small noise for numerical stability
    noise(z) = rand() * exp(z / 8)
    T_initial_noisy(z) = T_initial(z) + 1e-6 * noise(z)
    S_initial_noisy(z) = S_initial(z) + 1e-6 * noise(z)

    set!(model, T=T_initial_noisy, S=S_initial_noisy)
    update_state!(model)

    # Simulation configuration
    Δt₀ = 5minutes
    stop_time = 2days

    simulation = Simulation(model, Δt = Δt₀, stop_time = stop_time)

    # Progress monitoring
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

    simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(200))

    #####
    ##### Diagnostics
    #####

    u, v, w = model.velocities
    T, S = model.tracers.T, model.tracers.S

    # Custom kernel for computing density from T and S
    @inline function get_density(i, j, k, grid, b, C)
        eos = TEOS10.TEOS10EquationOfState()
        T, S = Oceananigans.BuoyancyModels.get_temperature_and_salinity(b, C)
        @inbounds ρ = TEOS10.ρ(T[i, j, k], S[i, j, k], 0, eos)
        return ρ
    end

    ρ_op = KernelFunctionOperation{Nothing, Nothing, Center}(get_density, model.grid, model.buoyancy, model.tracers)
    ρ = Field(ρ_op)

    # Base closure fields (first element in closure tuple)
    Ri = model.diffusivity_fields[1].Ri     # Richardson number
    ν = model.diffusivity_fields[1].κᵘ      # Viscosity
    κ = model.diffusivity_fields[1].κᶜ      # Diffusivity

    # NN residual fluxes (second element in closure tuple)
    wT_residual = model.diffusivity_fields[2].wT  # NN-predicted temperature flux
    wS_residual = model.diffusivity_fields[2].wS  # NN-predicted salinity flux

    # Gradients
    ∂u∂z = ∂z(u)
    ∂v∂z = ∂z(v)
    ∂T∂z = ∂z(T)
    ∂S∂z = ∂z(S)
    ∂ρ∂z = ∂z(ρ)

    # Base closure diffusive fluxes (down-gradient approximation)
    uw = ν * ∂u∂z
    vw = ν * ∂v∂z
    wT = κ * ∂T∂z
    wS = κ * ∂S∂z

    averaged_outputs = (; u, v, T, S, ρ, ν, κ, Ri, uw, vw, wT, wS, wT_residual, wS_residual, ∂u∂z, ∂v∂z, ∂T∂z, ∂S∂z, ∂ρ∂z)

    #####
    ##### Output writer
    #####

    mkpath(OUTPUT_DIR)
    filepath = joinpath(OUTPUT_DIR, prefix)

    simulation.output_writers[:jld2] = JLD2OutputWriter(model, averaged_outputs,
        filename = filepath,
        schedule = TimeInterval(10minutes),
        overwrite_existing = true)

    @info "Running simulation $(prefix)"

    run!(simulation)
end

#####
##### Run all cases
#####

for (field_dataset, case) in zip(field_datasets, cases)
    run_inference_NN(field_dataset, grid, OUTPUT_DIR, case)
end