using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity, CATKEMixingLength, CATKEEquation
using CairoMakie
using Printf
using Statistics

using SeawaterPolynomials
using SeawaterPolynomials:TEOS10
using NORiOceanParameterization

# Architecture
model_architecture = CPU()
SOBLLES_path = joinpath(get_SOBLLES_data_path(), "SOBLLES_jld2")
data_path = joinpath(SOBLLES_path, "data")
cases = readdir(data_path)
field_datasets = [FieldDataset(joinpath(data_path, sim, "instantaneous_timeseries.jld2"), backend=OnDisk()) for sim in cases]

# number of grid points
const Nz = 32
const Lz = 256

grid = RectilinearGrid(model_architecture,
                       topology = (Flat, Flat, Bounded),
                       size = Nz,
                       halo = 3,
                       z = (-Lz, 0))

OUTPUT_DIR = joinpath(pwd(), "figure_data", "CATKE_inference_results")

function run_inference_CATKE(data, grid, OUTPUT_DIR, prefix)
        dTdz = data.metadata["temperature_gradient"]
        dSdz = data.metadata["salinity_gradient"]

        T_surface = data.metadata["surface_temperature"]
        S_surface = data.metadata["surface_salinity"]

        T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(data.metadata["temperature_flux"]))
        S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(data.metadata["salinity_flux"]))
        u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(data.metadata["momentum_flux"]))

        f₀ = data.metadata["coriolis_parameter"]
        coriolis = FPlane(f=f₀)

        T_initial(z) = dTdz * z + T_surface
        S_initial(z) = dSdz * z + S_surface

        function CATKE_ocean_closure()
            mixing_length = CATKEMixingLength(Cᵇ=0.28)
            turbulent_kinetic_energy_equation = CATKEEquation()
            return CATKEVerticalDiffusivity(; mixing_length, turbulent_kinetic_energy_equation)
        end
        
        closure = CATKE_ocean_closure()

        model = HydrostaticFreeSurfaceModel(
            grid = grid,
            free_surface = ImplicitFreeSurface(),
            momentum_advection = WENO(grid = grid),
            tracer_advection = WENO(grid = grid),
            buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
            coriolis = coriolis,
            closure = closure,
            tracers = (:T, :S, :e),
            boundary_conditions = (; T = T_bcs, S = S_bcs, u = u_bcs),
        )

        noise(z) = rand() * exp(z / 8)

        T_initial_noisy(z) = T_initial(z) + 1e-6 * noise(z)
        S_initial_noisy(z) = S_initial(z) + 1e-6 * noise(z)

        set!(model, T=T_initial_noisy, S=S_initial_noisy)
        update_state!(model)

        Δt₀ = 5minutes
        stop_time = 2days

        simulation = Simulation(model, Δt = Δt₀, stop_time = stop_time)

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

        u, v, w = model.velocities
        T, S = model.tracers.T, model.tracers.S

        @inline function get_density(i, j, k, grid, b, C)
            eos = TEOS10.TEOS10EquationOfState()
            T, S = Oceananigans.BuoyancyModels.get_temperature_and_salinity(b, C)
            @inbounds ρ = TEOS10.ρ(T[i, j, k], S[i, j, k], 0, eos)
            return ρ
        end

        ρ_op = KernelFunctionOperation{Nothing, Nothing, Center}(get_density, model.grid, model.buoyancy, model.tracers)
        ρ = Field(ρ_op)

        ν = model.diffusivity_fields.κc
        κ = model.diffusivity_fields.κu

        ∂u∂z = ∂z(u)
        ∂v∂z = ∂z(v)
        ∂T∂z = ∂z(T)
        ∂S∂z = ∂z(S)
        ∂ρ∂z = ∂z(ρ)

        uw = ν * ∂u∂z
        vw = ν * ∂v∂z
        wT = κ * ∂T∂z
        wS = κ * ∂S∂z

        averaged_outputs = (; u, v, T, S, ρ, ν, κ, uw, vw, wT, wS, ∂u∂z, ∂v∂z, ∂T∂z, ∂S∂z, ∂ρ∂z)

        mkpath(OUTPUT_DIR)
        filepath = joinpath(OUTPUT_DIR, prefix)

        # #####
        # ##### Build checkpointer and output writer
        # #####
        simulation.output_writers[:jld2] = JLD2OutputWriter(model, averaged_outputs,
            filename = filepath,
            schedule = TimeInterval(10minutes),
            overwrite_existing = true)

        @info "Running simulation $(prefix)"

        run!(simulation)
end

results = [run_inference_CATKE(field_dataset, grid, OUTPUT_DIR, case) for (field_dataset, case) in zip(field_datasets, cases)]
