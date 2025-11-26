using Oceananigans
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: update_state!
using Printf
using SeawaterPolynomials
using SeawaterPolynomials: TEOS10
using NORiOceanParameterization
using NORiOceanParameterization.Implementation
using Statistics
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: TKEDissipationVerticalDiffusivity
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity, CATKEMixingLength, CATKEEquation

model_architecture = CPU()

const Nz = 96
const Δz = 8
const Lz = Nz * Δz

const dTdz = (30 - 10) / 2048
const dSdz = (37 - 34) / 2048

const T_surface = 30
const S_surface = 37
const Qᵀ = 0.0002
const Qˢ = -2.0e-5
const Qᵁ = -0.0001
const f₀ = 1e-4
const T_period = 1day
const S_period = 2.63158day

T_initial(z) = dTdz * z + T_surface
S_initial(z) = dSdz * z + S_surface

nn_closure = NORiClosureWithNN(arch=model_architecture)
kϵ_closure = TKEDissipationVerticalDiffusivity()

# # CATKE closure with ocean-tuned parameters (Cᵇ=0.28)
# function CATKE_ocean_closure()
#     mixing_length = CATKEMixingLength(Cᵇ=0.28)
#     turbulent_kinetic_energy_equation = CATKEEquation()
#     return CATKEVerticalDiffusivity(; mixing_length, turbulent_kinetic_energy_equation)
# end

# CATKE_closure = CATKE_ocean_closure()

function setup_model(closure)
    grid = RectilinearGrid(CPU(),
                           topology = (Flat, Flat, Bounded),
                           size = Nz,
                           halo = 3,
                           z = (-Lz, 0))

    @inline T_flux(t) = Qᵀ * cos(2π / T_period * t) + Qᵀ / 2
    @inline S_flux(t) = Qˢ * cos(2π / S_period * t)
    
    u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵁ))

    T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(T_flux))
    S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(S_flux))

    coriolis = FPlane(f=f₀)

    if closure isa CATKEVerticalDiffusivity
        tracers = (:T, :S, :e)
    elseif closure isa TKEDissipationVerticalDiffusivity
        tracers = (:T, :S, :e, :ϵ)
    else
        tracers = (:T, :S)
    end

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        free_surface = ImplicitFreeSurface(),
                                        momentum_advection = WENO(grid = grid),
                                        tracer_advection = WENO(grid = grid),
                                        buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
                                        coriolis = coriolis,
                                        closure = closure,
                                        tracers = tracers,
                                        boundary_conditions = (; T = T_bcs, S = S_bcs, u = u_bcs))

    noise(z) = rand() * exp(z / 8)

    T_initial_noisy(z) = T_initial(z) + 1e-6 * noise(z)
    S_initial_noisy(z) = S_initial(z) + 1e-6 * noise(z)
    
    set!(model, T=T_initial_noisy, S=S_initial_noisy)
    update_state!(model)

    return model
end

function run_simulation(model, Δt, OUTPUT_PATH=joinpath(pwd(), "figure_data", "long_integration_results"))
    stop_time = 60days
    simulation = Simulation(model, Δt = Δt, stop_time = stop_time)

    # add progress callback
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

    ubar = Field(Average(u, dims = (1,2)))
    vbar = Field(Average(v, dims = (1,2)))
    Tbar = Field(Average(T, dims = (1,2)))
    Sbar = Field(Average(S, dims = (1,2)))

    averaged_outputs = (; ubar, vbar, Tbar, Sbar)

    mkpath(OUTPUT_PATH)

    if model.closure isa CATKEVerticalDiffusivity
        closure_type = "CATKE" 
    elseif model.closure isa TKEDissipationVerticalDiffusivity
        closure_type = "kepsilon"
    else
        closure_type = "NN"
    end

    simulation.output_writers[:jld2] = JLD2OutputWriter(model, averaged_outputs,
                                                        filename = "$(OUTPUT_PATH)/$(closure_type)_aperiodic_T_$(T_period)_S_$(S_period)_dt_$(Δt)_dTdz_$(dTdz)_dSdz_$(dSdz)_QT_$(Qᵀ)_QS_$(Qˢ)_QU_$(Qᵁ)_f_$(f₀).jld2",
                                                        schedule = TimeInterval(120minutes),
                                                        overwrite_existing = true)

    run!(simulation)
end

setup_and_run_simulation(closure, Δt) = run_simulation(setup_model(closure), Δt)

# closures = [nn_closure, kϵ_closure, CATKE_closure]
closures = [nn_closure, kϵ_closure]
Δt = 5minutes
for (i, closure) in enumerate(closures)
    @info "Running simulation with closure $i and Δt: $Δt"
    setup_and_run_simulation(closure, Δt)
end