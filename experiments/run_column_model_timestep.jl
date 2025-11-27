using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: update_state!
using Printf
using SeawaterPolynomials
using SeawaterPolynomials: TEOS10
using NORiOceanParameterization
using Statistics
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: TKEDissipationVerticalDiffusivity
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity, CATKEMixingLength, CATKEEquation
using SeawaterPolynomials
using SeawaterPolynomials:TEOS10

model_architecture = CPU()

const Nz = 32
const Lz = 256

const dTdz = 0.015
const dSdz = 0.002

const T_surface = 20
const S_surface = 37
const Qᵀ = 0.0002
const Qˢ = -2.0e-5
const Qᵁ = -0.0001
const f₀ = 0

T_initial(z) = dTdz * z + T_surface
S_initial(z) = dSdz * z + S_surface

nn_closure = NORiClosureWithNN(arch=model_architecture)
function CATKE_ocean_closure()
    mixing_length = CATKEMixingLength(Cᵇ=0.28)
    turbulent_kinetic_energy_equation = CATKEEquation()
    return CATKEVerticalDiffusivity(; mixing_length, turbulent_kinetic_energy_equation)
  end
CATKE_closure = CATKE_ocean_closure()
kϵ_closure = TKEDissipationVerticalDiffusivity()

function setup_model(closure)
    grid = RectilinearGrid(CPU(),
                           topology = (Flat, Flat, Bounded),
                           size = Nz,
                           halo = 3,
                           z = (-Lz, 0))
    
    u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵁ))
    T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵀ))
    S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qˢ))

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

function run_simulation(model, Δt, OUTPUT_PATH)
    stop_time = 4days
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

    simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(20))
    
    u, v, w = model.velocities
    T, S = model.tracers.T, model.tracers.S

    ubar = Field(Average(u, dims = (1,2)))
    vbar = Field(Average(v, dims = (1,2)))
    Tbar = Field(Average(T, dims = (1,2)))
    Sbar = Field(Average(S, dims = (1,2)))

    averaged_outputs = (; ubar, vbar, Tbar, Sbar)

    if model.closure isa CATKEVerticalDiffusivity
        prefix = "CATKE"
    elseif model.closure isa TKEDissipationVerticalDiffusivity
        prefix = "kepsilon"
    else
        prefix = "NN"
    end

    SAVE_DIR = joinpath(OUTPUT_PATH, prefix)
    mkpath(SAVE_DIR)

    simulation.output_writers[:jld2] = JLD2OutputWriter(model, averaged_outputs,
                                                        filename = "$(SAVE_DIR)/dt_$(Δt)_dTdz_$(dTdz)_dSdz_$(dSdz)_QT_$(Qᵀ)_QS_$(Qˢ)_QU_$(Qᵁ)_f_$(f₀).jld2",
                                                        schedule = TimeInterval(240minutes),
                                                        overwrite_existing = true)

    run!(simulation)
end

OUTPUT_PATH = joinpath(@__DIR__, "..", "figure_data", "timestep_results")
mkpath(OUTPUT_PATH)

setup_and_run_simulation(closure, Δt) = run_simulation(setup_model(closure), Δt, OUTPUT_PATH)

closures = [nn_closure, CATKE_closure, kϵ_closure]
Δts = [1minute, 5minutes, 15minutes, 30minutes, 60minutes, 120minutes, 240minutes]
for (i, closure) in enumerate(closures), Δt in Δts
    @info "Running simulation with closure $i and Δt: $Δt"
    setup_and_run_simulation(closure, Δt)
end
#%%