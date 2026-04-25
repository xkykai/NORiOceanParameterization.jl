# Single-column ocean simulation at OS Papa with prescribed flux boundary conditions
# and a global flux correction to reduce heat and salinity budget discrepancies.
#
# This script sweeps over all closures and years (like column_model_ospapa_prescribedfluxes.jl)
# but applies a per-year global correction computed from the imbalance between observed
# upper-ocean heat content tendency and the prescribed surface flux

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "NORiImplementation"))
if !haskey(Pkg.project().dependencies, "NORiOceanParameterization")
    Pkg.develop(path=joinpath(@__DIR__, ".."))
end
if !haskey(Pkg.project().dependencies, "OceanStationPapa")
    Pkg.develop(path=joinpath(@__DIR__, "..", "OceanStationPapa"))
end

using NumericalEarth
using OceanStationPapa
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: TKEDissipationVerticalDiffusivity
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Models: buoyancy_field
using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Dates
using Printf
using CairoMakie
using CUDA
using SeawaterPolynomials
using SeawaterPolynomials: TEOS10
using Statistics
using JLD2

using NORiImplementation

arch = CPU()

good_years = [2010, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
start_dates = [DateTime(year, 11, 1) for year in good_years]

end_dates = start_dates .+ Day(120)

closures = [NumericalEarth.Oceans.default_ocean_closure(), TKEDissipationVerticalDiffusivity(), NORiClosureWithNN(arch=arch), NORiBaseClosureOnly()]

grid = RectilinearGrid(arch, size = 25,
                       x = -144.9,
                       y = 50.1,
                       z = (-200, 0),
                       topology = (Flat, Flat, Bounded))

coriolis = FPlane(latitude=50.1)

data_dir = joinpath(@__DIR__, "..", "data", "ospapa")
mkpath(data_dir)

free_surface = ImplicitFreeSurface()
timestepper = :QuasiAdamsBashforth2

Δt = 5minutes

ρ₀_fluxes = TEOS10.TEOS10EquationOfState().reference_density
cₚ_fluxes = 3991.0
data_timestep = 1hours

zCs = znodes(grid, Center())
upper_threshold = -100
upper_indices = findall(zCs .>= upper_threshold)
lower_indices = findall(zCs .< upper_threshold)
Lz_upper = length(upper_indices) * grid.z.Δᵃᵃᶜ
Lz_lower = length(lower_indices) * grid.z.Δᵃᵃᶜ

@inline flux_correction(i, j, grid, clock, model_fields, p) = p.correction

for (start_date, end_date) in zip(start_dates, end_dates)
    @info "starting processing for period $(start_date) to $(end_date)"
    stop_time = Dates.value(Second(end_date - start_date))
    times = 0:data_timestep:stop_time
    Nt = length(times)

    start_date_str = replace(string(start_date), ":" => "-")
    end_date_str = replace(string(end_date), ":" => "-")

    # Load observed T/S for this year
    obs_T_metadata = Metadata(:temperature; dataset=OceanStationPapaHourly(), start_date, end_date)
    obs_S_metadata = Metadata(:salinity;    dataset=OceanStationPapaHourly(), start_date, end_date)
    obs_T_fts = FieldTimeSeries(obs_T_metadata, grid, time_indices_in_memory=Nt)
    obs_S_fts = FieldTimeSeries(obs_S_metadata, grid, time_indices_in_memory=Nt)

    # Compute observed buoyancy from T/S using Oceananigans' buoyancy_perturbation kernel
    eos = TEOS10.TEOS10EquationOfState()
    seawater_buoyancy = SeawaterBuoyancy(equation_of_state=eos)

    obs_b_fts = FieldTimeSeries{Center, Center, Center}(grid, obs_T_fts.times)

    for n in 1:Nt
        Tn = obs_T_fts[n]
        Sn = obs_S_fts[n]
        tracers = (; T=Tn, S=Sn)

        b_op = KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbationᶜᶜᶜ, grid, seawater_buoyancy, tracers)
        b_field = Field(b_op)
        compute!(b_field)

        interior(obs_b_fts, :, :, :, n) .= interior(b_field)
    end

    jldopen(joinpath(data_dir, "obs_$(start_date_str)_to_$(end_date_str).jld2"), "w") do file
        file["T"] = obs_T_fts
        file["S"] = obs_S_fts
        file["b"] = obs_b_fts
        file["T_metadata"] = obs_T_metadata
        file["S_metadata"] = obs_S_metadata
    end

    # Compute upper-ocean heat/salt content and tendency
    T_content_upper = FieldTimeSeries{Nothing, Nothing, Nothing}(grid, times)
    S_content_upper = FieldTimeSeries{Nothing, Nothing, Nothing}(grid, times)

    interior(T_content_upper) .= mean(interior(obs_T_fts, :, :, upper_indices, :), dims=3) .* Lz_upper
    interior(S_content_upper) .= mean(interior(obs_S_fts, :, :, upper_indices, :), dims=3) .* Lz_upper

    T_content_tendency_upper = FieldTimeSeries{Nothing, Nothing, Nothing}(grid, times)
    S_content_tendency_upper = FieldTimeSeries{Nothing, Nothing, Nothing}(grid, times)

    interior(T_content_tendency_upper, :, :, :, 2:Nt) .= diff(interior(T_content_upper), dims=4) ./ data_timestep
    interior(S_content_tendency_upper, :, :, :, 2:Nt) .= diff(interior(S_content_upper), dims=4) ./ data_timestep

    # Compute prescribed surface fluxes
    ospapa_fluxes = ocean_station_papa_prescribed_fluxes(; start_date, end_date)

    wT_ospapa = -interior(ospapa_fluxes.Qnet, 1, 1, 1, :) ./ (ρ₀_fluxes * cₚ_fluxes)
    wS_ospapa = -interior(ospapa_fluxes.EMP, 1, 1, 1, :) .* (interior(obs_S_fts, 1, 1, grid.Nz, :)) ./ (ρ₀_fluxes * 1hours)

    mean_ospapa_flux_T = mean(wT_ospapa)
    mean_ospapa_flux_S = mean(wS_ospapa)

    mean_T_tendency_upper = mean(interior(T_content_tendency_upper, 1, 1, 1, :))
    mean_S_tendency_upper = mean(interior(S_content_tendency_upper, 1, 1, 1, :))

    global_wT_adjustment = -(mean_T_tendency_upper + mean_ospapa_flux_T)
    global_wS_adjustment = -(mean_S_tendency_upper + mean_ospapa_flux_S)

    @info "constant temperature flux adjustment = $(global_wT_adjustment), constant salinity flux adjustment = $(global_wS_adjustment)"

    for closure in closures
        if closure isa CATKEVerticalDiffusivity
            closure_str = "CATKE"
        elseif closure isa TKEDissipationVerticalDiffusivity
            closure_str = "kepsilon"
        elseif closure isa Tuple && any(c -> c isa NORiNNFluxClosure, closure)
            closure_str = "NORi"
        elseif closure isa NORiBaseVerticalDiffusivity
            closure_str = "NORiBase"
        end

        @info "Running simulation with closure $(closure_str) from $(start_date) to $(end_date) with global flux correction"

        # Use let block to capture the adjustment values for this year
        T_correction, S_correction = let wT_adj = global_wT_adjustment, wS_adj = global_wS_adjustment
            T_corr(i, j, grid, clock, model_fields, p) = flux_correction(i, j, grid, clock, model_fields, (; correction = wT_adj))
            S_corr(i, j, grid, clock, model_fields, p) = flux_correction(i, j, grid, clock, model_fields, (; correction = wS_adj))
            T_corr, S_corr
        end

        ospapa_bcs = ocean_station_papa_prescribed_flux_boundary_conditions(ospapa_fluxes; arch, temperature_flux_correction=T_correction, salinity_flux_correction=S_correction)

        ocean = ocean_simulation(grid;
                                Δt,
                                coriolis,
                                closure,
                                free_surface,
                                timestepper,
                                momentum_advection = WENO(),
                                tracer_advection = WENO(),
                                boundary_conditions = ospapa_bcs)

        # Set initial conditions from OS Papa buoy profiles
        set!(ocean.model, T=Metadatum(:temperature, dataset=OceanStationPapaHourly(), date=start_date),
                          S=Metadatum(:salinity,    dataset=OceanStationPapaHourly(), date=start_date))

        update_state!(ocean.model)

        simulation = Simulation(ocean.model; Δt, stop_time)

        wall_clock = Ref(time_ns())

        function progress(sim)
            elapsed = 1e-9 * (time_ns() - wall_clock[])
            msg = string("OS Papa buoy, iter: ", iteration(sim),
                        ", time: ", prettytime(sim),
                        ", wall time: ", prettytime(elapsed))
            wall_clock[] = time_ns()

            T = sim.model.tracers.T
            Nz = size(T, 3)
            msg *= @sprintf(", SST: %.2f °C, ", first(interior(T, 1, 1, Nz)))

            Tmax = maximum(sim.model.tracers.T)
            Tmin = minimum(sim.model.tracers.T)
            Smax = maximum(sim.model.tracers.S)
            Smin = minimum(sim.model.tracers.S)

            msg *= @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Tmax, Tmin)
            msg *= @sprintf("extrema(S): (%.2f, %.2f) g/kg, ", Smax, Smin)

            @info msg
            return nothing
        end

        simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))

        u, v, w = ocean.model.velocities
        T, S = ocean.model.tracers.T, ocean.model.tracers.S
        b = buoyancy_field(ocean.model)

        T_integral = Integral(T, dims=(1, 2, 3))
        S_integral = Integral(S, dims=(1, 2, 3))
        b_integral = Integral(b, dims=(1, 2, 3))

        file_dir = "$(start_date_str)_to_$(end_date_str)"

        output_dir = joinpath(pwd(), "figure_data", "ospapa", closure_str, file_dir)
        mkpath(output_dir)

        simulation.output_writers[:jld2] = JLD2Writer(ocean.model, (; u, v, T, S, b, T_integral, S_integral, b_integral);
                                                      filename = joinpath(output_dir, "instantaneous_fields"),
                                                      schedule = TimeInterval(1hours),
                                                      overwrite_existing = true)

        run!(simulation)
    end
end
