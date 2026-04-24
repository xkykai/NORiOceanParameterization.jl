using Oceananigans.Architectures: on_architecture
using Oceananigans.OutputReaders: Cyclical
using Oceananigans.Units: Time

function ocean_station_papa_prescribed_fluxes(architecture = CPU(), FT = Float64;
                                              start_date = first_date(OceanStationPapaFluxHourly(), :net_heat_flux),
                                              end_date = last_date(OceanStationPapaFluxHourly(), :net_heat_flux),
                                              dir = download_OceanStationPapa_cache,
                                              max_gap_hours = 72)
    mdkw = (; dataset = OceanStationPapaFluxHourly(), start_date, end_date, dir)
    surface_grid = RectilinearGrid(architecture, FT; size=(), topology=(Flat, Flat, Flat))

    function flux_fts(name)
        md = Metadata(name; mdkw...)
        download_dataset(md)
        fts = FieldTimeSeries(md, surface_grid;
                              time_indices_in_memory = length(md),
                              time_indexing = Cyclical())
        fill_gaps!(fts; max_gap = max_gap_hours)
        return fts
    end

    return (; Qnet = flux_fts(:net_heat_flux),
              Qlat = flux_fts(:latent_heat_flux),
              Qsen = flux_fts(:sensible_heat_flux),
              SWnet = flux_fts(:net_shortwave_radiation),
              LWnet = flux_fts(:net_longwave_radiation),
              τx = flux_fts(:zonal_stress),
              τy = flux_fts(:meridional_stress),
              evap = flux_fts(:evaporation),
              rain = flux_fts(:rain),
              EMP = flux_fts(:evaporation_minus_precipitation),
              Tsk = flux_fts(:skin_temperature))
end

no_correction(i, j, grid, clock, model_fields, p) = zero(grid)

function ocean_station_papa_prescribed_flux_boundary_conditions(fluxes;
                                                                arch = nothing,
                                                                ρ₀ = 1020.0,
                                                                cₚ = 3991.0,
                                                                u_momentum_flux_correction = no_correction,
                                                                v_momentum_flux_correction = no_correction,
                                                                temperature_flux_correction = no_correction,
                                                                salinity_flux_correction = no_correction)
    if !isnothing(arch)
        fluxes = map(fts -> on_architecture(arch, fts), fluxes)
    end

    @inline function u_momentum_flux_bc(i, j, grid, clock, model_fields, p)
        return -p.τx[1, 1, 1, Time(clock.time)] / p.ρ₀ + u_momentum_flux_correction(i, j, grid, clock, model_fields, p)
    end

    @inline function v_momentum_flux_bc(i, j, grid, clock, model_fields, p)
        return -p.τy[1, 1, 1, Time(clock.time)] / p.ρ₀ + v_momentum_flux_correction(i, j, grid, clock, model_fields, p)
    end

    @inline function temperature_flux_bc(i, j, grid, clock, model_fields, p)
        return -p.Qnet[1, 1, 1, Time(clock.time)] / (p.ρ₀ * p.cₚ) + temperature_flux_correction(i, j, grid, clock, model_fields, p)
    end

    @inline function salinity_flux_bc(i, j, grid, clock, model_fields, p)
        evaporation_minus_precipitation = p.EMP[1, 1, 1, Time(clock.time)] / (p.ρ₀ * 3600)
        S = model_fields.S[i, j, grid.Nz]
        return -S * evaporation_minus_precipitation + salinity_flux_correction(i, j, grid, clock, model_fields, p)
    end

    u_momentum_flux_params = (; τx=fluxes.τx, ρ₀)
    v_momentum_flux_params = (; τy=fluxes.τy, ρ₀)
    temperature_flux_params = (; Qnet=fluxes.Qnet, ρ₀, cₚ)
    salinity_flux_params = (; EMP=fluxes.EMP, ρ₀)

    u_top = FluxBoundaryCondition(u_momentum_flux_bc, discrete_form=true, parameters=u_momentum_flux_params)
    v_top = FluxBoundaryCondition(v_momentum_flux_bc, discrete_form=true, parameters=v_momentum_flux_params)
    T_top = FluxBoundaryCondition(temperature_flux_bc, discrete_form=true, parameters=temperature_flux_params)
    S_top = FluxBoundaryCondition(salinity_flux_bc, discrete_form=true, parameters=salinity_flux_params)

    return (; u = FieldBoundaryConditions(top=u_top),
              v = FieldBoundaryConditions(top=v_top),
              T = FieldBoundaryConditions(top=T_top),
              S = FieldBoundaryConditions(top=S_top))
end
