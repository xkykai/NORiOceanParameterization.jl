function ocean_station_papa_specific_humidity_fts(RHa, Ta, Pa, params)
    LX, LY, LZ = location(Ta)
    qa = FieldTimeSeries{LX, LY, LZ}(Ta.grid, Ta.times)
    pqa, pPa, pTa, pRHa = parent(qa), parent(Pa), parent(Ta), parent(RHa)
    pqa .= q_vap_from_RH.(Ref(params), pPa, pTa, pRHa ./ 100, Ref(Liquid()))
    return qa
end

function OceanStationPapaPrescribedAtmosphere(architecture = CPU(), FT = Float32;
                                              start_date = first_date(OceanStationPapaHourly(), :air_temperature),
                                              end_date = last_date(OceanStationPapaHourly(), :air_temperature),
                                              dir = download_OceanStationPapa_cache,
                                              surface_layer_height = 2.5,
                                              max_gap_hours = 72)
    mdkw = (; dataset = OceanStationPapaHourly(), start_date, end_date, dir)
    surface_grid = RectilinearGrid(architecture, FT; size=(), topology=(Flat, Flat, Flat))

    function ocean_station_papa_fts(name)
        md = Metadata(name; mdkw...)
        download_dataset(md)
        fts = FieldTimeSeries(md, surface_grid; time_indices_in_memory = length(md))
        fill_gaps!(fts; max_gap = max_gap_hours)
        return fts
    end

    ua = ocean_station_papa_fts(:eastward_wind)
    va = ocean_station_papa_fts(:northward_wind)
    Ta = ocean_station_papa_fts(:air_temperature)
    Pa = ocean_station_papa_fts(:sea_level_pressure)
    swa = ocean_station_papa_fts(:shortwave_radiation)
    lwa = ocean_station_papa_fts(:longwave_radiation)
    rain = ocean_station_papa_fts(:rain)

    thermo_params = AtmosphereThermodynamicsParameters(FT)
    RHa = ocean_station_papa_fts(:relative_humidity)
    qa = ocean_station_papa_specific_humidity_fts(RHa, Ta, Pa, thermo_params)

    return PrescribedAtmosphere(ua.grid, ua.times;
                                velocities = (u=ua, v=va),
                                tracers = (T=Ta, q=qa),
                                pressure = Pa,
                                freshwater_flux = (; rain),
                                downwelling_radiation = TwoBandDownwellingRadiation(shortwave=swa, longwave=lwa),
                                thermodynamics_parameters = thermo_params,
                                surface_layer_height = convert(FT, surface_layer_height))
end
