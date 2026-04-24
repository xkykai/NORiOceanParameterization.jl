import Oceananigans: location
import Oceananigans.Fields: set!

using Oceananigans.DistributedComputations: child_architecture

struct OceanStationPapaFluxHourly end

const OceanStationPapaFluxMetadata{D} = Metadata{<:OceanStationPapaFluxHourly, D}
const OceanStationPapaFluxMetadatum = Metadatum{<:OceanStationPapaFluxHourly}

metaprefix(::OceanStationPapaFluxMetadata) = "OceanStationPapaFluxMetadata"

default_download_directory(::OceanStationPapaFluxHourly) = mkpath(download_OceanStationPapa_cache)

available_variables(::OceanStationPapaFluxHourly) = ocean_station_papa_flux_variable_names

const ocean_station_papa_flux_variable_names = Dict(
    :net_heat_flux => "QNET",
    :latent_heat_flux => "QLAT",
    :sensible_heat_flux => "QSEN",
    :net_shortwave_radiation => "SWNET",
    :net_longwave_radiation => "LWNET",
    :zonal_stress => "TAUX",
    :meridional_stress => "TAUY",
    :evaporation => "EVAP",
    :rain => "RAIN",
    :evaporation_minus_precipitation => "EMP",
    :skin_temperature => "TSK",
)

dataset_variable_name(md::OceanStationPapaFluxMetadata) = ocean_station_papa_flux_variable_names[md.name]

location(::OceanStationPapaFluxMetadata) = (Center, Center, Center)
is_three_dimensional(::OceanStationPapaFluxMetadata) = false
conversion_units(::OceanStationPapaFluxMetadatum) = nothing
default_inpainting(::OceanStationPapaFluxMetadata) = nothing

Base.size(::OceanStationPapaFluxHourly, variable) = (1, 1, 1)

metadata_epoch(::OceanStationPapaFluxHourly) = DateTime(2007, 6, 8)
metadata_time_step(::OceanStationPapaFluxHourly) = 3600

const OCEAN_STATION_PAPA_FLUX_ALL_DATES = DateTime(2007, 6, 8):Hour(1):DateTime(2022, 2, 24)

all_dates(::OceanStationPapaFluxHourly, variable) = OCEAN_STATION_PAPA_FLUX_ALL_DATES

longitude_interfaces(::OceanStationPapaFluxHourly) = (OCEAN_STATION_PAPA_LONGITUDE, OCEAN_STATION_PAPA_LONGITUDE)
latitude_interfaces(::OceanStationPapaFluxHourly) = (OCEAN_STATION_PAPA_LATITUDE, OCEAN_STATION_PAPA_LATITUDE)

function native_grid(::OceanStationPapaFluxMetadata, arch=CPU(); halo=(3, 3, 3))
    return RectilinearGrid(arch; size=(), topology=(Flat, Flat, Flat))
end

const ERDDAP_BASE = "https://data.pmel.noaa.gov/pmel/erddap/tabledap"
const ERDDAP_FLUX_VARS = "time,QLAT,QSEN,QNET,LWNET,SWNET,TAU,TAUX,TAUY,RAIN,EVAP,EMP,TSK"

function download_ocean_station_papa_flux(; start_date, end_date, dir=download_OceanStationPapa_cache)
    filename = "ocs_papa_flux_raw_$(Dates.format(start_date, "yyyymmddTHHMMSS"))_$(Dates.format(end_date, "yyyymmddTHHMMSS")).nc"
    filepath = joinpath(dir, filename)

    if !isfile(filepath)
        t0 = Dates.format(start_date, "yyyy-mm-ddTHH:MM:SSZ")
        t1 = Dates.format(end_date, "yyyy-mm-ddTHH:MM:SSZ")
        url = "$(ERDDAP_BASE)/ocs_papa_flux.nc?$(ERDDAP_FLUX_VARS)&time>=$(t0)&time<=$(t1)"
        @info "Downloading Ocean Station Papa flux data from ERDDAP..."
        Downloads.download(url, filepath; progress=download_progress)
    end

    return filepath
end

flux_uniform_filename(start_date, end_date) =
    "ocs_papa_flux_uniform_$(Dates.format(start_date, "yyyymmddTHHMMSS"))_$(Dates.format(end_date, "yyyymmddTHHMMSS")).nc"

metadata_filename(::OceanStationPapaFluxHourly, name, date, bounding_box) = flux_uniform_filename(date, date)

build_filename(::OceanStationPapaFluxHourly, name, dates::AbstractArray, bounding_box) =
    flux_uniform_filename(first(dates), last(dates))

function download_dataset(md::OceanStationPapaFluxMetadata)
    uniform_path = joinpath(md.dir, metadata_filename(md))
    isfile(uniform_path) && return nothing

    if !(md.dates isa AbstractArray)
        error("OceanStationPapaFluxHourly uniform cache $(uniform_path) is missing; " *
              "construct ocean_station_papa_prescribed_fluxes or multi-date Metadata first.")
    end

    start_date = first(md.dates)
    end_date = last(md.dates)
    raw_path = download_ocean_station_papa_flux(; start_date, end_date, dir=md.dir)
    write_uniform_flux_file(raw_path, uniform_path, start_date, end_date)
    return nothing
end

function write_uniform_flux_file(raw_path, uniform_path, start_date, end_date)
    uniform_datetimes = start_date:Hour(1):end_date
    expanded = NCDataset(raw_path) do ds
        raw_times = DateTime.(ds["time"][:])
        dt_to_raw_idx = Dict(t => i for (i, t) in enumerate(raw_times))

        expanded = Dict{String, Vector{Float64}}()
        for ncname in values(ocean_station_papa_flux_variable_names)
            raw = Float64.(replace(ds[ncname][:], missing => NaN))
            uniform = fill(NaN, length(uniform_datetimes))
            for (j, t) in enumerate(uniform_datetimes)
                i = get(dt_to_raw_idx, t, nothing)
                isnothing(i) || (uniform[j] = raw[i])
            end
            expanded[ncname] = uniform
        end
        expanded
    end

    N = length(uniform_datetimes)
    NCDataset(uniform_path, "c") do out
        defDim(out, "X", 1)
        defDim(out, "Y", 1)
        defDim(out, "Z", 1)
        defDim(out, "TIME", N)

        time_var = defVar(out, "TIME", Float64, ("TIME",);
                          attrib=Dict("units" => "seconds since 1970-01-01 00:00:00",
                                      "calendar" => "standard"))
        time_var[:] = [Dates.datetime2unix(t) for t in uniform_datetimes]

        for (ncname, data) in expanded
            v = defVar(out, ncname, Float64, ("X", "Y", "Z", "TIME"))
            v[:, :, :, :] = reshape(data, 1, 1, 1, N)
        end
    end

    return uniform_path
end

function retrieve_data(metadata::OceanStationPapaFluxMetadatum)
    filepath = metadata_path(metadata)
    return NCDataset(filepath) do ds
        varname = dataset_variable_name(metadata)

        all_times = DateTime.(ds["TIME"][:])
        t_idx = findfirst(t -> t == metadata.dates, all_times)

        isnothing(t_idx) && error("Date $(metadata.dates) not found in Ocean Station Papa flux dataset")

        raw = ds[varname][1, 1, 1, t_idx]
        data = Float64(ismissing(raw) ? NaN : raw)
        return reshape([data], 1, 1, 1)
    end
end

function set!(target_field::Field, metadata::OceanStationPapaFluxMetadatum; kw...)
    grid = target_field.grid
    arch = child_architecture(grid)
    meta_field = Field(metadata, arch; kw...)
    parent(target_field) .= parent(meta_field)
    return target_field
end
