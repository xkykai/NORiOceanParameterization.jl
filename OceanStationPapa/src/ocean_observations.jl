import Oceananigans: location
import Oceananigans.Fields: set!

using Oceananigans.Architectures: on_architecture
using Oceananigans.DistributedComputations: child_architecture
using Oceananigans.Grids: znodes

struct OceanStationPapaHourly end

const OceanStationPapaMetadata{D} = Metadata{<:OceanStationPapaHourly, D}
const OceanStationPapaMetadatum = Metadatum{<:OceanStationPapaHourly}

metaprefix(::OceanStationPapaMetadata) = "OceanStationPapaMetadata"

default_download_directory(::OceanStationPapaHourly) = mkpath(download_OceanStationPapa_cache)

available_variables(::OceanStationPapaHourly) = ocean_station_papa_dataset_variable_names

const ocean_station_papa_dataset_variable_names = Dict(
    :temperature => "TEMP",
    :salinity => "PSAL",
    :eastward_wind => "UWND",
    :northward_wind => "VWND",
    :air_temperature => "AIRT",
    :relative_humidity => "RELH",
    :sea_level_pressure => "ATMS",
    :shortwave_radiation => "SW",
    :longwave_radiation => "LW",
    :rain => "RAIN",
    :eastward_velocity => "UCUR",
    :northward_velocity => "VCUR",
)

const ocean_station_papa_depth_variable_names = Dict(
    :temperature => "DEPTH",
    :salinity => "DEPPSAL",
    :eastward_velocity => "DEPCUR",
    :northward_velocity => "DEPCUR",
)

dataset_variable_name(data::OceanStationPapaMetadata) = ocean_station_papa_dataset_variable_names[data.name]

location(::OceanStationPapaMetadata) = (Center, Center, Center)
is_three_dimensional(md::OceanStationPapaMetadata) = md.name in (:temperature, :salinity, :eastward_velocity, :northward_velocity)
reversed_vertical_axis(::OceanStationPapaHourly) = true

function conversion_units(metadatum::OceanStationPapaMetadatum)
    name = metadatum.name
    name == :air_temperature && return OceanStationPapaCelsius()
    name == :sea_level_pressure && return OceanStationPapaMillibar()
    name == :rain && return OceanStationPapaMillimetersPerHour()
    name in (:eastward_velocity, :northward_velocity) && return OceanStationPapaCentimetersPerSecond()
    return nothing
end

default_inpainting(::OceanStationPapaMetadata) = nothing

metadata_filename(::OceanStationPapaMetadatum) = OCEAN_STATION_PAPA_FILENAME
metadata_filename(::OceanStationPapaHourly, name, date, bounding_box) = OCEAN_STATION_PAPA_FILENAME

function download_dataset(metadata::OceanStationPapaMetadata)
    download_ocean_station_papa_file(metadata.dir)
    return nothing
end

function inpainted_metadata_path(metadata::OceanStationPapaMetadata)
    filename = metadata_filename(first(metadata))
    without_ext = filename[1:end-3]
    varname = string(metadata.name)
    return joinpath(metadata.dir, without_ext * "_" * varname * "_inpainted.jld2")
end

metadata_epoch(::OceanStationPapaHourly) = DateTime(2007, 6, 7, 23, 0, 0)
metadata_time_step(::OceanStationPapaHourly) = 3600

const ocean_station_papa_times_cache = Ref{Vector{DateTime}}()
const ocean_station_papa_times_cached = Ref(false)

function ocean_station_papa_all_times(dir=download_OceanStationPapa_cache)
    if !ocean_station_papa_times_cached[]
        filepath = download_ocean_station_papa_file(dir)
        NCDataset(filepath) do ds
            ocean_station_papa_times_cache[] = DateTime.(ds["TIME"][:])
        end
        ocean_station_papa_times_cached[] = true
    end
    return ocean_station_papa_times_cache[]
end

all_dates(::OceanStationPapaHourly, variable) = ocean_station_papa_all_times()

const ocean_station_papa_depths_cache = Dict{Symbol, Vector{Float64}}()

function ocean_station_papa_depths(variable, dir=download_OceanStationPapa_cache)
    if !haskey(ocean_station_papa_depths_cache, variable)
        filepath = download_ocean_station_papa_file(dir)
        depths = NCDataset(filepath) do ds
            depthvar = ocean_station_papa_depth_variable_names[variable]
            Float64.(ds[depthvar][:])
        end
        ocean_station_papa_depths_cache[variable] = depths
    end
    return ocean_station_papa_depths_cache[variable]
end

function Base.size(::OceanStationPapaHourly, variable)
    if variable in (:temperature, :salinity, :eastward_velocity, :northward_velocity)
        depths = ocean_station_papa_depths(variable)
        return (1, 1, length(depths))
    else
        return (1, 1, 1)
    end
end

function z_interfaces(dataset::OceanStationPapaHourly; variable=:temperature)
    depths = ocean_station_papa_depths(variable)
    z_centers = sort(-depths)
    return centers_to_interfaces(z_centers)
end

z_interfaces(md::OceanStationPapaMetadata) = z_interfaces(md.dataset; variable=md.name)

longitude_interfaces(::OceanStationPapaHourly) = (OCEAN_STATION_PAPA_LONGITUDE, OCEAN_STATION_PAPA_LONGITUDE)
latitude_interfaces(::OceanStationPapaHourly) = (OCEAN_STATION_PAPA_LATITUDE, OCEAN_STATION_PAPA_LATITUDE)

function native_grid(metadata::OceanStationPapaMetadata, arch=CPU(); halo=(3, 3, 3))
    if is_three_dimensional(metadata)
        Nz = size(metadata.dataset, metadata.name)[3]
        z = z_interfaces(metadata)
        return RectilinearGrid(arch; size=Nz,
                               x=OCEAN_STATION_PAPA_LONGITUDE,
                               y=OCEAN_STATION_PAPA_LATITUDE,
                               z,
                               topology=(Flat, Flat, Bounded),
                               halo=(halo[3],))
    else
        return RectilinearGrid(arch; size=(), topology=(Flat, Flat, Flat))
    end
end

function retrieve_data(metadata::OceanStationPapaMetadatum)
    filepath = metadata_path(metadata)
    return NCDataset(filepath) do ds
        varname = dataset_variable_name(metadata)

        all_times = ds["TIME"][:]
        t_idx = findfirst(t -> t == metadata.dates, all_times)

        isnothing(t_idx) && error("Date $(metadata.dates) not found in Ocean Station Papa dataset")

        if is_three_dimensional(metadata)
            raw = ds[varname][1, 1, :, t_idx]
            qc_varname = varname * "_QC"

            if haskey(ds, qc_varname)
                qc = ds[qc_varname][1, 1, :, t_idx]
                for i in eachindex(raw)
                    q = ismissing(qc[i]) ? Int8(9) : Int8(qc[i])
                    q > 2 && (raw[i] = missing)
                end
            end

            data = Float64.(replace(raw, missing => NaN))
            reverse!(data)
            return reshape(data, 1, 1, :)
        else
            raw = ds[varname][1, 1, 1, t_idx]
            qc_varname = varname * "_QC"

            if haskey(ds, qc_varname)
                qc = ds[qc_varname][1, 1, 1, t_idx]
                q = ismissing(qc) ? Int8(9) : Int8(qc)
                q > 2 && (raw = missing)
            end

            data = Float64(ismissing(raw) ? NaN : raw)
            return reshape([data], 1, 1, 1)
        end
    end
end

function vertical_interpolate(::OceanStationPapaMetadatum, z_src, data_src, z_dst)
    result = similar(z_dst, Float64)

    valid = .!isnan.(data_src)
    zv = z_src[valid]
    dv = data_src[valid]

    if isempty(zv)
        result .= NaN
        return result
    end

    perm = sortperm(zv)
    zv = zv[perm]
    dv = dv[perm]

    for (i, zt) in enumerate(z_dst)
        if zt <= zv[1]
            result[i] = dv[1]
        elseif zt >= zv[end]
            result[i] = dv[end]
        else
            j = searchsortedlast(zv, zt)
            α = (zt - zv[j]) / (zv[j+1] - zv[j])
            result[i] = dv[j] + α * (dv[j+1] - dv[j])
        end
    end

    return result
end

function set!(target_field::Field, metadata::OceanStationPapaMetadatum; kw...)
    grid = target_field.grid
    arch = child_architecture(grid)

    if !is_three_dimensional(metadata)
        meta_field = Field(metadata, arch; kw...)
        parent(target_field) .= parent(meta_field)
        return target_field
    end

    meta_field = Field(metadata, arch; kw...)

    Lzt = grid.Lz
    Lzm = meta_field.grid.Lz

    if Lzt > Lzm
        throw(ArgumentError("The vertical range of the $(metadata.dataset) dataset ($(Lzm) m) is smaller than " *
                            "the target grid ($(Lzt) m). Some vertical levels cannot be filled with data."))
    end

    z_src = collect(znodes(meta_field.grid, Center()))
    z_dst = collect(znodes(grid, Center()))
    data_profile = Array(interior(meta_field, 1, 1, :))
    interpolated = vertical_interpolate(metadata, z_src, data_profile, z_dst)

    interior(target_field, 1, 1, :) .= on_architecture(arch, interpolated)
    return target_field
end
