module OceanStationPapa

export OceanStationPapaHourly
export OceanStationPapaFluxHourly
export OceanStationPapaPrescribedAtmosphere
export ocean_station_papa_prescribed_fluxes
export ocean_station_papa_prescribed_flux_boundary_conditions

using Dates
using Downloads
using NCDatasets
using Oceananigans
using Scratch
using Thermodynamics: Liquid, q_vap_from_RH

using NumericalEarth.DataWrangling: Metadata, Metadatum, metadata_path,
                                      first_date, last_date, download_progress
using NumericalEarth.Atmospheres: PrescribedAtmosphere
using NumericalEarth.Atmospheres: TwoBandDownwellingRadiation
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters

import NumericalEarth.DataWrangling: all_dates, available_variables, build_filename,
                                     conversion_units, convert_units, dataset_variable_name,
                                     default_download_directory, default_inpainting,
                                     download_dataset, inpainted_metadata_path,
                                     is_three_dimensional, latitude_interfaces,
                                     longitude_interfaces, metadata_epoch, metadata_filename,
                                     metadata_time_step, metaprefix, native_grid,
                                     retrieve_data, reversed_vertical_axis, z_interfaces

const OCEAN_STATION_PAPA_S3_URL = "https://noaa-oar-keo-papa-pds.s3.amazonaws.com/PAPA/"
const OCEAN_STATION_PAPA_FILENAME = "OS_PAPA_200706_M_TSVMBP_50N145W_hr.nc"
const OCEAN_STATION_PAPA_LONGITUDE = -144.9
const OCEAN_STATION_PAPA_LATITUDE = 50.1

download_OceanStationPapa_cache::String = ""

struct OceanStationPapaCelsius end
struct OceanStationPapaMillibar end
struct OceanStationPapaMillimetersPerHour end
struct OceanStationPapaCentimetersPerSecond end

@inline convert_units(T::FT, ::OceanStationPapaCelsius) where FT = T + convert(FT, 273.15)
@inline convert_units(P::FT, ::OceanStationPapaMillibar) where FT = P * convert(FT, 100)
@inline convert_units(r::FT, ::OceanStationPapaMillimetersPerHour) where FT = r / convert(FT, 3600)
@inline convert_units(V::FT, ::OceanStationPapaCentimetersPerSecond) where FT = V / convert(FT, 100)

function __init__()
    global download_OceanStationPapa_cache = @get_scratch!("OceanStationPapa")
end

function download_ocean_station_papa_file(dir=download_OceanStationPapa_cache)
    filepath = joinpath(dir, OCEAN_STATION_PAPA_FILENAME)
    if !isfile(filepath)
        url = OCEAN_STATION_PAPA_S3_URL * OCEAN_STATION_PAPA_FILENAME
        @info "Downloading Ocean Station Papa data from AWS S3..."
        Downloads.download(url, filepath; progress=download_progress)
    end
    return filepath
end

function centers_to_interfaces(z_centers)
    Nz = length(z_centers)
    z_faces = zeros(eltype(z_centers), Nz + 1)

    for k in 1:Nz-1
        z_faces[k+1] = (z_centers[k] + z_centers[k+1]) / 2
    end

    z_faces[1] = z_centers[1] - (z_faces[2] - z_centers[1])
    return z_faces
end

function fill_gaps!(fts::FieldTimeSeries; max_gap=6)
    data_cpu = Array(interior(fts))
    fill_gaps!(data_cpu; max_gap)
    copyto!(interior(fts), data_cpu)
    return fts
end

function fill_gaps!(data::AbstractArray; max_gap=6)
    spatial_inds = CartesianIndices(size(data)[1:end-1])
    for I in spatial_inds
        fill_gaps!(view(data, I, :); max_gap)
    end
    return data
end

function fill_gaps!(data::AbstractVector; max_gap=6)
    N = length(data)
    i = 1

    while i <= N
        if isnan(data[i])
            gap_start = i
            while i <= N && isnan(data[i])
                i += 1
            end

            gap_end = i - 1
            gap_length = gap_end - gap_start + 1

            if gap_start == 1 || gap_end == N
                if gap_start == 1 && gap_end < N
                    data[gap_start:gap_end] .= data[gap_end + 1]
                elseif gap_end == N && gap_start > 1
                    data[gap_start:gap_end] .= data[gap_start - 1]
                end
            elseif gap_length > max_gap
                @warn "Large gap of $gap_length hours at indices $gap_start:$gap_end left unfilled"
            else
                v0 = data[gap_start - 1]
                v1 = data[gap_end + 1]
                for j in gap_start:gap_end
                    α = (j - gap_start + 1) / (gap_length + 1)
                    data[j] = v0 + α * (v1 - v0)
                end
            end
        else
            i += 1
        end
    end

    return data
end

include("ocean_observations.jl")
include("flux_observations.jl")
include("prescribed_atmosphere.jl")
include("prescribed_fluxes.jl")

end # module OceanStationPapa
