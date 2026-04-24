using Test
using Dates
using OceanStationPapa
using NumericalEarth.DataWrangling: Metadata, Metadatum, conversion_units

@testset "OceanStationPapa API" begin
    @test OceanStationPapaHourly() isa OceanStationPapaHourly
    @test OceanStationPapaFluxHourly() isa OceanStationPapaFluxHourly
    @test isdefined(OceanStationPapa, :OceanStationPapaPrescribedAtmosphere)
    @test isdefined(OceanStationPapa, :ocean_station_papa_prescribed_fluxes)
    @test isdefined(OceanStationPapa, :ocean_station_papa_prescribed_flux_boundary_conditions)
    @test !isdefined(OceanStationPapa, :OSPapaHourly)
end

@testset "Metadata interface" begin
    start_date = DateTime(2012, 10, 1)
    end_date = DateTime(2012, 10, 1, 2)

    ocean_md = Metadata(:temperature;
                        dataset = OceanStationPapaHourly(),
                        start_date,
                        end_date)

    flux_md = Metadata(:net_heat_flux;
                       dataset = OceanStationPapaFluxHourly(),
                       start_date,
                       end_date)

    @test length(ocean_md) == 3
    @test length(flux_md) == 3
    @test size(OceanStationPapaFluxHourly(), :net_heat_flux) == (1, 1, 1)
    @test first(flux_md).filename == "ocs_papa_flux_uniform_20121001T000000_20121001T020000.nc"
end

@testset "Local unit markers avoid piracy" begin
    T = Metadatum(:air_temperature; dataset = OceanStationPapaHourly(), date = DateTime(2012, 10, 1))
    P = Metadatum(:sea_level_pressure; dataset = OceanStationPapaHourly(), date = DateTime(2012, 10, 1))
    r = Metadatum(:rain; dataset = OceanStationPapaHourly(), date = DateTime(2012, 10, 1))
    u = Metadatum(:eastward_velocity; dataset = OceanStationPapaHourly(), date = DateTime(2012, 10, 1))

    @test conversion_units(T) isa OceanStationPapa.OceanStationPapaCelsius
    @test conversion_units(P) isa OceanStationPapa.OceanStationPapaMillibar
    @test conversion_units(r) isa OceanStationPapa.OceanStationPapaMillimetersPerHour
    @test conversion_units(u) isa OceanStationPapa.OceanStationPapaCentimetersPerSecond

    @test OceanStationPapa.convert_units(1.0, conversion_units(T)) == 274.15
    @test OceanStationPapa.convert_units(1.0, conversion_units(P)) == 100.0
    @test OceanStationPapa.convert_units(3600.0, conversion_units(r)) == 1.0
    @test OceanStationPapa.convert_units(100.0, conversion_units(u)) == 1.0
end
