using Test

@testset "OceanStationPapa" begin
    repo = normpath(joinpath(@__DIR__, ".."))
    project = joinpath(repo, "OceanStationPapa")
    test_file = joinpath(@__DIR__, "OceanStationPapa", "runtests.jl")
    cmd = `$(Base.julia_cmd()) --project=$project $test_file`

    @test success(cmd)
end
