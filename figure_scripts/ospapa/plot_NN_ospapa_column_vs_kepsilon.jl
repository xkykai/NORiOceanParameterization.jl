using Pkg
Pkg.activate(joinpath(@__DIR__))
# Ensure parent package is available
if !haskey(Pkg.project().dependencies, "NORiOceanParameterization")
    Pkg.develop(path=joinpath(@__DIR__, "..", ".."))
end
if !haskey(Pkg.project().dependencies, "OceanStationPapa")
    Pkg.develop(path=joinpath(@__DIR__, "..", "..", "OceanStationPapa"))
end

include(joinpath(@__DIR__, "register_ospapa_datadep.jl"))

using CairoMakie
using Oceananigans
using Oceananigans.Units
using Statistics
using Dates
using JLD2
using NumericalEarth
using OceanStationPapa

good_years = [2010, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
start_month = 11

start_dates = [DateTime(year, start_month, 1) for year in good_years]
end_dates = start_dates .+ Day(120)

closure_strs = ["kepsilon", "NORi", "NORiBase"]
start_date_strs = [replace(string(start_date), ":" => "-") for start_date in start_dates]
end_date_strs = [replace(string(end_date), ":" => "-") for end_date in end_dates]

inference_dir = get_ospapa_inference_data_path()
filenames = [joinpath(inference_dir, closure_str, "$(start_date_str)_to_$(end_date_str)") for closure_str in closure_strs, (start_date_str, end_date_str) in zip(start_date_strs, end_date_strs)]

T_datas = [FieldTimeSeries(joinpath(filename, "instantaneous_fields"), "T") for filename in filenames]
S_datas = [FieldTimeSeries(joinpath(filename, "instantaneous_fields"), "S") for filename in filenames]
b_datas = [FieldTimeSeries(joinpath(filename, "instantaneous_fields"), "b") for filename in filenames]
#%%
observations_dir = get_ospapa_observations_data_path()
sim_times = T_datas[1].times
Nt = length(sim_times)

obs_T_datas = [jldopen(joinpath(observations_dir, "obs_$(start_date_str)_to_$(end_date_str).jld2"), "r")["T"] for (start_date_str, end_date_str) in zip(start_date_strs, end_date_strs)]
obs_S_datas = [jldopen(joinpath(observations_dir, "obs_$(start_date_str)_to_$(end_date_str).jld2"), "r")["S"] for (start_date_str, end_date_str) in zip(start_date_strs, end_date_strs)]
obs_b_datas = [jldopen(joinpath(observations_dir, "obs_$(start_date_str)_to_$(end_date_str).jld2"), "r")["b"] for (start_date_str, end_date_str) in zip(start_date_strs, end_date_strs)]

#%%
T_sims = zeros(length(closure_strs), length(start_dates), size(T_datas[1], 3), Nt)
S_sims = zeros(length(closure_strs), length(start_dates), size(S_datas[1], 3), Nt)
b_sims = zeros(length(closure_strs), length(start_dates), size(b_datas[1], 3), Nt)

T_obss = zeros(length(start_dates), size(obs_T_datas[1], 3), Nt)
S_obss = zeros(length(start_dates), size(obs_S_datas[1], 3), Nt)
b_obss = zeros(length(start_dates), size(obs_b_datas[1], 3), Nt)

for i in eachindex(closure_strs), j in eachindex(start_dates)
    T_sim = interior(T_datas[i, j], 1, 1, :, :)
    S_sim = interior(S_datas[i, j], 1, 1, :, :)
    b_sim = interior(b_datas[i, j], 1, 1, :, :)

    T_sims[i, j, :, :] .= T_sim
    S_sims[i, j, :, :] .= S_sim
    b_sims[i, j, :, :] .= b_sim
end

for j in eachindex(start_dates)
    T_obss[j, :, :] .= interior(obs_T_datas[j], 1, 1, :, :)
    S_obss[j, :, :] .= interior(obs_S_datas[j], 1, 1, :, :)
    b_obss[j, :, :] .= interior(obs_b_datas[j], 1, 1, :, :)
end

T_sim_means = mean(T_sims, dims=2)
S_sim_means = mean(S_sims, dims=2)
b_sim_means = mean(b_sims, dims=2)

T_obs_means = mean(T_obss, dims=1)
S_obs_means = mean(S_obss, dims=1)
b_obs_means = mean(b_obss, dims=1)
#%%
period_stride = 30 * 24 # Subsample every 30 days (in hours)

subsample_range = (1:period_stride:Nt)[2:end]
days_str = string.(Int.((sim_times[subsample_range] .- sim_times[1]) ./ days))

zCs = znodes(T_datas[1].grid, Center())
t_days = (sim_times .- sim_times[1]) ./ days

colors = Makie.wong_colors();
obs_color = (colors[3], 0.5)

profile_linewidth = 5
LES_linewidth = 12
initial_linewidth = 7

with_theme(theme_latexfonts()) do
    fig = Figure(size=(1500, 1200), fontsize=28)

    axTs = [Axis(fig[1, i], title="Day $(days_str[i])", xlabel = "T (°C)", ylabel = "z (m)") for i in eachindex(subsample_range)]
    axSs = [Axis(fig[3, i], xlabel = "S (psu)", ylabel = "z (m)", xticks=LinearTicks(4)) for i in eachindex(subsample_range)]
    axbs = [Axis(fig[5, i], xlabel = "b (m s⁻²)", ylabel = "z (m)", xticks=LinearTicks(3)) for i in eachindex(subsample_range)]

    for i in eachindex(axTs)
        lines!(axTs[i], vec(T_sim_means[1, 1, :, 1]), zCs, linewidth=initial_linewidth, label="Initial stratification", linestyle=:dash, color=colors[1])
        lines!(axSs[i], vec(S_sim_means[1, 1, :, 1]), zCs, linewidth=initial_linewidth, linestyle=:dash, color=colors[1])
        lines!(axbs[i], vec(b_sim_means[1, 1, :, 1]), zCs, linewidth=initial_linewidth, linestyle=:dash, color=colors[1])
    end

    for i in eachindex(subsample_range)
        lines!(axTs[i], vec(T_obs_means[1, :, subsample_range[i]]), zCs, linewidth=LES_linewidth, color=obs_color, label="Ocean station Papa")
        lines!(axSs[i], vec(S_obs_means[1, :, subsample_range[i]]), zCs, linewidth=LES_linewidth, color=obs_color)
        lines!(axbs[i], vec(b_obs_means[1, :, subsample_range[i]]), zCs, linewidth=LES_linewidth, color=obs_color)

        lines!(axTs[i], vec(T_sim_means[1, 1, :, subsample_range[i]]), zCs, linewidth=profile_linewidth, color=colors[6], label="k-ϵ closure")
        lines!(axSs[i], vec(S_sim_means[1, 1, :, subsample_range[i]]), zCs, linewidth=profile_linewidth, color=colors[6])
        lines!(axbs[i], vec(b_sim_means[1, 1, :, subsample_range[i]]), zCs, linewidth=profile_linewidth, color=colors[6])

        lines!(axTs[i], vec(T_sim_means[3, 1, :, subsample_range[i]]), zCs, linewidth=profile_linewidth, color=colors[2], label="Base closure")
        lines!(axSs[i], vec(S_sim_means[3, 1, :, subsample_range[i]]), zCs, linewidth=profile_linewidth, color=colors[2])
        lines!(axbs[i], vec(b_sim_means[3, 1, :, subsample_range[i]]), zCs, linewidth=profile_linewidth, color=colors[2])

        lines!(axTs[i], vec(T_sim_means[2, 1, :, subsample_range[i]]), zCs, linewidth=profile_linewidth, color=:black, label="NORi closure")
        lines!(axSs[i], vec(S_sim_means[2, 1, :, subsample_range[i]]), zCs, linewidth=profile_linewidth, color=:black)
        lines!(axbs[i], vec(b_sim_means[2, 1, :, subsample_range[i]]), zCs, linewidth=profile_linewidth, color=:black)
    end

    Label(fig[2, :], "Temperature (°C)", tellwidth=false)
    Label(fig[4, :], "Salinity (psu)", tellwidth=false)
    Label(fig[6, :], "Buoyancy (m s⁻²)", tellwidth=false)

    axs = vcat(axTs, axSs, axbs)

    linkxaxes!(axTs...)
    linkxaxes!(axSs...)
    linkxaxes!(axbs...)

    linkyaxes!(axs...)

    hideydecorations!.(axTs[2:end], ticks=false)
    hideydecorations!.(axSs[2:end], ticks=false)
    hideydecorations!.(axbs[2:end], ticks=false)

    hidexdecorations!.(axs, ticks=false, ticklabels=false)
    hideydecorations!.(axs, ticks=false, ticklabels=false, label=false)

    for ax in axs
        hidespines!(ax, :t, :r)
    end

    Legend(fig[7, :], axTs[1], orientation=:horizontal, patchsize=(50, 20))

    display(fig)
    # save(joinpath(@__DIR__, "..", "..", "figures", "NN_ospapa_comparison_column_vs_kepsilon.pdf"), fig)
end