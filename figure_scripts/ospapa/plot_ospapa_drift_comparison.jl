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
using RollingFunctions
using OceanStationPapa

good_years = [2010, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
start_month = 11

start_dates = [DateTime(year, start_month, 1) for year in good_years]
end_dates = start_dates .+ Day(120)

closure_strs = ["kepsilon", "CATKE", "NORi", "NORiBase"]
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
T_diffs = zeros(length(closure_strs), length(start_dates), size(T_datas[1], 3), Nt)
S_diffs = zeros(length(closure_strs), length(start_dates), size(S_datas[1], 3), Nt)
b_diffs = zeros(length(closure_strs), length(start_dates), size(b_datas[1], 3), Nt)

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

    T_obs = interior(obs_T_datas[j], 1, 1, :, :)
    S_obs = interior(obs_S_datas[j], 1, 1, :, :)
    b_obs = interior(obs_b_datas[j], 1, 1, :, :)

    T_diffs[i, j, :, :] .= T_sim .- T_obs
    S_diffs[i, j, :, :] .= S_sim .- S_obs
    b_diffs[i, j, :, :] .= b_sim .- b_obs

    T_sims[i, j, :, :] .= T_sim
    S_sims[i, j, :, :] .= S_sim
    b_sims[i, j, :, :] .= b_sim
end

for j in eachindex(start_dates)
    T_obss[j, :, :] .= interior(obs_T_datas[j], 1, 1, :, :)
    S_obss[j, :, :] .= interior(obs_S_datas[j], 1, 1, :, :)
    b_obss[j, :, :] .= interior(obs_b_datas[j], 1, 1, :, :)
end

T_diff_means = mean(T_diffs, dims=2)
S_diff_means = mean(S_diffs, dims=2)
b_diff_means = mean(b_diffs, dims=2)

T_drift_means = mean(T_diffs, dims=(2, 3))
S_drift_means = mean(S_diffs, dims=(2, 3))
b_drift_means = mean(b_diffs, dims=(2, 3))

windowsize = 24

for i in eachindex(closure_strs)
    T_drift_means[i, 1, 1, :] .= runmean(T_drift_means[i, 1, 1, :], windowsize)
    S_drift_means[i, 1, 1, :] .= runmean(S_drift_means[i, 1, 1, :], windowsize)
    b_drift_means[i, 1, 1, :] .= runmean(b_drift_means[i, 1, 1, :], windowsize)
end

T_sim_means = mean(T_sims, dims=2)
S_sim_means = mean(S_sims, dims=2)
b_sim_means = mean(b_sims, dims=2)

T_obs_means = mean(T_obss, dims=1)
S_obs_means = mean(S_obss, dims=1)
b_obs_means = mean(b_obss, dims=1)
#%%
colors = Makie.wong_colors();
zCs = znodes(T_datas[1].grid, Center())
t_days = (sim_times .- sim_times[1]) ./ days

T_difflim = (-maximum(abs.(T_diff_means)), maximum(abs.(T_diff_means)))
S_difflim = (-maximum(abs.(S_diff_means)), maximum(abs.(S_diff_means)))
b_difflim = (-maximum(abs.(b_diff_means)), maximum(abs.(b_diff_means)))

linewidth = 7

T_diff_levels = range(T_difflim[1], T_difflim[2], length=10)
S_diff_levels = range(S_difflim[1], S_difflim[2], length=10)
b_diff_levels = range(b_difflim[1], b_difflim[2], length=10)

yticks = LinearTicks(4)

with_theme(theme_latexfonts()) do
    fig = Figure(size=(2200, 1800), fontsize=35)

    axobs_T = Axis(fig[1, 1], title="Ocean station Papa", xlabel="Time (days)", ylabel="z (m)", yticks=yticks)
    axobs_S = Axis(fig[2, 1], xlabel="Time (days)", ylabel="z (m)", yticks=yticks)
    axobs_b = Axis(fig[3, 1], xlabel="Time (days)", ylabel="z (m)", yticks=yticks)

    axNORi_T = Axis(fig[1, 3], title="NORi closure", xlabel="Time (days)", ylabel="z (m)", yticks=yticks)
    axNORi_S = Axis(fig[2, 3], xlabel="Time (days)", ylabel="z (m)", yticks=yticks)
    axNORi_b = Axis(fig[3, 3], xlabel="Time (days)", ylabel="z (m)", yticks=yticks)

    axbase_T = Axis(fig[1, 4], title="Base closure", xlabel="Time (days)", ylabel="z (m)", yticks=yticks)
    axbase_S = Axis(fig[2, 4], xlabel="Time (days)", ylabel="z (m)", yticks=yticks)
    axbase_b = Axis(fig[3, 4], xlabel="Time (days)", ylabel="z (m)", yticks=yticks)

    axkepsilon_T = Axis(fig[1, 5], title="k-ϵ closure", xlabel="Time (days)", ylabel="z (m)", yticks=yticks)
    axkepsilon_S = Axis(fig[2, 5], xlabel="Time (days)", ylabel="z (m)", yticks=yticks)
    axkepsilon_b = Axis(fig[3, 5], xlabel="Time (days)", ylabel="z (m)", yticks=yticks)

    axCATKE_T = Axis(fig[1, 6], title="CATKE closure", xlabel="Time (days)", ylabel="z (m)", yticks=yticks)
    axCATKE_S = Axis(fig[2, 6], xlabel="Time (days)", ylabel="z (m)", yticks=yticks)
    axCATKE_b = Axis(fig[3, 6], xlabel="Time (days)", ylabel="z (m)", yticks=yticks)

    hmobs_T = contourf!(axobs_T, t_days, zCs, T_obs_means[1, :, :]', colormap=:turbo, levels=10)
    hmobs_S = contourf!(axobs_S, t_days, zCs, S_obs_means[1, :, :]', colormap=:turbo, levels=20)
    hmobs_b = contourf!(axobs_b, t_days, zCs, b_obs_means[1, :, :]', colormap=:turbo, levels=12)

    hmNORi_T = contourf!(axNORi_T, t_days, zCs, T_diff_means[3, 1, :, :]', colormap=:balance, levels=T_diff_levels)
    hmNORi_S = contourf!(axNORi_S, t_days, zCs, S_diff_means[3, 1, :, :]', colormap=:balance, levels=S_diff_levels)
    hmNORi_b = contourf!(axNORi_b, t_days, zCs, b_diff_means[3, 1, :, :]', colormap=:balance, levels=b_diff_levels)

    hmbase_T = contourf!(axbase_T, t_days, zCs, T_diff_means[4, 1, :, :]', colormap=:balance, levels=T_diff_levels)
    hmbase_S = contourf!(axbase_S, t_days, zCs, S_diff_means[4, 1, :, :]', colormap=:balance, levels=S_diff_levels)
    hmbase_b = contourf!(axbase_b, t_days, zCs, b_diff_means[4, 1, :, :]', colormap=:balance, levels=b_diff_levels)

    hmkepsilon_T = contourf!(axkepsilon_T, t_days, zCs, T_diff_means[1, 1, :, :]', colormap=:balance, levels=T_diff_levels)
    hmkepsilon_S = contourf!(axkepsilon_S, t_days, zCs, S_diff_means[1, 1, :, :]', colormap=:balance, levels=S_diff_levels)
    hmkepsilon_b = contourf!(axkepsilon_b, t_days, zCs, b_diff_means[1, 1, :, :]', colormap=:balance, levels=b_diff_levels)

    hmCATKE_T = contourf!(axCATKE_T, t_days, zCs, T_diff_means[2, 1, :, :]', colormap=:balance, levels=T_diff_levels)
    hmCATKE_S = contourf!(axCATKE_S, t_days, zCs, S_diff_means[2, 1, :, :]', colormap=:balance, levels=S_diff_levels)
    hmCATKE_b = contourf!(axCATKE_b, t_days, zCs, b_diff_means[2, 1, :, :]', colormap=:balance, levels=b_diff_levels)

    Colorbar(fig[1, 2], hmobs_T, label="Temperature (°C)")
    Colorbar(fig[2, 2], hmobs_S, label="Salinity (psu)")
    Colorbar(fig[3, 2], hmobs_b, label="Buoyancy (m s⁻²)")

    Colorbar(fig[1, 7], hmNORi_T, label="Temperature difference (°C)", colorrange=T_difflim)
    Colorbar(fig[2, 7], hmNORi_S, label="Salinity difference (psu)", colorrange=S_difflim)
    Colorbar(fig[3, 7], hmNORi_b, label="Buoyancy difference (m s⁻²)", colorrange=b_difflim)

    axT_drift = Axis(fig[4, :], xlabel="Time (days)", ylabel="Temperature drift (°C)", yticklabelcolor = :blue, ylabelcolor=:blue)
    axS_drift = Axis(fig[4, :], xlabel="Time (days)", ylabel="Salinity drift (psu)", yticklabelcolor = :red, yaxisposition = :right, ylabelcolor=:red)

    hlines!(axT_drift, [0], color=:black, linestyle=:dash, linewidth=linewidth)
    hlines!(axS_drift, [0], color=:black, linestyle=:dash, linewidth=linewidth)

    lines!(axT_drift, t_days, T_drift_means[1, 1, 1, :], linewidth=linewidth, color=:blue)
    lines!(axS_drift, t_days, S_drift_means[1, 1, 1, :], linewidth=linewidth, color=:red)

    axTs = [axobs_T, axNORi_T, axbase_T, axkepsilon_T, axCATKE_T]
    axSs = [axobs_S, axNORi_S, axbase_S, axkepsilon_S, axCATKE_S]
    axbs = [axobs_b, axNORi_b, axbase_b, axkepsilon_b, axCATKE_b]

    axs = vcat(axTs, axSs, axbs)

    hidexdecorations!.(axTs, ticks=false)
    hidexdecorations!.(axSs, ticks=false)

    hideydecorations!.(axTs[2:end], ticks=false)
    hideydecorations!.(axSs[2:end], ticks=false)
    hideydecorations!.(axbs[2:end], ticks=false)

    xlims!.(axs, 0, maximum(t_days)-0.1)
    ylims!.(axs, minimum(zCs), maximum(zCs))

    hidexdecorations!(axT_drift, ticks=false, ticklabels=false, label=false)
    hideydecorations!(axT_drift, ticks=false, ticklabels=false, label=false)

    hidespines!(axS_drift)
    hidexdecorations!(axS_drift)
    hideydecorations!(axS_drift, ticks=false, ticklabels=false, label=false)

    ylims!(axT_drift, -0.11, 0.11)
    ylims!(axS_drift, -0.035, 0.035)

    display(fig)
    # save(joinpath(@__DIR__, "..", "..", "figures", "NN_ospapa_comparison_drift.pdf"), fig)
end