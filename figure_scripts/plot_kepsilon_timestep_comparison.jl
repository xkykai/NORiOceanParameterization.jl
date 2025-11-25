using Statistics
using Oceananigans
using CairoMakie
using Oceananigans.Units
using Makie
using NORiOceanParameterization.Utils: find_min, find_max

Δts = [1minute, 30minutes, 60minutes, 120minutes]
Δt_labels = ["1 minute", "30 minutes", "1 hour", "2 hours"]

#%%
LES_FILE_PATH = "./figure_data/timestep_results/LES/timestep_dTdz_0.015_dSdz_0.002_QT_0.0002_QS_-2.0e-5_QU_-0.0001_T_20_S_37_f_0.jld2"

ubar_data_LES = FieldTimeSeries(LES_FILE_PATH, "ubar")
vbar_data_LES = FieldTimeSeries(LES_FILE_PATH, "vbar")
Tbar_data_LES = FieldTimeSeries(LES_FILE_PATH, "Tbar")
Sbar_data_LES = FieldTimeSeries(LES_FILE_PATH, "Sbar")

zC_LES = znodes(Tbar_data_LES.grid, Center())
#%%
dir_type = "kepsilon"
FILE_DIR = "figure_data/timestep_results/$(dir_type)"
ubar_datas = [FieldTimeSeries("$(FILE_DIR)/dt_$(Δt)_dTdz_0.015_dSdz_0.002_QT_0.0002_QS_-2.0e-5_QU_-0.0001_f_0.jld2", "ubar") for Δt in Δts]
vbar_datas = [FieldTimeSeries("$(FILE_DIR)/dt_$(Δt)_dTdz_0.015_dSdz_0.002_QT_0.0002_QS_-2.0e-5_QU_-0.0001_f_0.jld2", "vbar") for Δt in Δts]
Tbar_datas = [FieldTimeSeries("$(FILE_DIR)/dt_$(Δt)_dTdz_0.015_dSdz_0.002_QT_0.0002_QS_-2.0e-5_QU_-0.0001_f_0.jld2", "Tbar") for Δt in Δts]
Sbar_datas = [FieldTimeSeries("$(FILE_DIR)/dt_$(Δt)_dTdz_0.015_dSdz_0.002_QT_0.0002_QS_-2.0e-5_QU_-0.0001_f_0.jld2", "Sbar") for Δt in Δts]

#%%
zC = znodes(Tbar_datas[1].grid, Center())
zF = znodes(Tbar_datas[1].grid, Face())

Nt = length(Tbar_datas[1].times)
colors = Makie.wong_colors();

with_theme(theme_latexfonts()) do
    fig = Figure(size = (1100, 500), fontsize=20)
    axu = CairoMakie.Axis(fig[1, 1], xlabel = "u (m s⁻¹)", ylabel = "z (m)")
    axT = CairoMakie.Axis(fig[1, 2], xlabel = "T (°C)", ylabel = "z (m)")
    axS = CairoMakie.Axis(fig[1, 3], xlabel = "S (psu)", ylabel = "z (m)")

    n = 25

    ubarₙs = [interior(ubar_data[n], 1, 1, :) for ubar_data in ubar_datas]
    vbarₙs = [interior(vbar_data[n], 1, 1, :) for vbar_data in vbar_datas]
    Tbarₙs = [interior(Tbar_data[n], 1, 1, :) for Tbar_data in Tbar_datas]
    Sbarₙs = [interior(Sbar_data[n], 1, 1, :) for Sbar_data in Sbar_datas]

    title_str = "Time: $(round(Tbar_datas[1].times[n] / 86400)) days"

    ulim = (find_min([interior(ubar_data[n]) for ubar_data in ubar_datas[1:end-1]]...) - 1e-2, find_max([interior(ubar_data[n]) for ubar_data in ubar_datas[1:end-1]]...) + 1e-1)
    Tlim = (find_min([interior(Tbar_data) for Tbar_data in Tbar_datas]...), find_max([interior(Tbar_data) for Tbar_data in Tbar_datas]...))
    Slim = (find_min([interior(Sbar_data) for Sbar_data in Sbar_datas]...), find_max([interior(Sbar_data) for Sbar_data in Sbar_datas]...))

    lines!(axu, interior(ubar_data_LES[Nt], 1, 1, :), zC_LES, color=(colors[3], 0.5), label="Large eddy simulation", linewidth=12)
    lines!(axT, interior(Tbar_data_LES[Nt], 1, 1, :), zC_LES, color=(colors[3], 0.5), label="Large eddy simulation", linewidth=12)
    lines!(axS, interior(Sbar_data_LES[Nt], 1, 1, :), zC_LES, color=(colors[3], 0.5), label="Large eddy simulation", linewidth=12)

    linewidth = 4
    line_colors = [:blue, :gray, :red, :purple]

    for (i, Δt) in enumerate(Δts)
        # i_color = ifelse(i < 3, i, i + 1)
        i_color = i
        lines!(axu, ubarₙs[i], zC, label=Δt_labels[i], linewidth=linewidth, color=line_colors[i_color])
        lines!(axT, Tbarₙs[i], zC, label=Δt_labels[i], linewidth=linewidth, color=line_colors[i_color])
        lines!(axS, Sbarₙs[i], zC, label=Δt_labels[i], linewidth=linewidth, color=line_colors[i_color])
    end

    Legend(fig[2, :], axu, position = :lb, orientation=:horizontal, patchsize=(40, 20))

    xlims!(axu, ulim)
    linkyaxes!(axu, axT, axS)

    hidexdecorations!(axu, ticks=false, label=false, ticklabels=false)
    hidexdecorations!(axT, ticks=false, label=false, ticklabels=false)
    hidexdecorations!(axS, ticks=false, label=false, ticklabels=false)

    hideydecorations!(axu, ticks=false, label=false, ticklabels=false)
    hideydecorations!(axT, ticks=false)
    hideydecorations!(axS, ticks=false)

    hidespines!(axu, :t, :r)
    hidespines!(axT, :t, :r)
    hidespines!(axS, :t, :r)

    display(fig)
    # save("./figures/$(dir_type)_timesteps_LES_1D.pdf", fig)
end