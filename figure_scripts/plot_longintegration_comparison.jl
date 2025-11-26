using Oceananigans
using Oceananigans.Units
using CairoMakie
using Makie
using NORiOceanParameterization.Utils: find_min, find_max

#####
##### Configuration
#####

file_types = ["NN", "kepsilon"]
file_labels = ["NORi", "k-ϵ"]
FILE_DIR = joinpath(pwd(), "figure_data", "long_integration_results")

# Common filename pattern for column model results
filename_pattern = "aperiodic_T_86400.0_S_227368.51200000002_dt_300.0_dTdz_0.009765625_dSdz_0.00146484375_QT_0.0002_QS_-2.0e-5_QU_-0.0001_f_0.0001.jld2"

#####
##### Load Column Model Results
#####

ubar_datas = [FieldTimeSeries("$(FILE_DIR)/$(filetype)_$(filename_pattern)", "ubar") for filetype in file_types]
vbar_datas = [FieldTimeSeries("$(FILE_DIR)/$(filetype)_$(filename_pattern)", "vbar") for filetype in file_types]
Tbar_datas = [FieldTimeSeries("$(FILE_DIR)/$(filetype)_$(filename_pattern)", "Tbar") for filetype in file_types]
Sbar_datas = [FieldTimeSeries("$(FILE_DIR)/$(filetype)_$(filename_pattern)", "Sbar") for filetype in file_types]

#####
##### Load LES Reference Data
#####

les_filename = "LES_aperiodic_T_86400.0_S_227368.51200000002_dt_1_dTdz_0.009765625_dSdz_0.00146484375_QT_0.0002_QS_-2.0e-5_QU_-0.0001_f_0.0001.jld2"

ubar_data_LES = FieldTimeSeries("$(FILE_DIR)/$(les_filename)", "ubar")
vbar_data_LES = FieldTimeSeries("$(FILE_DIR)/$(les_filename)", "vbar")
Tbar_data_LES = FieldTimeSeries("$(FILE_DIR)/$(les_filename)", "Tbar")
Sbar_data_LES = FieldTimeSeries("$(FILE_DIR)/$(les_filename)", "Sbar")

#####
##### Setup Vertical Coordinates
#####

zC = znodes(Tbar_datas[1].grid, Center())
zC_LES = znodes(Tbar_data_LES.grid, Center())

times = Tbar_datas[1].times
Nt = length(times)

# Limit vertical range for plotting
z_bottom = -550.0
z_ind = findfirst(x -> x >= z_bottom, zC):length(zC)
z_ind_LES = findfirst(x -> x >= z_bottom, zC_LES):length(zC_LES)

#####
##### Plotting
#####

colors = Makie.wong_colors()
n = 721  # Time index to plot

with_theme(theme_latexfonts()) do
    fig = Figure(size=(1300, 500), fontsize=20)
    
    axu = CairoMakie.Axis(fig[1, 1], xlabel="u (m s⁻¹)", ylabel="z (m)", xticks=LinearTicks(4))
    axv = CairoMakie.Axis(fig[1, 2], xlabel="v (m s⁻¹)", ylabel="z (m)", xticks=LinearTicks(4))
    axT = CairoMakie.Axis(fig[1, 3], xlabel="T (°C)", ylabel="z (m)", xticks=LinearTicks(4))
    axS = CairoMakie.Axis(fig[1, 4], xlabel="S (psu)", ylabel="z (m)", xticks=LinearTicks(4))

    # Extract profiles at time index n
    ubarₙs = [interior(ubar_data[n], 1, 1, z_ind) for ubar_data in ubar_datas]
    vbarₙs = [interior(vbar_data[n], 1, 1, z_ind) for vbar_data in vbar_datas]
    Tbarₙs = [interior(Tbar_data[n], 1, 1, z_ind) for Tbar_data in Tbar_datas]
    Sbarₙs = [interior(Sbar_data[n], 1, 1, z_ind) for Sbar_data in Sbar_datas]

    # Initial conditions (same for all models)
    ubar_initial = interior(ubar_datas[1][1], 1, 1, z_ind)
    vbar_initial = interior(vbar_datas[1][1], 1, 1, z_ind)
    Tbar_initial = interior(Tbar_datas[1][1], 1, 1, z_ind)
    Sbar_initial = interior(Sbar_datas[1][1], 1, 1, z_ind)

    # Compute axis limits
    ulim = (find_min(ubarₙs...) - 3e-3, find_max(ubarₙs...) + 3e-3)
    vlim = (find_min(vbarₙs...) - 3e-3, find_max(vbarₙs...) + 3e-3)
    Tlim = (find_min(Tbarₙs..., Tbar_initial), find_max(Tbarₙs..., Tbar_initial))
    Slim = (find_min(Sbarₙs..., Sbar_initial), find_max(Sbarₙs..., Sbar_initial))

    # Plot initial conditions
    linewidth = 4
    lines!(axu, ubar_initial, zC[z_ind], label="Initial condition", linewidth=linewidth, linestyle=:dash)
    lines!(axv, vbar_initial, zC[z_ind], label="Initial condition", linewidth=linewidth, linestyle=:dash)
    lines!(axT, Tbar_initial, zC[z_ind], label="Initial condition", linewidth=linewidth, linestyle=:dash)
    lines!(axS, Sbar_initial, zC[z_ind], label="Initial condition", linewidth=linewidth, linestyle=:dash)

    # Plot LES reference
    lines!(axu, interior(ubar_data_LES[n], 1, 1, z_ind_LES), zC_LES[z_ind_LES], 
           label="Large-eddy simulation", linewidth=8, color=(colors[3], 0.5))
    lines!(axv, interior(vbar_data_LES[n], 1, 1, z_ind_LES), zC_LES[z_ind_LES], 
           label="Large-eddy simulation", linewidth=8, color=(colors[3], 0.5))
    lines!(axT, interior(Tbar_data_LES[n], 1, 1, z_ind_LES), zC_LES[z_ind_LES], 
           label="Large-eddy simulation", linewidth=8, color=(colors[3], 0.5))
    lines!(axS, interior(Sbar_data_LES[n], 1, 1, z_ind_LES), zC_LES[z_ind_LES], 
           label="Large-eddy simulation", linewidth=8, color=(colors[3], 0.5))

    # Plot column model results
    for (i, filetype) in enumerate(file_types)
        color = filetype == "NN" ? :black : colors[6]
        
        lines!(axu, ubarₙs[i], zC[z_ind], label=file_labels[i], linewidth=linewidth, color=color)
        lines!(axv, vbarₙs[i], zC[z_ind], label=file_labels[i], linewidth=linewidth, color=color)
        lines!(axT, Tbarₙs[i], zC[z_ind], label=file_labels[i], linewidth=linewidth, color=color)
        lines!(axS, Sbarₙs[i], zC[z_ind], label=file_labels[i], linewidth=linewidth, color=color)
    end

    # Set axis limits and styling
    xlims!(axu, ulim)
    xlims!(axv, vlim)
    xlims!(axT, Tlim)
    xlims!(axS, Slim)

    linkyaxes!(axu, axv, axT, axS)

    hidexdecorations!(axu, ticks=false, ticklabels=false, label=false)
    hidexdecorations!(axv, ticks=false, ticklabels=false, label=false)
    hidexdecorations!(axT, ticks=false, ticklabels=false, label=false)
    hidexdecorations!(axS, ticks=false, ticklabels=false, label=false)

    hideydecorations!(axu, ticks=false, ticklabels=false, label=false)
    hideydecorations!(axv, ticks=false)
    hideydecorations!(axT, ticks=false)
    hideydecorations!(axS, ticks=false)

    Legend(fig[2, :], axu, position=:lb, orientation=:horizontal, patchsize=(40, 20))

    display(fig)
    # save("./figures/longintegration_comparison.pdf", fig)
end