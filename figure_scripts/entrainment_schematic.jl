using CairoMakie

zs = -1:0.1:0
initial_stratification = 0:0.1:1
bs_noentrainment = vcat(collect(0:0.1:0.5), repeat([0.5], 5))
zs_fine = -1:0.01:0
b_entrainment = (0.375 - 0.3^2 / 2) / 0.7
bs_entrainment = vcat(collect(0:0.01:0.3), repeat([b_entrainment], 70))

with_theme(theme_latexfonts()) do
    fig = Figure(size = (650, 600), fontsize=20)
    ax = Axis(fig[1, 1], xlabel = "Buoyancy", ylabel = "Depth", yticks = ([-1, 0], ["Interior", "Surface"]))
    lines!(ax, initial_stratification, zs, linewidth = 4, label = L"t = 0", linestyle = (:dash, :dense))
    lines!(ax, bs_noentrainment, zs, linewidth = 4, label = L"$t = t_0$, no entrainment")
    lines!(ax, bs_entrainment, zs_fine, linewidth = 4, label = L"$t = t_0$, with entrainment")

    axislegend(ax; position = :rb, patchsize = (40, 20))
    hidexdecorations!(ax, label=false)
    hideydecorations!(ax, label=false, ticks=false, ticklabels=false)

    display(fig)
    # save("figures/entrainment_schematic.pdf", fig)
end