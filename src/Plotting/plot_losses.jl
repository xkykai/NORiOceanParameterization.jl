using NORiOceanParameterization.TrainingOperations: BaseClosureMode, NNMode
using CairoMakie
using Colors

"""
    plot_loss(::BaseClosureMode, losses, output_dir; epoch=1)

Plot training loss history for base closure parameterization training.

Creates a log-scale plot of total loss vs iterations and saves to PNG.

# Arguments
- `::BaseClosureMode`: Training mode indicator
- `losses`: NamedTuple containing `total` loss array
- `output_dir`: Directory path for saving the plot
- `epoch=1`: Epoch number for filename

# Output
Saves plot to `\$(output_dir)/losses_epoch\$(epoch).png`
"""
function plot_loss(::BaseClosureMode, losses, output_dir; epoch=1)
    colors = distinguishable_colors(10, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    
    fig = Figure(size=(1000, 600))
    axtotalloss = CairoMakie.Axis(fig[1, 1], title="Total Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)

    lines!(axtotalloss, losses.total, label="Total Loss", color=colors[1])

    axislegend(axtotalloss, position=:lb)
    save("$(output_dir)/losses_epoch$(epoch).png", fig, px_per_unit=8)
end