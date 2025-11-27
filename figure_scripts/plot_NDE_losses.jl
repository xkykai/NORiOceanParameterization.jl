using JLD2
using CairoMakie
using Makie

colors = Makie.wong_colors();
FILE_DIR = joinpath(@__DIR__, "..", "figure_data", "losses")

#####
##### Load Training Losses
#####

filename_loss = "NN_losses.jld2"
ind_losses_all, losses_all, loss_all, loss_prefactors = jldopen(joinpath(FILE_DIR, filename_loss)) do file
    return (
        file["ind_losses"],
        file["prefactored_losses"],
        file["prefactored_loss"],
        file["loss_prefactors"]
    )
end

#####
##### Load Validation Losses
#####

filename_loss_validation = "NN_losses_validation.jld2"
ind_losses_all_validation, losses_all_validation, loss_all_validation, loss_prefactors_validation = jldopen(joinpath(FILE_DIR, filename_loss_validation)) do file
    return (
        file["ind_losses"],
        file["prefactored_losses"],
        file["prefactored_loss"],
        file["loss_prefactors"]
    )
end

#####
##### Load Stage-wise Training Losses (different integration times)
#####

filename_loss_stages = "NN_losses_stages.jld2"
ind_losses_all_stages, losses_all_stages, loss_all_stages, loss_prefactors_stages = jldopen(joinpath(FILE_DIR, filename_loss_stages)) do file
    return (
        file["ind_losses"],
        file["prefactored_losses"],
        file["prefactored_loss"],
        file["loss_prefactors"]
    )
end

#####
##### Load Stage-wise Validation Losses
#####

filename_losses_stages_validation = "NN_losses_stages_validation.jld2"
ind_losses_all_stages_validation, losses_all_stages_validation, loss_all_stages_validation, loss_prefactors_stages_validation = jldopen(joinpath(FILE_DIR, filename_losses_stages_validation)) do file
    return (
        file["ind_losses"],
        file["prefactored_losses"],
        file["prefactored_loss"],
        file["loss_prefactors"]
    )
end

#####
##### Prepare Data for Plotting
#####

epochs = 1:length(loss_all)

# Convert loss vectors to matrices (each row = one case/simulation)
losses_all = hcat(losses_all...)
losses_all_validation = hcat(losses_all_validation...)

# Training stages with different integration times
epoch_stages = [0:2000, 2001:4001, 4002:6002]
losses_all_stages = [hcat(losses...) for losses in losses_all_stages]
losses_all_stages_validation = [hcat(losses...) for losses in losses_all_stages_validation]

#####
##### Create Figure
#####

ylims = (10^-3.2, 10^-2)

with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1400, 1200), fontsize=25)
    
    # Layout: top row (training/validation), bottom row (3 stages)
    g1 = fig[1, 1] = GridLayout()
    g2 = fig[2, 1] = GridLayout()
    
    # Top row: overall training and validation losses
    axtrain = Axis(g1[1, 1], xlabel="Epoch", ylabel="Loss", yscale=log10, title="Training")
    axval = Axis(g1[1, 2], xlabel="Epoch", ylabel="Loss", yscale=log10, title="Validation")
    
    # Bottom row: losses for different integration times
    ax1 = Axis(g2[1, 1], xlabel="Epoch", ylabel="Loss", yscale=log10, title="15-hour integration")
    ax2 = Axis(g2[1, 2], xlabel="Epoch", ylabel="Loss", yscale=log10, title="23.3-hour integration")
    ax3 = Axis(g2[1, 3], xlabel="Epoch", ylabel="Loss", yscale=log10, title="43.3-hour integration")

    # Plot individual case losses (grey) and mean loss (colored)
    lines!(axtrain, epochs, losses_all[1, :], color=(:darkgrey, 0.5), linewidth=3, label="Individual losses")
    for loss in eachrow(losses_all)[2:end]
        lines!(axtrain, epochs, loss, color=(:darkgrey, 0.5), linewidth=3)
    end
    lines!(axtrain, epochs, loss_all, color=colors[1], label="Mean loss", linewidth=4)

    lines!(axval, epochs, losses_all_validation[1, :], color=(:darkgrey, 0.5), linewidth=3, label="Individual losses")
    for loss in eachrow(losses_all_validation)[2:end]
        lines!(axval, epochs, loss, color=(:darkgrey, 0.5), linewidth=3)
    end
    lines!(axval, epochs, loss_all_validation, color=colors[2], label="Mean loss", linewidth=4)

    # Mark stage transitions
    vlines!(axtrain, [2001, 4002], color=:black, linestyle=:dash, linewidth=3)
    vlines!(axval, [2001, 4002], color=:black, linestyle=:dash, linewidth=3)

    # Link axes and set limits
    linkyaxes!(axtrain, axval)
    linkxaxes!(axtrain, axval)
    ylims!(axtrain, ylims)
    ylims!(axval, ylims)

    hidexdecorations!(axtrain, ticks=false, ticklabels=false, label=false)
    hideydecorations!(axtrain, ticks=false, ticklabels=false, label=false)
    hidexdecorations!(axval, ticks=false, ticklabels=false, label=false)
    hideydecorations!(axval, ticks=false)

    axislegend(axtrain, position=:rt)
    axislegend(axval, position=:rt)

    # Plot stage-wise losses
    for (i, ax) in enumerate([ax1, ax2, ax3])
        lines!(ax, epoch_stages[i], loss_all_stages[i], color=colors[1], linewidth=4, label="Mean training loss")
        lines!(ax, epoch_stages[i], loss_all_stages_validation[i], color=colors[2], linewidth=4, label="Mean validation loss")
    end

    hidedecorations!(ax1, ticks=false, ticklabels=false, label=false)
    hideydecorations!(ax2, ticks=false, ticklabels=false)
    hideydecorations!(ax3, ticks=false, ticklabels=false)
    hidexdecorations!(ax2, ticks=false, ticklabels=false, label=false)
    hidexdecorations!(ax3, ticks=false, ticklabels=false, label=false)

    axislegend(ax3, position=:rt)

    display(fig)
    # save("./figures/NDE_losses.svg", fig)
end