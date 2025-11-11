using NORiOceanParameterization.TrainingOperations: BaseClosureMode, NNMode
using NORiOceanParameterization.Utils: find_min, find_max
using CairoMakie

"""
    animate_data(::BaseClosureMode, train_data, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN, index, output_dir; coarse_size=32, epoch=1, suffix="")

Create animation comparing base closure predictions to LES truth.

Generates a multi-panel animation showing profiles (u, v, T, S, ρ), fluxes (uw, vw, wT, wS),
Richardson number, and diffusivities over time. Compares model predictions against LES ground truth.

# Arguments
- `::BaseClosureMode`: Training mode indicator
- `train_data`: LES ground truth data
- `diffusivities`: Truth diffusivities and Richardson numbers
- `sols_noNN`: Model-predicted profiles (u, v, T, S, ρ)
- `fluxes_noNN`: Model-predicted fluxes (uw, vw, wT, wS)
- `diffusivities_noNN`: Model-predicted diffusivities (ν, κ, Ri)
- `index`: Simulation index for filename
- `output_dir`: Directory path for saving animation

# Keyword Arguments
- `coarse_size=32`: Vertical grid resolution
- `epoch=1`: Epoch number for filename
- `suffix=""`: Optional filename suffix

# Output
Saves animation to `\$(output_dir)/training_\$(index)_epoch\$(epoch)\$(suffix).mp4`

# Plot Layout
- Row 1: u, v, T, S, ρ profiles, Richardson number
- Row 2: uw, vw, wT, wS fluxes, diffusivities (ν, κ)
"""
function animate_data(::BaseClosureMode, train_data, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN, index, output_dir; coarse_size=32, epoch=1, suffix="")
    fig = Figure(size=(1920, 1080))
    axu = CairoMakie.Axis(fig[1, 1], title="u", xlabel="u (m s⁻¹)", ylabel="z (m)")
    axv = CairoMakie.Axis(fig[1, 2], title="v", xlabel="v (m s⁻¹)", ylabel="z (m)")
    axT = CairoMakie.Axis(fig[1, 3], title="T", xlabel="T (°C)", ylabel="z (m)")
    axS = CairoMakie.Axis(fig[1, 4], title="S", xlabel="S (g kg⁻¹)", ylabel="z (m)")
    axρ = CairoMakie.Axis(fig[1, 5], title="ρ", xlabel="ρ (kg m⁻³)", ylabel="z (m)")
    axuw = CairoMakie.Axis(fig[2, 1], title="uw", xlabel="uw (m² s⁻²)", ylabel="z (m)")
    axvw = CairoMakie.Axis(fig[2, 2], title="vw", xlabel="vw (m² s⁻²)", ylabel="z (m)")
    axwT = CairoMakie.Axis(fig[2, 3], title="wT", xlabel="wT (m s⁻¹ °C)", ylabel="z (m)")
    axwS = CairoMakie.Axis(fig[2, 4], title="wS", xlabel="wS (m s⁻¹ g kg⁻¹)", ylabel="z (m)")
    axRi = CairoMakie.Axis(fig[1, 6], title="Ri", xlabel="Ri", ylabel="z (m)")
    axdiffusivity = CairoMakie.Axis(fig[2, 5], title="Diffusivity", xlabel="Diffusivity (m² s⁻¹)", ylabel="z (m)", xscale=log10)

    n = Observable(1)
    zC = train_data.metadata["zC"]
    zF = train_data.metadata["zF"]

    u_noNN = sols_noNN.u
    v_noNN = sols_noNN.v
    T_noNN = sols_noNN.T
    S_noNN = sols_noNN.S
    ρ_noNN = sols_noNN.ρ

    uw_noNN = fluxes_noNN.uw.total
    vw_noNN = fluxes_noNN.vw.total
    wT_noNN = fluxes_noNN.wT.total
    wS_noNN = fluxes_noNN.wS.total

    ulim = (find_min(train_data.profile.u.unscaled, u_noNN, -1e-8), find_max(train_data.profile.u.unscaled, u_noNN, 1e-8))
    vlim = (find_min(train_data.profile.v.unscaled, v_noNN, -1e-8), find_max(train_data.profile.v.unscaled, v_noNN, 1e-8))
    Tlim = (find_min(train_data.profile.T.unscaled, T_noNN), find_max(train_data.profile.T.unscaled, T_noNN))
    Slim = (find_min(train_data.profile.S.unscaled, S_noNN), find_max(train_data.profile.S.unscaled, S_noNN))
    ρlim = (find_min(train_data.profile.ρ.unscaled, ρ_noNN), find_max(train_data.profile.ρ.unscaled, ρ_noNN))

    uwlim = (find_min(train_data.flux.uw.column.unscaled, uw_noNN, -1e-8),
             find_max(train_data.flux.uw.column.unscaled, uw_noNN, 1e-8))
    vwlim = (find_min(train_data.flux.vw.column.unscaled, vw_noNN, -1e-8),
             find_max(train_data.flux.vw.column.unscaled, vw_noNN, 1e-8))
    wTlim = (find_min(train_data.flux.wT.column.unscaled, wT_noNN, -1e-8),
             find_max(train_data.flux.wT.column.unscaled, wT_noNN, 1e-8))
    wSlim = (find_min(train_data.flux.wS.column.unscaled, wS_noNN, -1e-8),
             find_max(train_data.flux.wS.column.unscaled, wS_noNN, 1e-8))

    Rilim = (find_min(diffusivities.Ri_truth, diffusivities_noNN.Ri,), 
             find_max(diffusivities.Ri_truth, diffusivities_noNN.Ri,),)

    diffusivitylim = (find_min(diffusivities_noNN.ν, diffusivities_noNN.κ), 
                      find_max(diffusivities_noNN.ν, diffusivities_noNN.κ),)

    u_truthₙ = @lift train_data.profile.u.unscaled[:, $n]
    v_truthₙ = @lift train_data.profile.v.unscaled[:, $n]
    T_truthₙ = @lift train_data.profile.T.unscaled[:, $n]
    S_truthₙ = @lift train_data.profile.S.unscaled[:, $n]
    ρ_truthₙ = @lift train_data.profile.ρ.unscaled[:, $n]

    uw_truthₙ = @lift train_data.flux.uw.column.unscaled[:, $n]
    vw_truthₙ = @lift train_data.flux.vw.column.unscaled[:, $n]
    wT_truthₙ = @lift train_data.flux.wT.column.unscaled[:, $n]
    wS_truthₙ = @lift train_data.flux.wS.column.unscaled[:, $n]

    u_noNNₙ = @lift u_noNN[:, $n]
    v_noNNₙ = @lift v_noNN[:, $n]
    T_noNNₙ = @lift T_noNN[:, $n]
    S_noNNₙ = @lift S_noNN[:, $n]
    ρ_noNNₙ = @lift ρ_noNN[:, $n]

    uw_noNNₙ = @lift uw_noNN[:, $n]
    vw_noNNₙ = @lift vw_noNN[:, $n]
    wT_noNNₙ = @lift wT_noNN[:, $n]
    wS_noNNₙ = @lift wS_noNN[:, $n]

    Ri_truthₙ = @lift diffusivities.Ri_truth[:, $n]
    Ri_noNNₙ = @lift diffusivities_noNN.Ri[:, $n]

    ν_noNNₙ = @lift diffusivities_noNN.ν[:, $n]
    κ_noNNₙ = @lift diffusivities_noNN.κ[:, $n]

    Qᵀ = train_data.metadata["temperature_flux"]
    Qˢ = train_data.metadata["salinity_flux"]
    f = train_data.metadata["coriolis_parameter"]
    times = train_data.times
    Nt = length(times)

    time_str = @lift "Qᵀ = $(Qᵀ) m s⁻¹ °C, Qˢ = $(Qˢ) m s⁻¹ g kg⁻¹, f = $(f) s⁻¹, Time = $(round(times[$n]/24/60^2, digits=3)) days"

    lines!(axu, u_truthₙ, zC, label="Truth")
    lines!(axu, u_noNNₙ, zC, label="Base closure only")

    lines!(axv, v_truthₙ, zC, label="Truth")
    lines!(axv, v_noNNₙ, zC, label="Base closure only")

    lines!(axT, T_truthₙ, zC, label="Truth")
    lines!(axT, T_noNNₙ, zC, label="Base closure only")

    lines!(axS, S_truthₙ, zC, label="Truth")
    lines!(axS, S_noNNₙ, zC, label="Base closure only")

    lines!(axρ, ρ_truthₙ, zC, label="Truth")
    lines!(axρ, ρ_noNNₙ, zC, label="Base closure only")

    lines!(axuw, uw_truthₙ, zF, label="Truth")
    lines!(axuw, uw_noNNₙ, zF, label="Base closure only")

    lines!(axvw, vw_truthₙ, zF, label="Truth")
    lines!(axvw, vw_noNNₙ, zF, label="Base closure only")

    lines!(axwT, wT_truthₙ, zF, label="Truth")
    lines!(axwT, wT_noNNₙ, zF, label="Base closure only")

    lines!(axwS, wS_truthₙ, zF, label="Truth")
    lines!(axwS, wS_noNNₙ, zF, label="Base closure only")

    lines!(axRi, Ri_truthₙ, zF, label="Truth")
    lines!(axRi, Ri_noNNₙ, zF, label="Base closure only")

    lines!(axdiffusivity, ν_noNNₙ, zF, label="ν, Base closure only")
    lines!(axdiffusivity, κ_noNNₙ, zF, label="κ, Base closure only")

    axislegend(axu, position=:rb)
    axislegend(axdiffusivity, position=:rb)
    
    Label(fig[0, :], time_str, font=:bold, tellwidth=false)

    xlims!(axu, ulim)
    xlims!(axv, vlim)
    xlims!(axT, Tlim)
    xlims!(axS, Slim)
    xlims!(axρ, ρlim)

    xlims!(axuw, uwlim)
    xlims!(axvw, vwlim)
    xlims!(axwT, wTlim)
    xlims!(axwS, wSlim)
    xlims!(axRi, Rilim)
    xlims!(axdiffusivity, diffusivitylim)

    CairoMakie.record(fig, "$(output_dir)/training_$(index)_epoch$(epoch)$(suffix).mp4", 1:Nt, framerate=15) do nn
        n[] = nn
    end
end