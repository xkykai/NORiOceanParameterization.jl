using Statistics

"""
    AbstractFeatureScaling

Abstract base type for all feature scaling methods.

Feature scaling is a method used to normalize or standardize the range of independent 
variables or features of data. Subtypes implement specific scaling strategies such as 
zero-mean unit-variance scaling or min-max scaling.

# Subtypes
- [`ZeroMeanUnitVarianceScaling`](@ref): Standardizes data to have zero mean and unit variance
- [`MinMaxScaling`](@ref): Scales data to a specified range [a, b]
- [`DiffusivityScaling`](@ref): Custom scaling for diffusivity parameters

# Interface
All concrete subtypes should implement:
- `scale(x, s::AbstractFeatureScaling)`: Transform data using the scaling
- `unscale(y, s::AbstractFeatureScaling)`: Inverse transform scaled data
"""
abstract type AbstractFeatureScaling end

#####
##### Zero-mean unit-variance feature scaling
#####

"""
    ZeroMeanUnitVarianceScaling{T} <: AbstractFeatureScaling

Feature scaler that standardizes data to have zero mean and unit variance.

This scaling method transforms data according to the formula:
```
y = (x - μ) / σ
```
where μ is the mean and σ is the standard deviation of the data.

# Fields
- `μ :: T`: Mean of the original data
- `σ :: T`: Standard deviation of the original data

# Examples
```julia
julia> data = [1.0, 2.0, 3.0, 4.0, 5.0];

julia> scaler = ZeroMeanUnitVarianceScaling(data);

julia> scaled = scale(data, scaler)
5-element Vector{Float64}:
 -1.414...
 -0.707...
  0.0
  0.707...
  1.414...

julia> unscale(scaled, scaler) ≈ data
true
```

# See Also
- [`MinMaxScaling`](@ref): Alternative scaling to a specified range
- [`scale`](@ref): Apply the scaling transformation
- [`unscale`](@ref): Invert the scaling transformation
"""
struct ZeroMeanUnitVarianceScaling{T} <: AbstractFeatureScaling
    μ :: T
    σ :: T
end

"""
    ZeroMeanUnitVarianceScaling(data)

Construct a zero-mean unit-variance scaler from data.

Computes the mean (μ) and standard deviation (σ) of the input data and creates
a scaler that will transform data to have zero mean and unit variance.

# Arguments
- `data`: Input data array from which to compute scaling parameters

# Returns
- `ZeroMeanUnitVarianceScaling`: Scaler with computed mean and standard deviation

# Examples
```julia
julia> data = [10.0, 20.0, 30.0, 40.0, 50.0];

julia> scaler = ZeroMeanUnitVarianceScaling(data);

julia> scaler.μ
30.0

julia> scaler.σ
15.811...

julia> scaled_data = scaler(data);  # Using callable interface

julia> mean(scaled_data) ≈ 0.0
true

julia> std(scaled_data) ≈ 1.0
true
```

# Notes
- The standard deviation is computed using the corrected sample standard deviation (n-1 denominator)
- For data with zero variance, the scaling may produce NaN or Inf values
"""
function ZeroMeanUnitVarianceScaling(data)
    μ, σ = mean(data), std(data)
    return ZeroMeanUnitVarianceScaling(μ, σ)
end

"""
    scale(x, s::ZeroMeanUnitVarianceScaling)

Apply zero-mean unit-variance scaling to data.

Transforms the input data `x` using the mean and standard deviation stored in the scaler `s`.

# Arguments
- `x`: Data to be scaled
- `s::ZeroMeanUnitVarianceScaling`: Scaler containing mean (μ) and standard deviation (σ)

# Returns
- Scaled data: `(x - μ) / σ`

# See Also
- [`unscale`](@ref): Inverse transformation
"""
scale(x, s::ZeroMeanUnitVarianceScaling) = (x .- s.μ) / s.σ

"""
    unscale(y, s::ZeroMeanUnitVarianceScaling)

Invert zero-mean unit-variance scaling.

Transforms scaled data `y` back to the original scale using the mean and standard deviation 
stored in the scaler `s`.

# Arguments
- `y`: Scaled data to be transformed back
- `s::ZeroMeanUnitVarianceScaling`: Scaler containing mean (μ) and standard deviation (σ)

# Returns
- Unscaled data: `σ * y + μ`

# See Also
- [`scale`](@ref): Forward transformation
"""
unscale(y, s::ZeroMeanUnitVarianceScaling) = s.σ * y .+ s.μ

#####
##### Min-max feature scaling
#####

"""
    MinMaxScaling{T} <: AbstractFeatureScaling

Feature scaler that transforms data to a specified range [a, b].

This scaling method transforms data according to the formula:
```
y = a + (x - x_min) * (b - a) / (x_max - x_min)
```

# Fields
- `a :: T`: Minimum value of the target range
- `b :: T`: Maximum value of the target range
- `data_min :: T`: Minimum value in the original data
- `data_max :: T`: Maximum value in the original data

# Examples
```julia
julia> data = [1.0, 2.0, 3.0, 4.0, 5.0];

julia> scaler = MinMaxScaling(data; a=0, b=1);

julia> scaled = scale(data, scaler)
5-element Vector{Float64}:
 0.0
 0.25
 0.5
 0.75
 1.0
```

# See Also
- [`ZeroMeanUnitVarianceScaling`](@ref): Alternative standardization method
- [`scale`](@ref): Apply the scaling transformation
- [`unscale`](@ref): Invert the scaling transformation
"""
struct MinMaxScaling{T} <: AbstractFeatureScaling
           a :: T
           b :: T
    data_min :: T
    data_max :: T
end

"""
    MinMaxScaling(data; a=0, b=1)

Construct a min-max scaler from data.

Computes the minimum and maximum values of the input data and creates a scaler
that will transform data to the range [a, b].

# Arguments
- `data`: Input data array from which to compute scaling parameters

# Keyword Arguments
- `a=0`: Minimum value of the target range (default: 0)
- `b=1`: Maximum value of the target range (default: 1)

# Returns
- `MinMaxScaling`: Scaler with computed min, max, and target range

# Examples
```julia
julia> data = [10.0, 20.0, 30.0, 40.0, 50.0];

julia> scaler = MinMaxScaling(data; a=0, b=1);

julia> scaler.data_min
10.0

julia> scaler.data_max
50.0

julia> scaled_data = scaler(data);

julia> extrema(scaled_data)
(0.0, 1.0)

# Scale to a different range
julia> scaler2 = MinMaxScaling(data; a=-1, b=1);

julia> scaled_data2 = scaler2(data);

julia> extrema(scaled_data2)
(-1.0, 1.0)
```

# Notes
- The scaler preserves the relative distances between data points
- For data with all identical values (data_min == data_max), scaling may produce NaN or Inf
"""
function MinMaxScaling(data; a=0, b=1)
    data_min, data_max = extrema(data)
    return MinMaxScaling{typeof(data_min)}(a, b, data_min, data_max)
end

"""
    scale(x, s::MinMaxScaling)

Apply min-max scaling to data.

Transforms the input data `x` to the range [a, b] using the min and max values stored in the scaler `s`.

# Arguments
- `x`: Data to be scaled
- `s::MinMaxScaling`: Scaler containing target range [a, b] and data range [data_min, data_max]

# Returns
- Scaled data: `a + (x - data_min) * (b - a) / (data_max - data_min)`

# See Also
- [`unscale`](@ref): Inverse transformation
"""
scale(x, s::MinMaxScaling) = s.a + (x - s.data_min) * (s.b - s.a) / (s.data_max - s.data_min)

"""
    unscale(y, s::MinMaxScaling)

Invert min-max scaling.

Transforms scaled data `y` back to the original scale using the parameters stored in the scaler `s`.

# Arguments
- `y`: Scaled data to be transformed back
- `s::MinMaxScaling`: Scaler containing target range [a, b] and data range [data_min, data_max]

# Returns
- Unscaled data: `data_min + (y - a) * (data_max - data_min) / (b - a)`

# See Also
- [`scale`](@ref): Forward transformation
"""
unscale(y, s::MinMaxScaling) = s.data_min .+ (y .- s.a) * (s.data_max - s.data_min) / (s.b - s.a)

#####
##### Convenience functions
#####

"""
    (s::AbstractFeatureScaling)(x)

Make scalers callable for convenient scaling.

Allows using a scaler as a function to apply the scaling transformation.

# Examples
```julia
julia> data = [1.0, 2.0, 3.0, 4.0, 5.0];

julia> scaler = ZeroMeanUnitVarianceScaling(data);

julia> scaled = scaler(data);  # Equivalent to scale(data, scaler)
```
"""
(s::AbstractFeatureScaling)(x) = scale(x, s)

"""
    Base.inv(s::AbstractFeatureScaling)

Return the inverse scaling function.

Returns a function that performs the inverse transformation (unscaling).

# Examples
```julia
julia> data = [1.0, 2.0, 3.0, 4.0, 5.0];

julia> scaler = MinMaxScaling(data);

julia> scaled = scaler(data);

julia> inv_scaler = inv(scaler);

julia> original = inv_scaler(scaled);

julia> original ≈ data
true
```
"""
Base.inv(s::AbstractFeatureScaling) = y -> unscale(y, s)

"""
    DiffusivityScaling{T} <: AbstractFeatureScaling

Custom feature scaler for diffusivity parameters (ν, κ).

This scaler applies affine transformations to viscosity (ν) and diffusivity (κ) parameters:
```
ν_scaled = ν₀ + ν * ν₁
κ_scaled = κ₀ + κ * κ₁
```

# Fields
- `ν₀ :: T`: Offset for viscosity scaling
- `κ₀ :: T`: Offset for diffusivity scaling
- `ν₁ :: T`: Multiplier for viscosity scaling
- `κ₁ :: T`: Multiplier for diffusivity scaling

# Examples
```julia
julia> scaler = DiffusivityScaling(1e-5, 1e-5, 0.1, 0.1);

julia> ν, κ = 0.5, 0.3;

julia> ν_scaled, κ_scaled = scaler((ν, κ))
(0.05001, 0.03001)

julia> inv(scaler)((ν_scaled, κ_scaled))
(0.5, 0.3)
```

# See Also
- [`scale(::Tuple, ::DiffusivityScaling)`](@ref)
- [`unscale(::Tuple, ::DiffusivityScaling)`](@ref)
"""
struct DiffusivityScaling{T} <: AbstractFeatureScaling
    ν₀ :: T
    κ₀ :: T
    ν₁ :: T
    κ₁ :: T
end

"""
    DiffusivityScaling(ν₀=1e-5, κ₀=1e-5, ν₁=0.1, κ₁=0.1)

Construct a diffusivity parameter scaler.

Creates a scaler for viscosity (ν) and diffusivity (κ) parameters with specified
offset and multiplier values.

# Arguments
- `ν₀=1e-5`: Offset for viscosity scaling (default: 1e-5)
- `κ₀=1e-5`: Offset for diffusivity scaling (default: 1e-5)
- `ν₁=0.1`: Multiplier for viscosity scaling (default: 0.1)
- `κ₁=0.1`: Multiplier for diffusivity scaling (default: 0.1)

# Returns
- `DiffusivityScaling`: Scaler for diffusivity parameters

# Examples
```julia
julia> scaler = DiffusivityScaling();

julia> scaler.ν₀
1.0e-5

julia> custom_scaler = DiffusivityScaling(1e-4, 1e-4, 0.2, 0.2);
```
"""
function DiffusivityScaling(ν₀=1e-5, κ₀=1e-5, ν₁=0.1, κ₁=0.1)
    return DiffusivityScaling(ν₀, κ₀, ν₁, κ₁)
end

"""
    scale(x, s::DiffusivityScaling)

Apply diffusivity scaling to (ν, κ) parameters.

# Arguments
- `x`: Tuple of (ν, κ) values to be scaled
- `s::DiffusivityScaling`: Scaler containing offset and multiplier parameters

# Returns
- Tuple of scaled values: `(ν₀ + ν * ν₁, κ₀ + κ * κ₁)`

# Examples
```julia
julia> scaler = DiffusivityScaling(0.0, 0.0, 1.0, 1.0);

julia> scale((0.5, 0.3), scaler)
(0.5, 0.3)
```
"""
function scale(x, s::DiffusivityScaling)
    ν, κ = x
    ν₀, κ₀, ν₁, κ₁ = s.ν₀, s.κ₀, s.ν₁, s.κ₁
    return ν₀ + ν * ν₁, κ₀ + κ * κ₁
end

"""
    unscale(y, s::DiffusivityScaling)

Invert diffusivity scaling for (ν, κ) parameters.

# Arguments
- `y`: Tuple of scaled (ν, κ) values to be transformed back
- `s::DiffusivityScaling`: Scaler containing offset and multiplier parameters

# Returns
- Tuple of unscaled values: `((ν - ν₀) / ν₁, (κ - κ₀) / κ₁)`

# Examples
```julia
julia> scaler = DiffusivityScaling(1e-5, 1e-5, 0.1, 0.1);

julia> scaled = (0.05001, 0.03001);

julia> unscale(scaled, scaler)
(0.5, 0.3)
```
"""
function unscale(y, s::DiffusivityScaling)
    ν, κ = y
    ν₀, κ₀, ν₁, κ₁ = s.ν₀, s.κ₀, s.ν₁, s.κ₁
    return (ν - ν₀) / ν₁, (κ - κ₀) / κ₁
end

(s::DiffusivityScaling)(x) = scale(x, s)
Base.inv(s::DiffusivityScaling) = y -> unscale(y, s)

"""
    write_scaling_params(scaling)

Convert scaling parameters to a nested NamedTuple for serialization.

Extracts all field values from each scaler in the input NamedTuple and returns them
in a format suitable for saving to disk (e.g., JSON, JLD2).

# Arguments
- `scaling`: NamedTuple of scalers (e.g., `(u=scaler1, v=scaler2, ...)`)

# Returns
- Nested NamedTuple with scaling parameters for each field

# Examples
```julia
julia> data_u = [1.0, 2.0, 3.0];
julia> data_v = [10.0, 20.0, 30.0];

julia> scaling = (u=ZeroMeanUnitVarianceScaling(data_u), 
                  v=ZeroMeanUnitVarianceScaling(data_v));

julia> params = write_scaling_params(scaling);

julia> params.u
(μ = 2.0, σ = 1.0)

julia> params.v
(μ = 20.0, σ = 10.0)
```

# See Also
- [`construct_zeromeanunitvariance_scaling`](@ref): Reconstruct scalers from saved parameters
"""
function write_scaling_params(scaling)
    return NamedTuple(key=>NamedTuple(param=>getproperty(scaling[key], param) for param in fieldnames(typeof(scaling[key]))) for key in keys(scaling))
end

"""
    construct_zeromeanunitvariance_scaling(scaling_params)

Reconstruct ZeroMeanUnitVarianceScaling scalers from saved parameters.

Takes a nested NamedTuple of scaling parameters (as produced by `write_scaling_params`)
and reconstructs the corresponding `ZeroMeanUnitVarianceScaling` objects.

# Arguments
- `scaling_params`: Nested NamedTuple containing μ and σ for each field

# Returns
- NamedTuple of reconstructed `ZeroMeanUnitVarianceScaling` objects

# Examples
```julia
julia> params = (u=(μ=2.0, σ=1.0), v=(μ=20.0, σ=10.0));

julia> scaling = construct_zeromeanunitvariance_scaling(params);

julia> scaling.u
ZeroMeanUnitVarianceScaling{Float64}(2.0, 1.0)

julia> scaling.v
ZeroMeanUnitVarianceScaling{Float64}(20.0, 10.0)
```

# See Also
- [`write_scaling_params`](@ref): Save scaling parameters
- [`ZeroMeanUnitVarianceScaling`](@ref): The scaler type being reconstructed
"""
function construct_zeromeanunitvariance_scaling(scaling_params)
    return NamedTuple(key=>ZeroMeanUnitVarianceScaling(scaling_params[key].μ, scaling_params[key].σ) for key in keys(scaling_params))
end