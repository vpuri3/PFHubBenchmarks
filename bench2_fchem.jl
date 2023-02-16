#
using LazyGrids, Plots

r = range(0, 1, 1000)
c, η = ndgrid(r, r)

h = @. η^3 * (6η^2 - 15η + 10)
g = @. η^2 * (1 - η)^2

ρ = √2.0
calpha = 0.3
cbeta  = 0.7
w = 1.0

falpha = @. ρ^2 * (c - calpha)^2
fbeta  = @. ρ^2 * (c - cbeta)^2

fchem = @. falpha * (1 - h) + fbeta * h + w * g
#fchem = @. falpha * (1 - η) + fbeta * η + w * g
fchem = @. falpha * (1 - η) + fbeta * η
p=contourf(r, r, fchem', xlabel="concentration (c)", ylabel="order parameter (η)", title="F_chem")
png(p, "bench2_fchem")
