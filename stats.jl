#
using Plots, CSV

fbench1 = "results/bench1_out.csv"
fbench2 = "results/bench2_out.csv"
fbench3 = "results/bench3_out.csv"
fbench6 = "results/bench6_out.csv"

fb1 = CSV.File(fbench1)
fb2 = CSV.File(fbench2)
fb3 = CSV.File(fbench3)
fb6 = CSV.File(fbench6)

fb1.time
fb1.total_free_energy

###############
# BENCH 1, 2, 6
###############

title_E = "Total Free Energy vs Time"
title_C = "Total Solute vs Time (Normalized)"

kw_E = (;title=title_E, width=2.0, color=:black, label=nothing, xaxis=:log,)
kw_C = (;title=title_C, width=2.0, color=:black, label=nothing, xaxis=:log,
        ylim=(0,1.01)
       )

pb1_E = plot(fb1.time, fb1.total_free_energy; kw_E...)
pb2_E = plot(fb2.time, fb2.total_free_energy; kw_E...)
pb6_E = plot(fb6.time, fb6.total_free_energy; kw_E...)

png(pb1_E, "results/bench1_E")
png(pb2_E, "results/bench2_E")
png(pb6_E, "results/bench6_E")

pb1_C = plot(fb1.time, fb1.total_solute / fb1.total_solute[begin]; kw_C...)
pb2_C = plot(fb2.time, fb2.total_solute / fb2.total_solute[begin]; kw_C...)
pb6_C = plot(fb6.time, fb6.total_solute / fb6.total_solute[begin]; kw_C...)

png(pb1_C, "results/bench1_C")
png(pb2_C, "results/bench2_C")
png(pb6_C, "results/bench6_C")

###############
# BENCH 3
###############

title_S = "Solid Fraction vs Time"
kw_E = (;title=title_E, width=2.0, color=:black, label=nothing)
kw_S = (;title=title_S, width=2.0, color=:black, label=nothing)

pb3_E = plot(fb3.time, fb3.total_free_energy; kw_E...)
pb3_S = plot(fb3.time, fb3.solid_fraction; kw_S...)

png(pb3_E, "results/bench3_E")
png(pb3_S, "results/bench3_S")
#
