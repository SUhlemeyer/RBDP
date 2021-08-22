# Install Packages
# using Pkg
# Pkg.add("JuMP")
# Pkg.add("Cbc")
# Pkg.add("XLSX")


using JuMP
using Cbc
import XLSX
import CSV
import DataFrames


Vinit = 100                                     # Initial fill level
Vfinal = 100                                    # Lower bound for final fill level
eta_in = 0.9                                    # Efficiency factor in
eta_out = 0.95                                  # Efficiency factor out
beta = 0.1                                      # Energy loss factor
hV=1                                            # Step size storage
hx=100                                          # Step size purchased energy
l = 0                                           # Lower bound purchased energy
u = 1000                                        # Upper bound purchased energy
extern = 200                                    # Constant energy consumption
cmin = 0                                        # Lower bound storage
cmax = 5000                                     # Upper bound storage


dat = XLSX.readxlsx("Day_Ahead_Auktion_2018.xlsx")
df = dat["Day-Ahead_Auktion_Preise"]

start = 180
ende = 181

m = (ende-start)*24
p = zeros(m)
Z = extern * ones(m)

for i in start:ende-1
    for j in 2:25
        p[(i-start)*24+j-1] = df[i,j]
    end
end

println(p)



# Create a model
storage = Model()

# Create our variables
# x =  x/hx
l=l/hx
u=u/hx
println("l: ", cmin, "\t u: ", cmax)
# variables:
#   x * hx : purchased energy discretized in steps of size hx
#   y : share of purchased energy used for charging
#   zeta : share of purchased energy  used for consumption
#   v : energy in storage

@variable(storage, x[i=1:length(p)], Int, lower_bound=l, upper_bound=u);
@variable(storage, v[i=0:length(p)], lower_bound=cmin, upper_bound=cmax)
@variable(storage, y[i=1:length(p)], lower_bound=0)
@variable(storage, zeta[i=1:length(p)], lower_bound=0)

@constraint(storage, initial, v[0] == Vinit )
@constraint(storage, final, v[end] >= Vfinal )
@constraint(storage, update[i = 1:length(x)], v[i] == (1-beta) * v[i-1] + eta_in * y[i] -  zeta[i] )
@constraint(storage, input[i =1:length(x)], x[i]*hx >= y[i] )
@constraint(storage, consuming[i =1:length(x)], x[i]*hx - y[i] + eta_out *zeta[i] == Z[i] )

@objective(storage, Min, sum(p[i]*x[i] for i in 1:length(x)) )



set_optimizer(storage, Cbc.Optimizer)
set_optimizer_attribute(storage, "logLevel", 1)

optimize!(storage)


termination_status(storage)


# Display the solution
println("v: ", value.(v))
println("x: ", value.(x))
println("y: ", value.(y))
println("zeta: ", value.(zeta))
