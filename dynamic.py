import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt




def energy_loss(beta, x):
    return (1 - beta) * x


def inv_energy_loss(beta, x):
    return x / (1 - beta)


def round_down(x,h):
    return int(x - (x % h))


def round_up(x,h):
    if x % h == 0:
        return int(x)
    else:
        return int(x - (x % h) + h)


start = 0
end = 1

df = pd.read_excel('Day_Ahead_Auktion_2018.xlsx', sheet_name=1, engine="openpyxl")
m = (end-start)*24
p = np.zeros(m)
for i in range(start,end):
    for j in range(1,25):
        p[i-start+j-1] = df['Unnamed: {}'.format(j)][i]

Z = 200*np.ones(m)
Vinit = 100
Vfinal = 100
eta = 0.8
V = np.zeros(m)
l = 0
u = 1000

beta = 0.1
hV=1
hx=100

error = sum((1-beta)**(m-i) for i in range(1, m+1))*hV

c = 0
C = round_down(500 - error, hV)

if C < c:
    print('The step width hV is too big. The upper bound on the error is larger than the storage. Hence, feasibility of the rounded solution cannot be guaranteed. Try again with a finer discretization.')
    exit()

z = np.inf*np.ones((int((C-c)/hV) + 1,m)) 
x = np.zeros((int((C-c)/hV) + 1,m))
A = np.inf*np.ones((int((C-c)/hV) + 1,m)) 

policy = np.zeros(m)

tic = time()

# TODO
# Lade/Entladeverlust berücksichtigen bzw Strom direkt verbrauchen
# Immer lieber verbauchen als einspeichern


for d in range(c,C+hV,hV):
    if (d - energy_loss(beta, Vinit) + Z[0])/eta % hx == 0:
        e = (d - energy_loss(beta, Vinit) + Z[0])/eta
        z[int((d-c)/hV)][0] = e*p[0]
        x[int((d-c)/hV)][0] = e


for j in range(1, m):
    for d in range(c,C+hV,hV):
        e = d+Z[j]
        for k in range(l,u+hx,hx):
            lb = round_up(inv_energy_loss(beta, e-eta*k ), hV)
            if inv_energy_loss(beta, e-(eta*k)+hV) % hV == 0:
                ub = round_down(inv_energy_loss(beta, e-(eta*k)+hV), hV)
            else:
                ub = round_down(inv_energy_loss(beta, e-(eta*k)+hV), hV) + 1
            for before in range(lb, ub):
                if c <= before and before <= C:
                    if z[int((before-c)/hV)][j-1] + p[j]*k < z[int((d-c)/hV)][j]:
                        z[int((d-c)/hV)][j] = z[int((before-c)/hV)][j-1] + p[j]*k
                        x[int((d-c)/hV)][j] = k
                        A[int((d-c)/hV)][j] = before

final_z = z[:, m-1]
final = final_z[int((Vfinal-c)/hV):]

dfinal = ((Vfinal-c)/hV + np.argmin(final))*hV

feasible = False

while not feasible:
    d = int(dfinal)
    V[m-1] = d
    policy[m-1] = x[int((d-c)/hV)][m-1]
    for i in range(m-2,-1,-1):
        d = A[int((d-c)/hV)][i+1]
        if d < c or d > C:
            print("no feasible fill level in time step ", i, " for final level ", dfinal)
            feasible = False
            dfinal += hV
            if dfinal > C:
                exit()
            break
        else:
            feasible = True
        policy[i] = x[int((d-c)/hV)][i]
        V[i] = d

    
toc = time()

print("Time elapsed: ", toc - tic)

V_real = np.zeros(m)
V_real[0] = energy_loss(beta, Vinit) + eta*policy[0] - Z[0]

for i in range(1,m):
    V_real[i] = energy_loss(beta, V_real[i-1]) + eta*policy[i] - Z[i]
#print(policy)
#print(V)
print(np.min(V_real), np.max(V_real))
print(policy@p/100)

plt.step(np.arange(m+1), np.concatenate(([0], policy)), where='post', color='r', linestyle='--', linewidth=2)
plt.step(np.arange(m+1), np.concatenate(([Vinit], V)), where='post', color='k', linestyle=':', linewidth=2)

plt.plot([0, 24], [l, l], color='r', linestyle='--', linewidth=1)
plt.plot([0, 24], [u, u], color='r', linestyle='--', linewidth=1)
plt.plot([0, 24], [c, c], color='k', linestyle=':', linewidth=1)
plt.plot([0, 24], [C, C], color='k', linestyle=':', linewidth=1)

plt.xticks([0,2,4,6,8,10,12,14,16,18,20,22,24])
legend = ['energy','fill level']
plt.legend(legend, loc=1)
plt.show()

