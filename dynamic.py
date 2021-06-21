import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt




def energy_loss(beta, x):
    return (1 - beta) * x


def inv_energy_loss(beta, x):
    return x / (1 - beta)


def round_down(x,h):
    return x - (x % h)


def round_up(x,h):
    return x - (x % h) + h


start = 180
end = 181

df = pd.read_excel('Day_Ahead_Auktion_2018.xlsx', sheet_name=1, engine="openpyxl")
m = (end-start)*24
p = np.zeros(m)
for i in range(start,end):
    for j in range(1,25):
        p[i-start+j-1] = df['Unnamed: {}'.format(j)][i]
Z = 200*np.ones(m)
Vinit = 100
Vfinal = 100
eta = 1
V = np.zeros(m)
l = 0
u = 1000
c = 0
C = 500
beta = 0.1
hV=1                                                        
hx=100
z = np.inf*np.ones((int((C-c)/hV) + 1,m)) 
x = np.zeros((int((C-c)/hV) + 1,m))
policy = np.zeros(m)

tic = time()


V[0] = np.ceil(energy_loss(beta, Vinit)) - Z[0]
cmin = V[0] + l*eta
while cmin < c:
    cmin += hx*eta
cmax = V[0] + u*eta
while cmax > C:
    cmax -= hx*eta
    

for d in range(c,C+hV,hV):
    if cmin<=d and d<=cmax:
        if (d-V[0])/eta % hx == 0:
            e = (d-V[0])/eta
            z[int((d-c)/hV)][0] = e*p[0]
            x[int((d-c)/hV)][0] = e

oldmin = cmin
oldmax = cmax

for j in range(1, m):

    cmin = np.ceil(energy_loss(beta, cmin)) - Z[j] + l*eta
    while cmin < c:
        cmin += hx*eta

    cmax = np.ceil(energy_loss(beta, cmax)) - Z[j] + u*eta
    while cmax > C:
        cmax -= hx*eta
        
    for d in range(c,C+hV,hV):
        if cmin <= d and d <= cmax:
            V[j] = -Z[j]
            e = (d-V[j])/eta
            for k in range(l,u+hx,hx):
                before = round_down(inv_energy_loss(beta, (e-k)), hV)
                if oldmin <= before and before <= oldmax:
                    if z[int((before-c)/hV)][j-1] + p[j]*k < z[int((d-c)/hV)][j]:
                        z[int((d-c)/hV)][j] = z[int((before-c)/hV)][j-1] + p[j]*k
                        x[int((d-c)/hV)][j] = k
    oldmin = cmin
    oldmax = cmax


final_z = z[:, m-1]
final = final_z[int((Vfinal-c)/hV):]
d = int((Vfinal-c)/hV + np.argmin(final)*hV)


V[m-1] = d

policy[m-1] = x[int((d-c)/hV)][m-1]
for i in range(m-2,-1,-1):
    d = round_down(inv_energy_loss(beta, d - eta*x[int((d-c)/hV)][i+1] + Z[i+1]), hV)
    if d < c or d > C:
        print("fill level not feasible :(")
        break
    policy[i] = x[int((d-c)/hV)][i]
    V[i] = d
    
toc = time()

print("Time elapsed: ", toc - tic)


V[0] = np.ceil(energy_loss(beta, Vinit)) + policy[0] - Z[0]
for i in range(1,m):
    V[i] = np.ceil(energy_loss(beta, V[i-1])) + policy[i] - Z[i]
print(V)
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

