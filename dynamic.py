import pandas as pd
import numpy as np
from utils import round_down, round_up, energy_loss, inv_energy_loss
from plot import step_functions

from sacred import Experiment
import sys
import logging



#TODO: Speichergrößen variieren
#TODO: Speichergröße gegen Kosten (Tradeoff-Analyse). 10-er Schritte bis 5000 (1 Monat)
#TODO: Figure1 ändern




ex = Experiment('dynamic_optimization')

log = logging.getLogger()
log.handlers = []

log_format = logging.Formatter('%(asctime)s || %(name)s - [%(levelname)s] - %(message)s')

streamhandler = logging.StreamHandler(sys.stdout)
streamhandler.setFormatter(log_format)
log.addHandler(streamhandler)

log.setLevel('INFO')
ex.logger = log


@ex.config
def config():
    args = dict(
        price_file = 'Day_Ahead_Auktion_2018.xlsx',         # Excel File with Day Ahead Market Prices
        start = 0,                                          # First day to optimize, included (in [0, 364])
        end = 3,                                            # Last day to optimize, excluded (in [1, 365])
        t = 24,                                             # How many time steps per day
        Vinit = 100,                                        # Initial fill level
        Vfinal = 100,                                       # Lower bound for final fill level
        eta_in = 0.9,                                       # Efficiency factor in
        eta_out = 0.95,                                     # Efficiency factor out
        beta = 0.1,                                         # Energy loss factor
        hV=1,                                               # Step size storage
        hx=100,                                             # Step size purchased energy
        l = 0,                                              # Lower bound purchased energy
        u = 1000,                                           # Upper bound purchased energy
        extern = 200,                                       # Constant energy consumption 
        cmin = 0,                                           # Lower bound storage
        cmax = 5000,                                        # Upper bound storage
    )



@ex.automain
def main(args):
    df = pd.read_excel(args['price_file'], sheet_name=1, engine="openpyxl")
    m = (args['end']-args['start'])*args['t']
    p = np.zeros(m)
    for i in range(args['start'],args['end']):
        for j in range(1,25):
            p[args['t']*(i-args['start'])+j-1] = df['Unnamed: {}'.format(j)][i]
            
    Z = args['extern']*np.ones(m)
    V = np.zeros(m)

    error = sum((1-args['beta'])**(m-i) for i in range(1, m+1))*args['hV']
    C = round_down(args['cmax'] - error, args['hV'])
    if C < args['cmin']:
        print('The step width hV is too big. The upper bound on the error is larger than the storage. Hence, feasibility of the rounded solution cannot be guaranteed. Try again with a finer discretization.')
        exit()

    z = np.inf*np.ones((int((C-args['cmin'])/args['hV']) + 1,m)) 
    x = np.zeros((int((C-args['cmin'])/args['hV']) + 1,m))
    A = np.inf*np.ones((int((C-args['cmin'])/args['hV']) + 1,m)) 

    policy = np.zeros(m)

    for k in range(args['l'],args['u']+args['hx'],args['hx']):
        
        zeta = max(Z[0] - k, 0)
        y = max(k - Z[0], 0)
        d = energy_loss(args['beta'], args['Vinit']) + args['eta_in']*y - (1/args['eta_out'])*zeta
            
        if args['cmin'] <= d and d <= C:
            z[int((d-args['cmin'])/args['hV'])][0] = k*p[0]
            x[int((d-args['cmin'])/args['hV'])][0] = k


    for j in range(1, m):
        for d in range(args['cmin'],C+args['hV'],args['hV']):
            e = d+Z[j]
            for k in range(args['l'],args['u']+args['hx'],args['hx']):
                
                zeta = max(Z[j] - k, 0)
                y = max(k - Z[j], 0)

                lb = round_up(inv_energy_loss(args['beta'], d - args['eta_in']*y + (1/args['eta_out'])*zeta ), args['hV'])
                
                if inv_energy_loss(args['beta'], d - args['eta_in']*y + (1/args['eta_out'])*zeta + args['hV']) % args['hV'] == 0:
                    ub = round_down(inv_energy_loss(args['beta'], d - args['eta_in']*y + (1/args['eta_out'])*zeta + args['hV']), args['hV'])
                else:
                    ub = round_down(inv_energy_loss(args['beta'], d - args['eta_in']*y + (1/args['eta_out'])*zeta + args['hV']), args['hV']) + 1
                for before in range(lb, ub):
                    if args['cmin'] <= before and before <= C:
                        if z[int((before-args['cmin'])/args['hV'])][j-1] + p[j]*k < z[int((d-args['cmin'])/args['hV'])][j]:
                            z[int((d-args['cmin'])/args['hV'])][j] = z[int((before-args['cmin'])/args['hV'])][j-1] + p[j]*k
                            x[int((d-args['cmin'])/args['hV'])][j] = k
                            A[int((d-args['cmin'])/args['hV'])][j] = before

    final_z = z[:, m-1]
    final = final_z[int((args['Vfinal']-args['cmin'])/args['hV']):]

    dfinal = ((args['Vfinal']-args['cmin'])/args['hV'] + np.argmin(final))*args['hV']
    d = int(dfinal)
    V[m-1] = d
    policy[m-1] = x[int((d-args['cmin'])/args['hV'])][m-1]

    for i in range(m-2,-1,-1):
        d = A[int((d-args['cmin'])/args['hV'])][i+1]
        if d < args['cmin'] or d > C:
            print("no feasible fill level in time step ", i, " for final level ", dfinal)
            break
        policy[i] = x[int((d-args['cmin'])/args['hV'])][i]
        V[i] = d

    V_real = np.zeros(m)
    V_real[0] = energy_loss(args['beta'], args['Vinit']) + args['eta_in']*max(policy[i] - Z[i], 0) - (1/args['eta_out'])*max(Z[i] - policy[i], 0)

    for i in range(1,m):
        V_real[i] = energy_loss(args['beta'], V_real[i-1]) + args['eta_in']*max(policy[i] - Z[i], 0) - (1/args['eta_out'])*max(Z[i] - policy[i], 0)

    if np.min(V_real) < args['cmin'] or np.max(V_real) > args['cmax']:
        print('Error: Solution not feasible!')
    print("Costs: {:.2f}€".format(policy@p/100))


    step_functions(m, args['l'], args['u'], args['cmin'], args['cmax'], args['Vinit'], policy, V, args['start'], args['end'])







