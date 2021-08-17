import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib import rcParams


def boxplot(df, y_min, y_max, y_steps, y_label, start_time_step=1, end_time_step=25, teiler=1):

    x_step = 1
    stunden = (end_time_step - start_time_step) / x_step
    hours = [datetime.time(x_step * num).strftime("%H:00") for num in range(1, int(stunden)+1)]
    ax1 = df.iloc[:, start_time_step:end_time_step].plot.box(patch_artist=True, showfliers=True,
                                                             flierprops=dict(markeredgecolor='#6d005e', marker='+', markersize=0.5),
                                                             color={'boxes': '#89ba17', 
                                                                    'whiskers': '#6d005e',
                                                                    'medians': 'White',
                                                                    'caps': '#89ba17'})
    ax1.set_xticks(np.arange(0, end_time_step - start_time_step, teiler * x_step))
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_xticklabels(np.asarray(hours))
    ax1.set_yticks(np.arange(y_min, y_max, y_steps))
    plt.xlabel("Uhrzeit")
    plt.ylabel(y_label)
    plt.tight_layout(1)
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams.update({'font.size': 10})
    ax1.grid()
    plt.savefig("./figures/boxplot.pdf", dpi=150)
    
    
def heatmap(data, m, t):
    X = np.linspace(1, t, t)
    Y = np.linspace(1, int(m/t), int(m/t))
    dfp = pd.DataFrame(data, index=Y, columns=X)
    sns.heatmap(dfp, cmap='jet')
    plt.savefig("./figures/heatmap.pdf", dpi=150)
    
    
def step_functions(m, l, u, c, C, Vinit, policy, V, start, end):
    plt.step(np.arange(m+1), np.concatenate(([0], policy)), where='post', color='r', linestyle='--', linewidth=2)
    plt.step(np.arange(m+1), np.concatenate(([Vinit], V)), where='post', color='k', linestyle=':', linewidth=2)

    plt.plot([0, m], [l, l], color='r', linestyle='--', linewidth=1)
    plt.plot([0, m], [u, u], color='r', linestyle='--', linewidth=1)
    plt.plot([0, m], [c, c], color='k', linestyle=':', linewidth=1)
    plt.plot([0, m], [C, C], color='k', linestyle=':', linewidth=1)

    plt.xticks(np.arange(0, m, 5))
    legend = ['energy','fill level']
    plt.legend(legend, loc=1)
    plt.savefig("./figures/step_function_{}_{}.pdf".format(start, end), dpi=150)
    
    

#data = p.reshape(int(m/t), t)
#res_bus_agg = pd.DataFrame(policy.reshape(int(m/t), t))    
#boxplot(res_bus_agg, 0, 1000, 100, "Purchased Energy")


