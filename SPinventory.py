# ==============================================================================
# description     :Single period inventory analysis toolbox
# author          :Roberto Pinto
# date            :2019.09.25
# version         :1.0
# notes           :This software is meant for teaching purpose only and it is provided as-is
# python_version  :3.7
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete, truncnorm


__author__ = "Roberto Pinto"
__copyright__ = "Copyright 2019"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Roberto Pinto"
__email__ = "roberto.pinto@unibg.it"
__status__ = "Use for teaching only"

print('WARNING: This software is designed and provided for educational purpose only')


def tn(mu, sigma, lower, upper, n):
    """ n realizations of a truncated normal between lower and upper """
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    np.random.seed(seed=123456)
    return truncnorm(a, b, mu, sigma).rvs(n)


def data(case=1, plot=False, size=10000):

    if case == 1:
        values = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]
        probabilities = [0.01, 0.02, 0.04, 0.08, 0.09, 0.11, 0.16, 0.20, 0.11, 0.10, 0.04, 0.02, 0.01, 0.01]
        demand = rv_discrete(values=(values, probabilities), seed=12345).rvs(size=size)
    elif case == 2:
        demand = tn(750, 300, 100, 1100, size)
    elif case == 3:
        demand = tn(250, 650, 100, 1100, size)
    elif case == 4:
        demand = tn(250, 250, 100, 1100, size)
    else:
        print('ERROR: case id not valid')
        return

    if plot:
        plt.figure(figsize=(10, 5), dpi=96)
        plt.hist(demand, bins=100, density=True, stacked=True, label='Demand frequency')
        plt.xlabel('Demand')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    return demand


def simulate(init_inventory=0, selling_price=0, purchasing_cost=0, salvage_value=0, goodwill_cost=0, case=1, size=10000, ret=False):
    if not ret:
        print('THINKING...', end='')

    demand = data(case=case)
    inventory = init_inventory
    revenue_stream = []
    lost_sales_stream = []
    salvage_value_stream = []
    leftover_stream = []

    for t in range(size):
        sales = min(inventory, demand[t])
        lost_sales = max(0., demand[t] - inventory)
        leftover = max(0., inventory - demand[t])

        revenue_stream.append(sales * selling_price)
        leftover_stream.append(leftover)
        salvage_value_stream.append(leftover * salvage_value)
        lost_sales_stream.append(lost_sales)

    avg_profit = int(np.mean(revenue_stream) - purchasing_cost * inventory + np.mean(leftover_stream) * salvage_value - np.mean(lost_sales_stream) * goodwill_cost)
    avg_revenue = int(np.mean(revenue_stream))
    purchasing_cost_per_period = int(purchasing_cost) * inventory
    avg_salvage_value = int(np.mean(leftover_stream) * salvage_value)
    avg_leftover = int(np.mean(leftover_stream))
    avg_lost_sales = int(np.mean(lost_sales_stream))
    avg_lost_sales_value = int(avg_lost_sales * goodwill_cost)
    if not ret:
        print('DONE!')
        print()
        print(f'Average profit:                 {avg_profit:>10} €')
        print('- ' * 30)
        print(f'Average revenue:                {avg_revenue:>10} €')
        print(f'Purchasing cost:                {-purchasing_cost_per_period:>10} €')
        if salvage_value != 0:
            print(f'Average salvage value:          {avg_salvage_value:>10} €')
        if goodwill_cost > 0:
            print(f'Average lost sales (value):     {-avg_lost_sales_value:>10} €')
        print('- ' * 30)
        print(f'Average leftover:               {avg_leftover:>10} units')
        print(f'Average lost sales (quantity):  {avg_lost_sales:>10} units')
    else:
        return int(np.mean(revenue_stream) - purchasing_cost * inventory + np.mean(leftover_stream) * salvage_value - np.mean(lost_sales_stream) * goodwill_cost)


def plot_profits(levels=None, selling_price=0, purchasing_cost=0, salvage_value=0, goodwill_cost=0, case=1, size=10000):
    if not isinstance(levels, list):
        levels = [levels]
    print('THINKING...', end='')
    profits = []
    for each in levels:
        profits.append(simulate(init_inventory=each,
                                selling_price=selling_price,
                                purchasing_cost=purchasing_cost,
                                salvage_value=salvage_value,
                                goodwill_cost=goodwill_cost,
                                case=case,
                                size=size,
                                ret=True))
    print('PLOTTING...', end='')
    plt.figure(figsize=(10, 5), dpi=96)
    plt.plot(levels, profits, label='Average profit', marker='o')
    plt.ylabel('€')
    plt.table(cellText=[levels, profits],
              rowLabels=['Inventory', 'Profit'],
              colLabels=None,
              loc='bottom')
    plt.xticks([])
    plt.legend()
    plt.show()
