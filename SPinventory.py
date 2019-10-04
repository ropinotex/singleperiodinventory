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
from scipy.stats import rv_discrete


__author__ = "Roberto Pinto"
__copyright__ = "Copyright 2019"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Roberto Pinto"
__email__ = "roberto.pinto@unibg.it"
__status__ = "Use for teaching only"

print('WARNING: This software is designed and provided for educational purpose only')


def simulate(init_inventory=0, selling_price=0, purchasing_cost=0, salvage_value=0, goodwill_cost=0, case=1, size=10000, ret=False):
    if not ret:
        print('THINKING...', end='')
    if case == 1:
        values = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]
        probabilities = [0.01, 0.02, 0.04, 0.08, 0.09, 0.11, 0.16, 0.20, 0.11, 0.10, 0.04, 0.02, 0.01, 0.01]
        demand = rv_discrete(values=(values, probabilities)).rvs(size=size)
    elif case == 2:
        demand = 0
    else:
        print('ERROR: case id not valid')
        return

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

    if not ret:
        print('DONE!')
        print(f'Average profit:     {np.mean(revenue_stream) - purchasing_cost * inventory + np.mean(leftover_stream) * salvage_value - np.mean(lost_sales_stream) * goodwill_cost:>10}')
        print(f'Average revenue:    {np.mean(revenue_stream):>10}')
        print(f'Purchasing cost:    {float(purchasing_cost) * inventory:>10}')
        if salvage_value > 0:
            print(f'Average salvage value:    {np.mean(leftover_stream) * salvage_value:>10}')
        print(f'Average leftover:   {np.mean(leftover_stream):>10}')
        if goodwill_cost > 0:
            print(f'Average lost sales (value): {np.mean(lost_sales_stream) * goodwill_cost:>10}')
        print(f'Average lost sales (quantity): {np.mean(lost_sales_stream):>10}')
    else:
        return np.mean(revenue_stream) - purchasing_cost * inventory + np.mean(leftover_stream) * salvage_value - np.mean(lost_sales_stream) * goodwill_cost


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
    plt.figure(dpi=200)
    plt.plot(levels, profits, label='Average profit')
    plt.xlabel('Inventory')
    plt.ylabel('€')
    plt.legend()
    plt.show()
