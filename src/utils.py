import copy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pypfopt.plotting import plot_efficient_frontier


def custom_efficient_frontier(ef, show_assets=True, figsize=(12, 10)):

    # Create a copy of the efficient frontier class instance
    ef_copy = copy.deepcopy(ef)

    fig, ax = plt.subplots(figsize=figsize)

    # Create copies for each portfolio
    ef_max_sharpe = copy.deepcopy(ef_copy)
    ef_min_vol = copy.deepcopy(ef_copy)
    ef_max_quadratic_utility = copy.deepcopy(ef_copy)

    # Plot the efficient frontier
    plot_efficient_frontier(ef_copy, ax=ax, show_assets=show_assets)

    # Find the tangency portfolio
    ef_max_sharpe.max_sharpe()
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*",
               s=200, c="r", label="Max Sharpe")

    # Find the minimum volatility portfolio
    ef_min_vol.min_volatility()
    ret_min_vol, std_min_vol, _ = ef_min_vol.portfolio_performance()
    ax.scatter(std_min_vol, ret_min_vol, marker="p",
               s=200, c="g", label="Min Volatility")

    # Find max quadratic utility portfolio
    ef_max_quadratic_utility.max_quadratic_utility(risk_aversion=0.1)
    ret_max_quadratic_utility, std_max_quadratic_utility, _ = ef_max_quadratic_utility.portfolio_performance()
    ax.scatter(std_max_quadratic_utility, ret_max_quadratic_utility,
               marker="^", s=200, c="b", label="Max Quadratic Utility")

    # Generate random portfolios
    n_samples = 10000
    w = np.random.dirichlet(np.ones(ef_copy.n_assets), n_samples)
    # Compute the returns for each random portfolio
    rets = w.dot(ef_copy.expected_returns)
    # Compute the standard deviation of each random portfolio
    stds = np.sqrt(np.diag(w @ ef_copy.cov_matrix @ w.T))
    sharpes = rets / stds  # Compute the Sharpe ratio for each random portfolio
    # Plot the random portfolios, using a colormap to indicate the Sharpe ratio
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Output
    ax.set_title("Efficient Frontier with Random Portfolios")
    ax.legend()
    plt.tight_layout()

    return ax
