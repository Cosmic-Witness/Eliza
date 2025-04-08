import yfinance
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from colorama import init, Fore, Back, Style

# Initialize colorama for cross‑platform ANSI
init(autoreset=True)

class eliza:
    def __init__(self):
        pass

    def fit(self, stock_ticker, index_ticker, plot=True):
        # ╔═ ELIZA Banner ════════════════════════════════════
        print(Style.BRIGHT + Fore.GREEN + Back.BLACK +
              "\n ███████╗ ██╗      ██████╗  ██████╗  █████╗  \n"
              " ██╔════╝ ██║      ╚══██╔╝  ╚═██╔═╝  ██╔══██╗ \n"
              " █████╗   ██║         ██║      ██║    ███████║\n"
              " ██╔══╝   ██║         ██║     ██║     ██╔══██║\n"
              " ███████╗ ███████╗ ██████╗  ██████╗ ██║  ██║ \n"
              " ╚══════╝ ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ \n")

        # ── Data Fetching ───────────────────────────────────
        index = yfinance.download(index_ticker,
                                  start="2024-01-01",
                                  end="2025-04-07",
                                  auto_adjust=True)
        stock = yfinance.download(stock_ticker,
                                  start="2024-01-01",
                                  end="2025-04-07",
                                  auto_adjust=True)

        # ── Return Calculations ─────────────────────────────
        index['R'] = index['Close'].pct_change()
        stock['R'] = stock['Close'].pct_change()
        returns = pd.concat([index['R'], stock['R']],
                            axis=1, join='inner').dropna()
        returns.columns = ['Market', 'Stock']

        # ── Volatility ──────────────────────────────────────
        vol_m = returns['Market'].std() * (252**0.5)
        vol_s = returns['Stock'].std() * (252**0.5)
        print(Fore.CYAN + Style.BRIGHT + f"• {index_ticker} Volatility: " +
              Fore.GREEN + f"{vol_m*100:6.2f}%")
        print(Fore.CYAN + Style.BRIGHT + f"• {stock_ticker} Volatility: " +
              Fore.GREEN + f"{vol_s*100:6.2f}%\n")

        # ── CAPM Regression ──────────────────────────────────
        X = sm.add_constant(returns['Market'])
        y = returns['Stock']
        model = sm.OLS(y, X).fit()

        beta       = model.params['Market']
        alpha      = model.params['const']
        resid_mean = model.resid.mean()

        # ── Regression Results ──────────────────────────────
        print(Fore.MAGENTA + Style.BRIGHT + "┌─ Regression Results ──────────────────────")
        print(Fore.MAGENTA + "│ " + Fore.YELLOW + f"Mean Residuals: {resid_mean:.8f}")
        print(Fore.MAGENTA + "│ " + Fore.YELLOW + f"Beta          : {beta:.4f}")
        print(Fore.MAGENTA + "│ " + Fore.YELLOW + f"Alpha         : {alpha:.8e}")
        print(Fore.MAGENTA + "└───────────────────────────────────────────\n")

        # ── Optional Plot ───────────────────────────────────
        if plot:
            plt.figure(figsize=(10, 6))
            plt.xlabel(f'{index_ticker} Returns')
            plt.ylabel(f'{stock_ticker} Returns')
            plt.title(f'{stock_ticker} vs {index_ticker} CAPM')
            plt.grid(True)
            plt.axhline(0, ls='--', lw=0.8)
            plt.scatter(returns['Market'], returns['Stock'], alpha=0.5)
            plt.plot(returns['Market'], beta*returns['Market'] + alpha, lw=2)
            plt.show()

# Instantiate and run ELIZA
if __name__ == "__main__":
    eliza = eliza()
    eliza.fit("TSLA", "SPY", plot=False)
