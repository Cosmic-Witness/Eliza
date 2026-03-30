import yfinance
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
from colorama import init, Fore, Back, Style
from functools import wraps
import numpy as np

init(autoreset=True)

externalities = []

class eliza:
    def __init__(self, stock_ticker: str, index_ticker: str, start_date: str, end_date: str):

        self.index_ticker = index_ticker
        self.stock_ticker = stock_ticker

        self.index = yfinance.download(index_ticker,start=start_date,end=end_date,auto_adjust=True)

        self.stock = yfinance.download(stock_ticker,start=start_date,end=end_date,auto_adjust=True)

        externalities.append((self.stock, self.index))

        if self.index.empty or self.stock.empty:
            raise RuntimeError(f"Download failed for one or both tickers. " 
            f"You are likely rate-limited.")

        print(Style.BRIGHT + Fore.GREEN + Back.BLACK +
              "\n ███████╗ ██╗      ██████╗  ██████╗  █████╗  \n"
              " ██╔════╝ ██║      ╚══██╔╝  ╚═██╔═╝  ██╔══██╗ \n"
              " █████╗   ██║         ██║      ██║    ███████║\n"
              " ██╔══╝   ██║         ██║     ██║     ██╔══██║\n"
              " ███████╗ ███████╗ ██████╗  ██████╗ ██║  ██║ \n"
              " ╚══════╝ ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ \n")

    def capm(self, plot: bool, annualized: bool):

        self.index['R'] = self.index['Close'].pct_change()
        self.stock['R'] = self.stock['Close'].pct_change()
        returns = pd.concat([self.index['R'], self.stock['R']], axis=1, join='inner').dropna()
        returns.columns = ['Market', 'Stock']

        market_return = (1 + returns['Market']).prod() - 1
        stock_return  = (1 + returns['Stock']).prod()  - 1

        if annualized:
            n_days = len(returns)
            market_return = (1 + market_return) ** (252 / n_days) - 1
            stock_return  = (1 + stock_return)  ** (252 / n_days) - 1

        print(Fore.LIGHTWHITE_EX + Style.DIM + "\n┌── Total Returns ─────────────────────────────────────")
        print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + f"│ {self.index_ticker} Return      : " + Fore.GREEN + f"{market_return*100:.2f}%")
        print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + f"│ {self.stock_ticker} Return      : " + Fore.CYAN + f"{stock_return*100:.2f}%")
        print(Fore.LIGHTWHITE_EX + Style.DIM + "└──────────────────────────────────────────────────────\n")

        X = sm.add_constant(returns['Market'])
        y = returns['Stock']
        model = sm.OLS(y, X).fit()

        beta       = model.params['Market']
        alpha      = model.params['const']
        resid_mean = model.resid.mean()

        return_predictions = model.predict(X)
        actual_predictions = returns['Stock']
        residual_returns   = actual_predictions - return_predictions

        print(residual_returns.head())

        if annualized:
            vol_m    = returns['Market'].std() * (252 ** 0.5)
            vol_s    = returns['Stock'].std()  * (252 ** 0.5)
            idio_vol = math.sqrt((vol_s ** 2) - (beta ** 2 * vol_m ** 2))

            print(Fore.CYAN + Style.BRIGHT + f"• {self.index_ticker} Volatility: " +
                  Fore.GREEN + f"{vol_m*100:6.2f}%")
            print(Fore.CYAN + Style.BRIGHT + f"• {self.stock_ticker} Volatility: " +
                  Fore.GREEN + f"{vol_s*100:6.2f}%")
            print(Fore.CYAN + Style.BRIGHT + f"• Idiosyncratic Volatility: " +
                  Fore.GREEN + f"{idio_vol*100:6.2f}%\n")
        else:
            vol_m    = returns['Market'].std()
            vol_s    = returns['Stock'].std()
            idio_vol = math.sqrt((vol_s ** 2) - (beta ** 2 * vol_m ** 2))

            total_vol = (((vol_m * beta) ** 2) + (vol_s ** 2)) ** 0.5
            print(Fore.CYAN + Style.BRIGHT + f"• {self.index_ticker} Volatility: " +
                  Fore.GREEN + f"{vol_m*100:6.2f}%")
            print(Fore.CYAN + Style.BRIGHT + f"• {self.stock_ticker} Volatility: " +
                  Fore.GREEN + f"{vol_s*100:6.2f}%")
            print(Fore.CYAN + Style.BRIGHT + f"• Idiosyncratic Volatility: " +
                  Fore.GREEN + f"{idio_vol*100:6.2f}%\n")

        print(Fore.MAGENTA + Style.BRIGHT + "┌─ Regression Results ──────────────────────")
        print(Fore.MAGENTA + "│ " + Fore.YELLOW + f"Mean Residuals: {resid_mean:.8f}")
        print(Fore.MAGENTA + "│ " + Fore.YELLOW + f"Beta          : {beta:.4f}")
        print(Fore.MAGENTA + "│ " + Fore.YELLOW + f"Alpha         : {alpha:.8f}")
        print(Fore.MAGENTA + "└───────────────────────────────────────────\n")

        if plot:
            plt.figure(figsize=(10, 6))
            plt.xlabel(f'{self.index_ticker} Returns')
            plt.ylabel(f'{self.stock_ticker} Returns')
            plt.title(f'{self.stock_ticker} vs {self.index_ticker} CAPM')
            plt.grid(True)
            plt.axhline(0, ls='--', lw=0.8)
            plt.scatter(returns['Market'], returns['Stock'], alpha=0.5)
            plt.plot(returns['Market'], beta * returns['Market'] + alpha, lw=2, color='red')
            plt.show()

    def risk_decomposition(self, stock_value: float, beta: float, market_vol: float, idiosyncratic_vol: float):
        market_vol        = market_vol / 100
        idiosyncratic_vol = idiosyncratic_vol / 100

        market_risk       = beta * market_vol * stock_value
        idiosyncratic_risk = idiosyncratic_vol * stock_value
        total_risk        = np.sqrt(market_risk ** 2 + idiosyncratic_risk ** 2)

        print(Fore.LIGHTBLACK_EX + Style.DIM + "\n┌── Risk Decomposition ───────────────────────────────")
        print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + f"│ Stock Value         : " + Fore.CYAN + f"${stock_value:,.2f}")
        print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + f"│ Beta                : " + Fore.MAGENTA + f"{beta:.4f}")
        print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + f"│ Market Volatility   : " + Fore.GREEN + f"{market_vol*100:.2f}%")
        print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + f"│ Idio Volatility     : " + Fore.GREEN + f"{idiosyncratic_vol*100:.2f}%")

        print(Fore.LIGHTBLACK_EX + "├────────────────────────────────────────────────────")
        print(Fore.LIGHTCYAN_EX + Style.BRIGHT + f"│ Market Risk         : " + Fore.YELLOW + f"${market_risk:,.2f}")
        print(Fore.LIGHTCYAN_EX + Style.BRIGHT + f"│ Idiosyncratic Risk  : " + Fore.YELLOW + f"${idiosyncratic_risk:,.2f}")
        print(Fore.LIGHTCYAN_EX + Style.BRIGHT + f"│ Total Risk          : " + Fore.RED + Style.BRIGHT + f"${total_risk:,.2f}")
        print(Fore.LIGHTBLACK_EX + "└────────────────────────────────────────────────────\n")

        return {
            "Market Risk": market_risk,
            "Idiosyncratic Risk": idiosyncratic_risk,
            "Total Risk": total_risk
        }

    def performance_attribution(self, net_market_values: pd.Series, betas: pd.Series, noise):
        raise NotImplementedError

    def idiosyncratic_attribution(self):
        raise NotImplementedError


if __name__ == "__main__":
    eli = eliza("NVDA", "SPY", "2026-01-01", "2026-03-29")
    eli.capm(True, True)
