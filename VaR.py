
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import yfinance as yf
import customtkinter as ctk

from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# CONFIGURATION
# ============================================================

MARKETS = {
    "Gold futures (GC=F)": {"ticker": "GC=F", "currency": "USD"},
    "S&P 500 index (^GSPC)": {"ticker": "^GSPC", "currency": "USD"},
    "Nasdaq-100 index (^NDX)": {"ticker": "^NDX", "currency": "USD"},
    "CAC 40 (^FCHI)": {"ticker": "^FCHI", "currency": "EUR"},
    "EUR/USD (EURUSD=X)": {"ticker": "EURUSD=X", "currency": "USD"},
    "Apple shares (AAPL)": {"ticker": "AAPL", "currency": "USD"},
}

CONFIDENCE_LEVELS = {
    "95%": 0.95,
    "99%": 0.99,
    "99.9%": 0.999,
}

STRESS_PERIODS = {
    "COVID crisis": ("2019-09-01", "2020-09-01"),
    "Financial crisis": ("2008-09-01", "2009-09-01"),
}

SIMULATIONS = 100_000

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================
# GENERAL HELPERS
# ============================================================

def parse_number(value):
    """Accept numbers using either a decimal point or comma."""
    value = value.strip().replace(" ", "").replace(",", ".")

    if not value:
        raise ValueError("Please enter a number.")

    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            "Please enter a valid number, e.g. 1000000 or 10.50."
        ) from exc

    if not np.isfinite(number):
        raise ValueError("The number must be finite.")

    return number


def format_number(value, decimals=2):
    """Format numbers with spaces for thousands."""
    return f"{value:,.{decimals}f}".replace(",", " ")


def get_market_data(ticker, period="1y", start=None, end=None):
    """
    Download market data from Yahoo Finance.

    When start and end are specified, end is exclusive.
    """
    if start is not None and end is not None:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    else:
        data = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
        )

    if data.empty:
        raise ValueError(
            f"No market data available for {ticker} "
            "over the selected period."
        )

    # Handle both ordinary and MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        data = pd.DataFrame({"Close": close}, index=data.index)

    elif "Close" in data.columns:
        data = data[["Close"]].copy()

    else:
        raise ValueError("The downloaded data contains no Close column.")

    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Close"]).sort_index()

    if len(data) < 3:
        raise ValueError("Not enough market data for the selected period.")

    return data


def rendement(data):
    """Calculate daily simple returns."""
    return data["Close"].pct_change().dropna()


def historical_parameter_estimation(data):
    """
    Estimate daily mean return mu and daily volatility sigma
    from historical returns.
    """
    returns = rendement(data)

    if len(returns) < 2:
        raise ValueError("Not enough returns to estimate parameters.")

    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))

    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(
            "Unable to estimate valid daily mean and volatility."
        )

    return mu, sigma


# ============================================================
# LOSS DISTRIBUTIONS, VAR AND EXPECTED SHORTFALL
# ============================================================

def parametric_losses(mu, sigma, days, simulations=SIMULATIONS):
    """
    Parametric loss distribution.

    Daily returns are approximated by a normal distribution.
    The aggregate N-day return is approximated by:
        N * mu + sqrt(N) * sigma * Z
    """
    if days <= 0 or sigma <= 0:
        raise ValueError("Days and volatility must be positive.")

    rng = np.random.default_rng()

    simulated_returns = rng.normal(
        loc=mu * days,
        scale=sigma * np.sqrt(days),
        size=simulations,
    )

    # Positive value = loss; negative value = gain.
    return -simulated_returns


def historical_losses(data, days):
    """
    Calculate overlapping N-day historical losses using compounded
    daily returns. Each loss is a fraction of the portfolio value.
    """
    if days <= 0:
        raise ValueError("The holding period must be positive.")

    returns = rendement(data).to_numpy(dtype=float)

    if len(returns) < days:
        raise ValueError(
            f"Not enough historical returns ({len(returns)}) "
            f"for a {days}-day holding period."
        )

    growth_factors = 1 + returns
    losses = []

    for end_index in range(days, len(growth_factors) + 1):
        window = growth_factors[end_index - days:end_index]
        cumulative_growth = np.prod(window)
        losses.append(1 - cumulative_growth)

    losses = np.asarray(losses, dtype=float)
    return losses[np.isfinite(losses)]


def monte_carlo_losses(mu, sigma, days, simulations=SIMULATIONS):
    """Monte Carlo losses from a lognormal price process."""
    if days <= 0 or sigma <= 0:
        raise ValueError("Days and volatility must be positive.")

    rng = np.random.default_rng()

    # Approximate log-return parameters from historical simple returns.
    log_mu = np.log(max(1 + mu, 1e-12)) - 0.5 * sigma**2

    cumulative_log_returns = rng.normal(
        loc=log_mu * days,
        scale=sigma * np.sqrt(days),
        size=simulations,
    )

    cumulative_growth = np.exp(cumulative_log_returns)

    return 1 - cumulative_growth


def calculate_var_es(losses, portfolio_value, confidence):
    """
    Calculate VaR and Expected Shortfall from the same loss distribution.

    ES = mean(losses[losses >= VaR]).
    """
    losses = np.asarray(losses, dtype=float)
    losses = losses[np.isfinite(losses)]

    if len(losses) == 0:
        raise ValueError("The loss distribution is empty.")

    var_fraction = float(np.quantile(losses, confidence))
    tail_losses = losses[losses >= var_fraction]

    if len(tail_losses) == 0:
        raise ValueError("Unable to calculate Expected Shortfall.")

    es_fraction = float(np.mean(tail_losses))

    var_value = portfolio_value * var_fraction
    es_value = portfolio_value * es_fraction

    return var_value, es_value


# ============================================================
# DISTRIBUTIONS FOR FINANCIAL ANALYSIS
# ============================================================

def gaussian(x, mu, sigma):
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    return (
        1 / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    )


def log_norm(x, mu, sigma):
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    valid = x > 0

    result[valid] = (
        1 / (x[valid] * sigma * np.sqrt(2 * np.pi))
        * np.exp(
            -((np.log(x[valid]) - mu) ** 2) / (2 * sigma**2)
        )
    )

    return result


# ============================================================
# MAIN APPLICATION
# ============================================================

class VaRApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("VaR & Expected Shortfall")
        self.geometry("1150x780")
        self.minsize(950, 650)

        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(
            self.sidebar,
            text="RISK ANALYSIS",
            font=ctk.CTkFont(size=21, weight="bold"),
        ).pack(padx=15, pady=(30, 25))

        ctk.CTkButton(
            self.sidebar,
            text="VaR",
            command=self.build_var_page,
        ).pack(fill="x", padx=15, pady=7)

        ctk.CTkButton(
            self.sidebar,
            text="VaR during crisis",
            command=self.build_stressed_page,
        ).pack(fill="x", padx=15, pady=7)

        ctk.CTkButton(
            self.sidebar,
            text="Financial analysis",
            command=self.build_analysis_page,
        ).pack(fill="x", padx=15, pady=7)

        ctk.CTkLabel(
            self.sidebar,
            text="Appearance",
        ).pack(padx=15, pady=(35, 5), anchor="w")

        self.theme_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["Dark", "Light", "System"],
            command=ctk.set_appearance_mode,
        )
        self.theme_menu.set("Dark")
        self.theme_menu.pack(fill="x", padx=15, pady=5)

        self.content = ctk.CTkScrollableFrame(self)
        self.content.pack(
            side="right",
            fill="both",
            expand=True,
            padx=15,
            pady=15,
        )

        self.build_var_page()

    # --------------------------------------------------------
    # COMMON UI HELPERS
    # --------------------------------------------------------

    def clear_content(self):
        for widget in self.content.winfo_children():
            widget.destroy()

    def add_title(self, title, subtitle=None):
        ctk.CTkLabel(
            self.content,
            text=title,
            font=ctk.CTkFont(size=25, weight="bold"),
        ).pack(anchor="w", padx=8, pady=(12, 5))

        if subtitle:
            ctk.CTkLabel(
                self.content,
                text=subtitle,
                text_color=("gray35", "gray75"),
                wraplength=800,
                justify="left",
            ).pack(anchor="w", padx=8, pady=(0, 15))

    def add_field(self, label_text, placeholder, default=""):
        ctk.CTkLabel(
            self.content,
            text=label_text,
            text_color=("gray25", "gray80"),
        ).pack(anchor="w", padx=8, pady=(10, 3))

        entry = ctk.CTkEntry(
            self.content,
            placeholder_text=placeholder,
            height=38,
        )

        if default != "":
            entry.insert(0, str(default))

        entry.pack(fill="x", padx=8, pady=(0, 5))
        return entry

    def add_market_menu(self):
        ctk.CTkLabel(
            self.content,
            text="Financial instrument",
            text_color=("gray25", "gray80"),
        ).pack(anchor="w", padx=8, pady=(10, 3))

        menu = ctk.CTkOptionMenu(
            self.content,
            values=list(MARKETS.keys()),
        )
        menu.set("Gold futures (GC=F)")
        menu.pack(fill="x", padx=8, pady=(0, 5))

        return menu

    def add_portfolio_input(self, market_menu):
        ctk.CTkLabel(
            self.content,
            text="Portfolio input type",
            text_color=("gray25", "gray80"),
        ).pack(anchor="w", padx=8, pady=(12, 3))

        mode = ctk.CTkSegmentedButton(
            self.content,
            values=["Portfolio value", "Number of shares/units"],
        )
        mode.set("Portfolio value")
        mode.pack(fill="x", padx=8, pady=(0, 8))

        label = ctk.CTkLabel(
            self.content,
            text="Total portfolio value (USD)",
            text_color=("gray25", "gray80"),
        )
        label.pack(anchor="w", padx=8, pady=(8, 3))

        entry = ctk.CTkEntry(
            self.content,
            placeholder_text="e.g. 1000000",
            height=38,
        )
        entry.insert(0, "10000")
        entry.pack(fill="x", padx=8, pady=(0, 5))

        def update_currency(choice):
            currency = MARKETS[choice]["currency"]

            if mode.get() == "Portfolio value":
                label.configure(text=f"Total portfolio value ({currency})")
            else:
                label.configure(text="Number of shares/units")

        def update_mode(choice):
            entry.delete(0, "end")

            if choice == "Portfolio value":
                entry.insert(0, "10000")
            else:
                entry.insert(0, "100")

            update_currency(market_menu.get())

        mode.configure(command=update_mode)
        market_menu.configure(command=update_currency)
        update_currency(market_menu.get())

        return mode, entry

    def get_portfolio_value(self, mode, entry, ticker):
        """
        In quantity mode, estimate the position value as:
            quantity * latest closing price.

        This simple calculation does not apply futures contract
        multipliers or convert foreign currencies.
        """
        value = parse_number(entry.get())

        if value <= 0:
            raise ValueError("The entered value must be positive.")

        if mode == "Portfolio value":
            return value

        if not value.is_integer():
            raise ValueError(
                "The number of shares/units must be a whole number."
            )

        data = get_market_data(ticker, period="5d")
        latest_price = float(data["Close"].iloc[-1])

        if latest_price <= 0:
            raise ValueError("The latest market price is invalid.")

        return value * latest_price

    def add_confidence_menu(self):
        ctk.CTkLabel(
            self.content,
            text="Confidence level",
            text_color=("gray25", "gray80"),
        ).pack(anchor="w", padx=8, pady=(12, 3))

        menu = ctk.CTkSegmentedButton(
            self.content,
            values=list(CONFIDENCE_LEVELS.keys()),
        )
        menu.set("99%")
        menu.pack(fill="x", padx=8, pady=(0, 15))

        return menu

    def add_days_input(self):
        return self.add_field(
            "Holding period (trading days)",
            "e.g. 10",
            "1",
        )

    def display_results(
        self,
        result_label,
        details_label,
        var_value,
        es_value,
        currency,
        confidence_label,
        method,
        market,
        portfolio_value,
        days,
        mu,
        sigma,
        start=None,
        end=None,
    ):
        result_label.configure(
            text=(
                f"VaR ({confidence_label}): "
                f"{currency} {format_number(var_value)}\n"
                f"Expected Shortfall (ES): "
                f"{currency} {format_number(es_value)}"
            )
        )

        details = (
            f"Instrument: {market}\n"
            f"Method: {method}\n"
            f"Portfolio value used: {currency} "
            f"{format_number(portfolio_value)}\n"
            f"Holding period: {days} trading day(s)\n"
            f"Confidence level: {confidence_label}\n"
            f"Estimated daily mean return (mu): {mu:.6%}\n"
            f"Estimated daily volatility (sigma): {sigma:.6%}"
        )

        if start is not None and end is not None:
            details += f"\nHistorical period used: {start} to {end}"

        details_label.configure(text=details)

    # --------------------------------------------------------
    # PAGE 1: VAR
    # --------------------------------------------------------

    def build_var_page(self):
        self.clear_content()

        self.add_title(
            "Value at Risk (VaR)",
            "Calculate parametric VaR and Expected Shortfall using "
            "historical returns. The displayed currency follows the "
            "selected instrument.",
        )

        self.var_market = self.add_market_menu()

        self.var_portfolio_mode, self.var_price = (
            self.add_portfolio_input(self.var_market)
        )

        self.var_days = self.add_days_input()
        self.var_confidence = self.add_confidence_menu()

        ctk.CTkButton(
            self.content,
            text="Calculate VaR and ES",
            height=42,
            command=self.compute_var,
        ).pack(fill="x", padx=8, pady=10)

        self.var_result = ctk.CTkLabel(
            self.content,
            text="VaR: —\nExpected Shortfall: —",
            font=ctk.CTkFont(size=20, weight="bold"),
            wraplength=800,
            justify="left",
        )
        self.var_result.pack(anchor="w", padx=8, pady=20)

        self.var_details = ctk.CTkLabel(
            self.content,
            text="",
            justify="left",
            wraplength=800,
        )
        self.var_details.pack(anchor="w", padx=8, pady=5)

    def compute_var(self):
        try:
            days_value = parse_number(self.var_days.get())

            if not days_value.is_integer() or days_value <= 0:
                raise ValueError(
                    "Holding period must be a positive whole number."
                )

            days = int(days_value)
            market = self.var_market.get()
            market_info = MARKETS[market]
            ticker = market_info["ticker"]
            currency = market_info["currency"]

            portfolio_value = self.get_portfolio_value(
                self.var_portfolio_mode.get(),
                self.var_price,
                ticker,
            )

            data = get_market_data(ticker, period="1y")
            mu, sigma = historical_parameter_estimation(data)

            confidence_label = self.var_confidence.get()
            confidence = CONFIDENCE_LEVELS[confidence_label]

            losses = parametric_losses(mu, sigma, days)
            var_value, es_value = calculate_var_es(
                losses, portfolio_value, confidence
            )

            self.display_results(
                self.var_result,
                self.var_details,
                var_value,
                es_value,
                currency,
                confidence_label,
                "Parametric",
                market,
                portfolio_value,
                days,
                mu,
                sigma,
            )

        except Exception as exc:
            messagebox.showerror("VaR calculation error", str(exc))

    # --------------------------------------------------------
    # PAGE 2: VAR DURING CRISIS
    # --------------------------------------------------------

    def build_stressed_page(self):
        self.clear_content()

        self.add_title(
            "VaR during crisis",
            "Estimate VaR and Expected Shortfall using a historical "
            "crisis window. Custom windows must be at least one calendar year.",
        )

        self.stress_market = self.add_market_menu()

        self.stress_portfolio_mode, self.stress_price = (
            self.add_portfolio_input(self.stress_market)
        )

        self.stress_days = self.add_days_input()
        self.stress_confidence = self.add_confidence_menu()

        ctk.CTkLabel(
            self.content,
            text="Calculation method",
            text_color=("gray25", "gray80"),
        ).pack(anchor="w", padx=8, pady=(12, 3))

        self.stress_method = ctk.CTkOptionMenu(
            self.content,
            values=["Historical", "Parametric", "Monte Carlo"],
        )
        self.stress_method.set("Historical")
        self.stress_method.pack(fill="x", padx=8, pady=(0, 8))

        ctk.CTkLabel(
            self.content,
            text="Crisis period",
            text_color=("gray25", "gray80"),
        ).pack(anchor="w", padx=8, pady=(12, 3))

        self.stress_period = ctk.CTkOptionMenu(
            self.content,
            values=["COVID crisis", "Financial crisis", "Custom dates"],
            command=self.update_stress_period,
        )
        self.stress_period.set("COVID crisis")
        self.stress_period.pack(fill="x", padx=8, pady=(0, 8))

        self.stress_start = self.add_field(
            "Start date (YYYY-MM-DD)",
            "e.g. 2019-09-01",
            "2019-09-01",
        )

        self.stress_end = self.add_field(
            "End date (YYYY-MM-DD)",
            "e.g. 2020-09-01",
            "2020-09-01",
        )

        ctk.CTkLabel(
            self.content,
            text=(
                "Daily mean return (mu) and daily volatility (sigma) "
                "are estimated automatically from the selected crisis period."
            ),
            text_color=("gray35", "gray75"),
            justify="left",
            wraplength=800,
        ).pack(anchor="w", padx=8, pady=12)

        ctk.CTkButton(
            self.content,
            text="Calculate VaR and ES",
            height=42,
            command=self.compute_stressed_var,
        ).pack(fill="x", padx=8, pady=10)

        self.stress_result = ctk.CTkLabel(
            self.content,
            text="VaR: —\nExpected Shortfall: —",
            font=ctk.CTkFont(size=20, weight="bold"),
            wraplength=800,
            justify="left",
        )
        self.stress_result.pack(anchor="w", padx=8, pady=20)

        self.stress_details = ctk.CTkLabel(
            self.content,
            text="",
            justify="left",
            wraplength=800,
        )
        self.stress_details.pack(anchor="w", padx=8, pady=5)

    def update_stress_period(self, choice):
        if choice == "Custom dates":
            return

        start, end = STRESS_PERIODS[choice]

        self.stress_start.delete(0, "end")
        self.stress_start.insert(0, start)

        self.stress_end.delete(0, "end")
        self.stress_end.insert(0, end)

    def compute_stressed_var(self):
        try:
            days_value = parse_number(self.stress_days.get())

            if not days_value.is_integer() or days_value <= 0:
                raise ValueError(
                    "Holding period must be a positive whole number."
                )

            days = int(days_value)

            start_text = self.stress_start.get().strip()
            end_text = self.stress_end.get().strip()

            start_date = pd.Timestamp(
                pd.to_datetime(start_text, errors="raise")
            ).normalize()

            end_date = pd.Timestamp(
                pd.to_datetime(end_text, errors="raise")
            ).normalize()

            if end_date <= start_date:
                raise ValueError("The end date must be after the start date.")

            if end_date < start_date + pd.DateOffset(years=1):
                raise ValueError(
                    "The selected period must be at least one calendar year."
                )

            market = self.stress_market.get()
            market_info = MARKETS[market]
            ticker = market_info["ticker"]
            currency = market_info["currency"]

            portfolio_value = self.get_portfolio_value(
                self.stress_portfolio_mode.get(),
                self.stress_price,
                ticker,
            )

            # Yahoo Finance excludes the end date; add one day so that
            # the selected end date is included in the requested window.
            data = get_market_data(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            )

            mu, sigma = historical_parameter_estimation(data)

            method = self.stress_method.get()
            confidence_label = self.stress_confidence.get()
            confidence = CONFIDENCE_LEVELS[confidence_label]

            if method == "Historical":
                losses = historical_losses(data, days)

            elif method == "Parametric":
                losses = parametric_losses(mu, sigma, days)

            else:
                losses = monte_carlo_losses(mu, sigma, days)

            var_value, es_value = calculate_var_es(
                losses, portfolio_value, confidence
            )

            self.display_results(
                self.stress_result,
                self.stress_details,
                var_value,
                es_value,
                currency,
                confidence_label,
                method,
                market,
                portfolio_value,
                days,
                mu,
                sigma,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )

        except Exception as exc:
            messagebox.showerror(
                "VaR during crisis calculation error", str(exc)
            )

    # --------------------------------------------------------
    # PAGE 3: FINANCIAL ANALYSIS
    # --------------------------------------------------------

    def build_analysis_page(self):
        self.clear_content()

        self.add_title(
            "Financial analysis",
            "Compare historical returns with a normal distribution, "
            "or prices with a lognormal distribution.",
        )

        self.analysis_market = self.add_market_menu()

        ctk.CTkLabel(
            self.content,
            text="Analysis",
            text_color=("gray25", "gray80"),
        ).pack(anchor="w", padx=8, pady=(12, 3))

        self.analysis_type = ctk.CTkOptionMenu(
            self.content,
            values=["Returns vs Normal", "Prices vs Log-normal"],
        )
        self.analysis_type.set("Returns vs Normal")
        self.analysis_type.pack(fill="x", padx=8, pady=(0, 10))

        ctk.CTkButton(
            self.content,
            text="Generate analysis",
            height=42,
            command=self.plot_analysis,
        ).pack(fill="x", padx=8, pady=10)

        self.plot_frame = ctk.CTkFrame(self.content)
        self.plot_frame.pack(
            fill="both",
            expand=True,
            padx=8,
            pady=15,
        )

    def plot_analysis(self):
        try:
            market_info = MARKETS[self.analysis_market.get()]
            data = get_market_data(market_info["ticker"], period="1y")
            analysis = self.analysis_type.get()

            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            fig = Figure(figsize=(8, 5), dpi=100)
            ax = fig.add_subplot(111)

            if analysis == "Returns vs Normal":
                returns = rendement(data)
                mu = float(returns.mean())
                sigma = float(returns.std(ddof=1))

                if sigma <= 0:
                    raise ValueError(
                        "Volatility must be positive to plot a normal fit."
                    )

                ax.hist(
                    returns,
                    bins=40,
                    density=True,
                    alpha=0.65,
                    label="Historical returns",
                )

                x = np.linspace(
                    float(returns.min()),
                    float(returns.max()),
                    500,
                )

                ax.plot(
                    x,
                    gaussian(x, mu, sigma),
                    linewidth=2,
                    label="Normal distribution",
                )

                ax.set_title("Daily returns vs Normal distribution")
                ax.set_xlabel("Daily return")
                ax.set_ylabel("Probability density")

            else:
                prices = data["Close"].dropna()

                if (prices <= 0).any():
                    raise ValueError(
                        "Lognormal analysis requires positive prices."
                    )

                log_prices = np.log(prices)
                mu = float(log_prices.mean())
                sigma = float(log_prices.std(ddof=1))

                if sigma <= 0:
                    raise ValueError("Price dispersion must be positive.")

                ax.hist(
                    prices,
                    bins=40,
                    density=True,
                    alpha=0.65,
                    label="Historical prices",
                )

                x = np.linspace(
                    float(prices.min()),
                    float(prices.max()),
                    500,
                )

                ax.plot(
                    x,
                    log_norm(x, mu, sigma),
                    linewidth=2,
                    label="Lognormal distribution",
                )

                ax.set_title("Prices vs Lognormal distribution")
                ax.set_xlabel(
                    f"Market price ({market_info['currency']})"
                )
                ax.set_ylabel("Probability density")

            ax.legend()
            ax.grid(True, alpha=0.25)
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as exc:
            messagebox.showerror("Financial analysis error", str(exc))


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app = VaRApp()
    app.mainloop()
