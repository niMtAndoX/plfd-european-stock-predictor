import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

# Base data directory: src/data_processing/data
BASE_DATA_DIR = Path(__file__).resolve().parent / "data"

# Use pathlib for OS-independent paths and verify files exist before reading
csv_dir = BASE_DATA_DIR / "clean_features"
csv_paths = {
    "SSE": csv_dir / "SSE_features.csv",
    "KOSPI": csv_dir / "KOSPI_features.csv",
    "TAIEX": csv_dir / "TAIEX_features.csv",
    "STOXX600": csv_dir / "STOXX600_features.csv",
    "NIKKEI225": csv_dir / "NIKKEI225_features.csv",

}

# Output folder for images: src/data_processing/data/img
img_dir = BASE_DATA_DIR / "img"
img_dir.mkdir(parents=True, exist_ok=True)

def plot_data_daily():
    for name, path in csv_paths.items():
        path = Path(path)
        if not path.exists():
            print(f"Warning: file not found, skipping {name}: {path}")
            continue

        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.set_index("Date")

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot Return_t
        ax1.plot(df.index, df["Return_t"], color="blue", label="Return_t")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Return_t", color="blue")
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))  # <-- als %
        ax1.tick_params(axis="y", labelcolor="blue")

        # Second y-axis for STD_5
        ax2 = ax1.twinx()
        ax2.plot(df.index, df["STD_5"], color="red", label="STD_5")
        ax2.set_ylabel("STD_5", color="red")
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))  # <-- als %
        ax2.tick_params(axis="y", labelcolor="red")

        plt.title(f"{name} — Return_t & STD_5 over time")
        fig.tight_layout()

        # Save figure
        out_path = img_dir / f"{name}_return_std.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"Saved plot for {name} at {out_path}")

def plot_all_in_one_daily():
    plt.figure(figsize=(14, 7), dpi=200)

    for name, path in csv_paths.items():
        path = Path(path)
        if not path.exists():
            print(f"Warning: file not found, skipping {name}: {path}")
            continue

        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.set_index("Date")

        # Plot Return_t for each index
        plt.plot(df.index, df["Return_t"], label=name)

    plt.xlabel("Date")
    plt.ylabel("Return_t")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))  # <-- as %
    plt.title("Return_t over time — all indices")
    plt.legend()
    plt.tight_layout()

    out_path = img_dir / "all_indices_return.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved combined plot to {out_path}")

def plot_data_yearly():
    for name, path in csv_paths.items():
        path = Path(path)
        if not path.exists():
            print(f"Warning: file not found, skipping {name}: {path}")
            continue

        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.set_index("Date")

        # --- Compute yearly performance & volatility ---
        # 1) Compute cumulative annual return
        yearly_return = (1 + df["Return_t"]).resample("YE").prod() - 1

        # 2) Compute annualized volatility (std of returns * sqrt(252))
        yearly_vol = df["Return_t"].resample("YE").std() * (252 ** 0.5)

        # Combine into one DataFrame for plotting
        yearly_df = pd.DataFrame({
            "Return": yearly_return,
            "Volatility": yearly_vol
        })

        # --- Plot ---
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(yearly_df.index.year, yearly_df["Return"], label="Annual Return", marker="o")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Return", color="blue")
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))  # <-- as %
        ax1.tick_params(axis="y", labelcolor="blue")

        ax2 = ax1.twinx()
        ax2.bar(yearly_df.index.year, yearly_df["Volatility"], alpha=0.3, color="gray", label="Annual Volatility")
        ax2.set_ylabel("Volatility (annualized)", color="gray")
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))  # <-- as %
        ax2.tick_params(axis="y", labelcolor="gray")

        plt.title(f"{name} — Yearly Return & Volatility")
        fig.tight_layout()

        # Save
        fig.savefig(img_dir / f"{name}_yearly.png", dpi=300)
        plt.close()
        print(f"Saved yearly plot for {name}")

def plot_all_in_one_yearly():
    plt.figure(figsize=(14, 7), dpi=200)

    for name, path in csv_paths.items():
        path = Path(path)
        if not path.exists():
            print(f"Warning: file not found, skipping {name}: {path}")
            continue

        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.set_index("Date")

        # compute yearly return: compound daily returns within each year
        yearly_return = (1 + df["Return_t"]).resample("YE").prod() - 1

        plt.plot(yearly_return.index.year, yearly_return.values, label=name)

    plt.xlabel("Year")
    plt.ylabel("Annual Return")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))  # <-- as %
    plt.title("Annual Return by Index")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = img_dir / "all_indices_annual_return.png"
    plt.savefig(out_path)
    plt.close()
    print("Saved combined plot to", out_path)

def main():
    plot_data_daily()
    plot_all_in_one_daily()
    plot_data_yearly()
    plot_all_in_one_yearly()

if __name__ == "__main__":
    main()
