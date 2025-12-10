import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

# Base folder for all data relative to this script (src/data_processing/data)
BASE_DATA_DIR = Path(__file__).resolve().parent / "data"

# Index tickers
TICKERS = {
    "SSE": "000001.SS",
    "KOSPI": "^KS11",
    "TAIEX": "^TWII",
   # "PSEI": "^PSEI",
    "STOXX600": "^STOXX",
    "NIKKEI225": "^N225"
}

# Output folder for raw index CSVs: src/data_processing/data/indices
OUTPUT_DIR = BASE_DATA_DIR / "indices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Timespan of collected data ---
end_date = datetime.today()
start_date = "2015-01-01"


def download_index(ticker, start, end):
    """Download daily index data."""
    return yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        progress=False
    )

def load_data():
    for name, ticker in TICKERS.items():
        print(f"Downloading {name} ({ticker})...")
        df = download_index(ticker, start_date, end_date)

        print(f"{name}: downloaded shape = {df.shape}")  # ...existing code...

        # Add index name column for clarity
        df["Index"] = name

        # Save as individual CSV (write index as a "Date" column)
        file_path = OUTPUT_DIR / f"{name}_20yr_daily.csv"
        df.reset_index().to_csv(file_path, index=False)
        print(f"Saved {file_path}")

    print("All downloads complete.")


def build_clean_datasets(csv_paths):
    """
    csv_paths: dict like
        {
            "CSI300": "data/indices/CSI300_20yr_daily.csv",
            "KOSPI": "data/indices/KOSPI_20yr_daily.csv",
            ...
        }

    Returns:
        dict: {name: DataFrame_with_features}
    """

    # 1) Load all price series, one per index
    price_series = {}  # {name: Series of prices}

    print("\n--- Loading raw CSVs and extracting price columns ---")
    for name, path in csv_paths.items():
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.set_index("Date")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]  # drop duplicate dates

        if df.empty:
            print(f"WARNING: {name} ({path}) is EMPTY!")
            continue

        # choose price series: prefer Adj Close, fallback to Close
        # Coerce to numeric to avoid strings causing arithmetic errors
        if "Adj Close" in df.columns:
            s = pd.to_numeric(df["Adj Close"], errors="coerce").rename(name)
        else:
            s = pd.to_numeric(df["Close"], errors="coerce").rename(name)

        # If the entire series is non-numeric after coercion, skip it
        if s.dropna().empty:
            print(f"WARNING: {name} has no numeric price data after coercion, skipping.")
            continue

        print(f"{name}: shape = {df.shape}, dates {df.index.min()} -> {df.index.max()}")
        price_series[name] = s

    # If any are missing, that will reduce the common date set
    if len(price_series) == 0:
        raise ValueError("No non-empty indices loaded. Check your downloads/tickers.")

    # 2) Combine into one DataFrame of prices, keeping only dates
    #    where ALL indices have data (inner join on index)
    prices = pd.concat(price_series.values(), axis=1, join="inner")
    prices = prices.sort_index()

    # Drop rows with any NaN so all remaining rows are numeric for every index.
    # This prevents pct_change and other numeric ops from encountering strings/NaNs.
    before_rows = len(prices)
    prices = prices.dropna(how="any")
    after_rows = len(prices)
    if after_rows != before_rows:
        print(f"Dropped {before_rows - after_rows} rows with missing/non-numeric prices. New shape: {prices.shape}")

    print(f"\nCombined price table shape: {prices.shape}")
    print(f"Common dates: {prices.index.min()} -> {prices.index.max()}")

    if prices.empty:
        raise ValueError(
            "No common dates across all indices. "
            "Check that all CSVs cover overlapping time periods."
        )

    # 3) For each index (each column), build a feature DataFrame
    result = {}

    for name in prices.columns:
        price = prices[name]

        # daily returns
        returns = price.pct_change()

        # build feature DataFrame
        feat = pd.DataFrame(index=prices.index)

        feat["Return_t"] = returns

        # lagged returns
        feat["Return_t_1"] = returns.shift(1)
        feat["Return_t_2"] = returns.shift(2)
        feat["Return_t_3"] = returns.shift(3)

        # moving averages of returns (using full windows)
        feat["SMA_5"] = returns.rolling(window=5).mean()
        feat["SMA_10"] = returns.rolling(window=10).mean()

        # rolling std of returns
        feat["STD_5"] = returns.rolling(window=5).std()

        # Drop rows where any feature is NaN (initial window rows)
        feat = feat.dropna()

        print(f"{name}: feature df shape = {feat.shape}")
        result[name] = feat

    return result



# Input CSV paths for feature building (point to src/data_processing/data/indices)
csv_paths = {
    "SSE": str(OUTPUT_DIR / "SSE_20yr_daily.csv"),
    "KOSPI": str(OUTPUT_DIR / "KOSPI_20yr_daily.csv"),
    "TAIEX": str(OUTPUT_DIR / "TAIEX_20yr_daily.csv"),
    #"PSEI": str(OUTPUT_DIR / "PSEI_20yr_daily.csv"),
    "STOXX600": str(OUTPUT_DIR / "STOXX600_20yr_daily.csv"),
    "NIKKEI225": str(OUTPUT_DIR / "NIKKEI225_20yr_daily.csv"),
}


def main():
    load_data()

    datasets = build_clean_datasets(csv_paths)

    # Output clean features under src/data_processing/data/clean_features
    out_dir = BASE_DATA_DIR / "clean_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save each DataFrame
    for name, df in datasets.items():
        out_path = out_dir / f"{name}_features.csv"
        df.reset_index().to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


