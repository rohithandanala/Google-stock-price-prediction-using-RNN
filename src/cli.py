"""Console script for time_series_forecasting_of_stock_prices_using_deep_learning."""
import time_series_forecasting_of_stock_prices_using_deep_learning

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for time_series_forecasting_of_stock_prices_using_deep_learning."""
    console.print("Replace this message by putting your code into "
               "time_series_forecasting_of_stock_prices_using_deep_learning.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
