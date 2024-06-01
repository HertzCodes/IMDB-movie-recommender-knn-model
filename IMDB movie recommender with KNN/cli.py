import time

import typer
import webscraper
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print


def scrape():
    print('[bold red]Started![/bold red] Sending requests to IMDB!')
    webscraper.scrape_movie_urls()
    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
    ) as progress:
        progress.add_task(description="Processing...", total=None)
        webscraper.scrape_movie_data()
    print('Done!')


if __name__ == "__main__":
    typer.run(scrape)