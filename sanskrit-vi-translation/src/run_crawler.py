import sys
from pathlib import Path
import csv
import argparse

# Add parent directory to path to allow importing modules
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

from crawlers.thuvienhoasen import ThuvienhoasenCrawler
from crawlers.budsas import BudsasCrawler


def save_crawled_data(data, output_file):
    file_exists = Path(output_file).exists()
    keys = data[0].keys() if data else []

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)
    print(f"Saved {len(data)} items to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Sanskrit-Vietnamese Data Crawlers"
    )
    parser.add_argument("--url", type=str, required=True, help="URL to crawl")
    parser.add_argument(
        "--source",
        type=str,
        default="budsas",
        choices=["thuvienhoasen", "budsas"],
        help="Source type",
    )

    args = parser.parse_args()

    crawler = None
    if args.source == "thuvienhoasen":
        crawler = ThuvienhoasenCrawler()
    elif args.source == "budsas":
        crawler = BudsasCrawler()

    if crawler:
        data = crawler.run(args.url)
        if data:
            # Save raw data for manual inspection/alignment
            output_path = current_dir.parent / "data" / "crawled_raw.csv"
            save_crawled_data(data, output_path)
    else:
        print("Invalid source specified.")


if __name__ == "__main__":
    main()
