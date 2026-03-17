"""Generate an interactive HTML report from weight analysis JSON."""

import argparse
import json
from pathlib import Path


def build_report(data_paths: list[str], output_path: str):
    """Read analysis JSON files and generate HTML report.

    Args:
        data_paths: List of paths to JSON files from analyze_weights.py.
        output_path: Path to write the HTML report.
    """
    datasets = []
    for p in data_paths:
        with open(p) as f:
            datasets.append(json.load(f))

    # Load template
    template_path = Path(__file__).parent / "templates" / "report_template.html"
    template = template_path.read_text()

    # Embed data as JSON
    data_json = json.dumps(datasets)
    html = template.replace("DATA_PLACEHOLDER", data_json)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Report written to {output_path}")


def main():
    p = argparse.ArgumentParser(description="Generate HTML report from analysis JSON")
    p.add_argument("--data", nargs="+", required=True, help="Path(s) to analysis JSON files")
    p.add_argument("--output", default="analysis_results/report.html")
    args = p.parse_args()
    build_report(args.data, args.output)


if __name__ == "__main__":
    main()
