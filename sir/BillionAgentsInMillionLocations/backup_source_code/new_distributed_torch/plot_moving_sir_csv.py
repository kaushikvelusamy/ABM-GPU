import argparse

from moving_sir_plotting import plot_summary


def main():
    parser = argparse.ArgumentParser(description="Plot distributed moving-agent SIR summary CSV to a PNG file")
    parser.add_argument("--csv", type=str, required=True, help="Summary CSV path")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path; default uses the summary CSV filename")
    parser.add_argument("--title", type=str, default="Distributed Moving-Agent SIR", help="Plot title")
    parser.add_argument("--x-label", type=str, default="Day", help="Horizontal-axis label; use Day if one timestep represents one day")
    parser.add_argument("--config", type=str, default=None, help="Optional run-config JSON path for parameter annotation; auto-detected when omitted")
    args = parser.parse_args()

    saved = plot_summary(
        args.csv,
        args.out,
        title=args.title,
        x_label=args.x_label,
        config_json=args.config,
    )
    print(f"saved {saved}")


if __name__ == "__main__":
    main()
