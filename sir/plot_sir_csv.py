import argparse
import csv

import matplotlib.pyplot as plt


def read_sir_csv(path):
    steps = []
    susceptible = []
    infected = []
    recovered = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"step", "susceptible", "infected", "recovered"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"{path} must have CSV header: step,susceptible,infected,recovered"
            )

        for row in reader:
            steps.append(int(float(row["step"])))
            susceptible.append(float(row["susceptible"]))
            infected.append(float(row["infected"]))
            recovered.append(float(row["recovered"]))

    return steps, susceptible, infected, recovered


def plot_sir_csv(csv_path, out_png, title="SIR curve from CSV"):
    steps, s, i, r = read_sir_csv(csv_path)

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, s, label="Susceptible")
    plt.plot(steps, i, label="Infected")
    plt.plot(steps, r, label="Recovered")
    plt.xlabel("Step")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"saved {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot SIR CSV (step,susceptible,infected,recovered) to PNG"
    )
    parser.add_argument("--csv", type=str, required=True, help="Input CSV path")
    parser.add_argument("--out", type=str, default="sir_from_csv.png", help="Output PNG path")
    parser.add_argument("--title", type=str, default="SIR curve from CSV", help="Plot title")
    args = parser.parse_args()

    plot_sir_csv(csv_path=args.csv, out_png=args.out, title=args.title)


if __name__ == "__main__":
    main()
