import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric-curve-path",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--verification-levels-path",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    with arguments.metric_curve_path.open(
        encoding="utf-8",
        newline="",
    ) as input_file:
        metric_rows = list(csv.DictReader(input_file))

    with arguments.verification_levels_path.open(
        encoding="utf-8",
        newline="",
    ) as input_file:
        level_rows = list(csv.DictReader(input_file))

    thresholds = [float(row["threshold"]) for row in metric_rows]
    false_match_rates = [float(row["false_match_rate"]) for row in metric_rows]
    true_match_rates = [float(row["true_match_rate"]) for row in metric_rows]

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 8),
        sharex=True,
    )

    axes[0].plot(
        thresholds,
        false_match_rates,
        color="crimson",
        label="FMR",
    )
    axes[0].set_ylabel("False Match Rate")
    axes[0].set_yscale(
        "symlog",
        linthresh=0.0001,
    )
    axes[0].grid(
        alpha=0.3,
    )
    axes[0].legend()

    axes[1].plot(
        thresholds,
        true_match_rates,
        color="royalblue",
        label="TMR",
    )
    axes[1].set_xlabel("Similarity Threshold")
    axes[1].set_ylabel("True Match Rate")
    axes[1].grid(
        alpha=0.3,
    )
    axes[1].legend()

    level_colors = {
        "1": "green",
        "2": "orange",
        "3": "purple",
    }

    for level_row in level_rows:
        level = level_row["level"]
        level_threshold = float(level_row["threshold"])

        for axis in axes:
            axis.axvline(
                level_threshold,
                color=level_colors[level],
                linestyle="--",
                alpha=0.8,
                label=f"Level {level}",
            )

    axes[0].legend()
    axes[1].legend()

    figure.suptitle("Face Verification FMR/TMR by Threshold")
    figure.tight_layout()

    arguments.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        arguments.output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    fmr_tmr_figure, fmr_tmr_axis = plt.subplots(
        figsize=(9, 6),
    )

    fmr_tmr_axis.plot(
        false_match_rates,
        true_match_rates,
        color="royalblue",
        linewidth=2,
        label="FMR/TMR curve",
    )

    for level_row in level_rows:
        level = level_row["level"]
        level_fmr = float(level_row["observed_false_match_rate"])
        level_tmr = float(level_row["true_match_rate"])

        fmr_tmr_axis.scatter(
            level_fmr,
            level_tmr,
            color=level_colors[level],
            s=70,
            zorder=3,
            label=f"Level {level}",
        )

        fmr_tmr_axis.annotate(
            f"Level {level}",
            xy=(level_fmr, level_tmr),
            xytext=(8, -15),
            textcoords="offset points",
        )

    fmr_tmr_axis.set_xscale(
        "symlog",
        linthresh=0.0001,
    )
    fmr_tmr_axis.set_xlabel("False Match Rate (FMR)")
    fmr_tmr_axis.set_ylabel("True Match Rate (TMR)")
    fmr_tmr_axis.set_title("TMR by FMR")
    fmr_tmr_axis.grid(
        alpha=0.3,
    )
    fmr_tmr_axis.legend()

    fmr_tmr_figure.tight_layout()

    fmr_tmr_output_path = arguments.output_path.parent / "tmr_by_fmr.png"

    fmr_tmr_figure.savefig(
        fmr_tmr_output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fmr_tmr_figure)

    print(f"Chart saved to: {arguments.output_path}")
    print(f"FMR/TMR chart saved to: {fmr_tmr_output_path}")


if __name__ == "__main__":
    main()
