#!/usr/bin/env python3
"""Plot per-species rank positions and residual histograms from PSEA debug TSVs.

Requires pandas, numpy and matplotlib. No QIIME 2 or R dependencies.
"""
import argparse
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def read_tsv(path):
    # Preserve identifiers such as 00123 and NA verbatim.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def require_columns(table, columns, label):
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")


def load_data(input_path, ranked_path=None):
    table = read_tsv(input_path)
    require_columns(table, ["term", "gene", "deltaZ", "in_gene_list"], "Input")
    flags = table.in_gene_list.str.strip().str.lower()
    if not flags.isin(["true", "false", "1", "0"]).all():
        raise ValueError("in_gene_list must contain TRUE/FALSE or 1/0")
    table["in_gene_list"] = flags.isin(["true", "1"])
    table["deltaZ"] = pd.to_numeric(table.deltaZ, errors="coerce")
    selected = table[table.in_gene_list].copy()
    if not np.isfinite(selected.deltaZ).all():
        raise ValueError("Included genes must have finite deltaZ scores")
    if (selected.groupby("gene").deltaZ.nunique() > 1).any():
        raise ValueError("The same gene has conflicting deltaZ values")

    if ranked_path is None:
        sibling = input_path.with_name(
            input_path.name.removesuffix("_gsea_input.tsv") + "_ranked_gene_list.tsv"
        )
        if sibling.is_file():
            ranked_path = sibling
    if ranked_path is not None:
        ranked = read_tsv(ranked_path)
        require_columns(ranked, ["gene", "score", "rank"], "Ranked list")
        ranked["score"] = pd.to_numeric(ranked.score, errors="raise")
        ranked["rank"] = pd.to_numeric(ranked["rank"], errors="raise")
        if ranked.gene.duplicated().any():
            raise ValueError("Ranked list contains duplicate gene IDs")
        if not np.isfinite(ranked[["score", "rank"]].to_numpy()).all():
            raise ValueError("Ranked list contains nonfinite values")
        ranked = ranked.sort_values("rank")
        if not np.array_equal(ranked["rank"].to_numpy(), np.arange(1, len(ranked) + 1)):
            raise ValueError("Ranked list ranks must be consecutive integers starting at 1")
        source = str(ranked_path)
    else:
        warnings.warn(
            "No companion ranked list found. Reconstructing from included genes in "
            "the input TSV; background genes absent from its memberships cannot be plotted."
        )
        ranked = selected.drop_duplicates("gene").copy()
        if "gene_list_rank" in ranked:
            ranked["rank"] = pd.to_numeric(ranked.gene_list_rank, errors="raise")
            if (not np.isfinite(ranked["rank"]).all()
                    or (ranked["rank"] < 1).any()
                    or (ranked["rank"] % 1 != 0).any()
                    or ranked["rank"].duplicated().any()):
                raise ValueError("Input contains invalid or conflicting ranks")
            ranked = ranked.sort_values("rank")
        else:
            ranked = ranked.sort_values("deltaZ", ascending=False, kind="stable")
            ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked = ranked.rename(columns={"deltaZ": "score"})[["gene", "score", "rank"]]
        source = "Reconstructed background (input TSV only)"
    if (np.diff(ranked.score.to_numpy()) > 0).any():
        raise ValueError("Scores must be descending in rank order")
    lookup = ranked.set_index("gene")
    if not selected.gene.isin(lookup.index).all():
        raise ValueError("Some included genes are absent from the supplied ranked list")
    if not np.allclose(selected.deltaZ, selected.gene.map(lookup.score)):
        raise ValueError("Ranked list scores disagree with the input TSV")
    return table, ranked, source


def plot_species(table, ranked, term, bins=30, status=None):
    genes = table.loc[(table.term == term) & table.in_gene_list, "gene"].unique()
    members = ranked[ranked.gene.isin(genes)]
    color = "#e41a1c" if status == "missing" else "#377eb8"
    fig, (ax, hist) = plt.subplots(2, 1, figsize=(11, 8.5), layout="constrained")
    label = f"Species {term}" + (f" ({status})" if status else "")
    ax.plot(ranked["rank"], ranked.score, color="0.55", linewidth=1)
    ax.axhline(0, linestyle="--", color="0.65", linewidth=0.8)
    ax.scatter(members["rank"], members.score, color=color, alpha=0.45, s=16)
    ax.plot(members["rank"], np.full(len(members), 0.02), "|", color=color,
            transform=ax.get_xaxis_transform(), markersize=9)
    ax.set(title=f"Rank positions — {label}", xlabel="Rank in gene_list", ylabel="deltaZ")
    if len(members):
        hist.hist(members.score, bins=bins, color=color, edgecolor="white")
        fraction = f"{(members.score > 0).mean():.3f}"
    else:
        hist.text(.5, .5, "No peptides in the ranked list", ha="center",
                  transform=hist.transAxes)
        fraction = "N/A"
    hist.axvline(0, linestyle="--", color="0.3", linewidth=0.8)
    hist.set(title=f"{label} | n = {len(members)} | positive fraction = {fraction}",
             xlabel="deltaZ for this species", ylabel="Peptide / epitope count")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="*_gsea_input.tsv")
    parser.add_argument("--output", type=Path, required=True, help="Multi-page PDF path")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--species", nargs="+", help="Species IDs; space or comma separated")
    selection.add_argument("--top", type=int, help="Top N by adjusted p-value; requires --results")
    parser.add_argument("--results", type=Path, help="Matching *_gsea_output.tsv (optional)")
    parser.add_argument("--ranked-list", type=Path, help="Exact *_ranked_gene_list.tsv; auto-detected if adjacent")
    parser.add_argument("--png-dir", type=Path, help="Also save each species as a PNG")
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args(argv)
    if args.bins < 1 or (args.top is not None and args.top < 1):
        parser.error("--bins and --top must be positive")
    if args.top and not args.results:
        parser.error("--top requires --results; significance is not encoded in an input TSV")
    table, ranked, source = load_data(args.input, args.ranked_list)
    results = None
    if args.results:
        results = read_tsv(args.results)
        require_columns(results, ["ID"], "Results")
    available = list(table.term.unique())
    if args.species:
        terms = list(dict.fromkeys(t.strip() for item in args.species for t in item.split(",") if t.strip()))
        missing = set(terms) - set(available)
        if missing:
            parser.error(f"Species IDs absent from input: {', '.join(sorted(missing))}")
    elif args.top:
        require_columns(results, ["p.adjust", "pvalue"], "Results")
        for col in ["p.adjust", "pvalue"]:
            results[col] = pd.to_numeric(results[col], errors="coerce")
        eligible = results[np.isfinite(results["p.adjust"]) & results.ID.isin(available)]
        terms = eligible.sort_values(["p.adjust", "pvalue", "ID"]).ID.drop_duplicates().head(args.top).tolist()
    else:
        terms = available
    if not terms:
        parser.error("No species available for plotting")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.png_dir:
        args.png_dir.mkdir(parents=True, exist_ok=True)
    returned = set(results.ID) if results is not None else None
    from urllib.parse import quote
    with PdfPages(args.output) as pdf:
        pdf.infodict()["Subject"] = f"Input: {args.input}; ranked background: {source}"
        for term in terms:
            status = None if returned is None else ("returned" if term in returned else "missing")
            fig = plot_species(table, ranked, term, args.bins, status)
            pdf.savefig(fig)
            if args.png_dir:
                fig.savefig(args.png_dir / f"species_{quote(term, safe='')}.png", dpi=180)
            plt.close(fig)
    print(f"Wrote {len(terms)} species pages to {args.output}\nRanked background: {source}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(f"Error: {error}")
