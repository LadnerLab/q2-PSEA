import altair as alt
import math
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import qiime2


def volcano(
    output_dir: str,
    pairs: pd.DataFrame,
    psea_tables: pd.DataFrame = None,
    colors_file: qiime2.Metadata = None,
    x: list = None,
    y: list = None,
    taxa: list = None,
    xy_access: list = ["x", "y"],
    taxa_access: str = None,
    x_threshold: float = 0.4,
    y_threshold: float = 0.05,
    log: bool = True,
    xy_labels: list = ["x", "y"],
 ) -> None:
    alt.data_transformers.disable_max_rows()

    if psea_tables is not None:
        x = []
        y = []
        taxa = []
        for _, table_df in sorted(psea_tables.items()):
            x.append(table_df.loc[:, xy_access[0]].to_list())
            y.append(table_df.loc[:, xy_access[1]].to_list())
            if taxa_access:
                taxa.append(table_df.loc[:, taxa_access].to_list())
    elif x and y:
        x = [x]
        y = [y]
        if not taxa:
            taxa = []
        else:
            taxa = [taxa]

    pair_strs = pairs.apply(
        lambda row: f"{row.iloc[0]}~{row.iloc[1]}", axis=1
    )
    unsorted_pairs = pair_strs.tolist()
    pair_2_title = dict(zip(
        pair_strs,
        pairs.iloc[:, 2].astype(str) if len(pairs.columns) >= 3
        else [""] * len(pair_strs),
    ))

    pairs_list = sorted(unsorted_pairs)

    sample_dropdown = alt.binding_select(
        options=unsorted_pairs, name="Sample Select"
    )
    sample_select = alt.selection_point(
        fields=["pair"],
        bind=sample_dropdown,
        name="pair",
        value=[{"pair": unsorted_pairs[0]}],
    )

    volcano_dict = {"x": list(), "y": list(), "pair": list()}

    charting_ys = []
    for i in range(len(x)):
        if log:
            charting_ys.append(-np.log10(y[i]))
            sort = "ascending"
        else:
            charting_ys.append(y[i])
            sort = "descending"
        volcano_dict["x"].extend(x[i])
        volcano_dict["y"].extend(charting_ys[i])
        volcano_dict["pair"].extend([pairs_list[i]] * len(x[i]))
    volcano_df = pd.DataFrame(volcano_dict)

    volcano_chart = alt.Chart(volcano_df).mark_circle(
        size=50, color="black"
    ).encode(
        x=alt.X("x:Q", title=xy_labels[0]),
        y=alt.Y("y:Q", title=xy_labels[1], sort=sort),
    ).add_params(
        sample_select
    ).transform_filter(
        sample_select
    )
    final_chart = alt.layer(volcano_chart)

    highlight_df = None
    if taxa:
        highlight_dict = {
            "x": list(), "y": list(),
            "taxa": list(), "pair": list(),
        }
        for i in range(len(taxa)):
            for j in range(len(taxa[i])):
                if y[i][j] < y_threshold and abs(x[i][j]) > x_threshold:
                    highlight_dict["x"].append(x[i][j])
                    highlight_dict["y"].append(charting_ys[i][j])
                    highlight_dict["taxa"].append(taxa[i][j])
                    highlight_dict["pair"].append(pairs_list[i])
        highlight_df = pd.DataFrame(highlight_dict)

        color_scale = alt.Scale(range=[
            "#E69F00", "#56B4E9", "#009E73",
            "#F0E442", "#0072B2", "#D55E00",
            "#CC79A7",
        ])
        legend = alt.Legend(title="Significant Taxa")
        if colors_file is not None:
            all_species = list(set(highlight_df["taxa"].to_list()))
            num_extra_colors = len(all_species)

            color_df = colors_file.to_dataframe()
            color_df = color_df[color_df.index.isin(all_species)]
            species_list = color_df.index.to_list()
            colors_list = color_df.iloc[:, 0].to_list()
            num_extra_colors -= len(species_list)

            color_iter = iter(
                plt.cm.rainbow(np.linspace(0, 1, num_extra_colors))
            )

            for species in all_species:
                if species not in species_list:
                    species_list.append(species)
                    colors_list.append(clr.to_hex(next(color_iter)))

            color_scale = alt.Scale(domain=species_list, range=colors_list)
            legend = alt.Legend(
                title="Significant Taxa",
                columns=int(math.ceil(len(species_list) / 30)),
                symbolLimit=0,
            )

        highlight_chart = alt.Chart(highlight_df).mark_circle(
            size=60, filled=True, opacity=1.0
        ).encode(
            x=alt.X("x:Q", title=xy_labels[0]),
            y=alt.Y("y:Q", title=xy_labels[1], sort=sort),
            color=alt.Color("taxa:N", scale=color_scale, legend=legend),
            tooltip="taxa",
        ).add_params(
            sample_select
        ).transform_filter(
            sample_select
        )
        final_chart = alt.layer(final_chart, highlight_chart).resolve_scale(
            color="independent"
        )

    titleDf = pd.DataFrame(pair_2_title.items(), columns=["pair", "title"])
    title = alt.Chart(titleDf).mark_text(size=25, dx=150).encode(
        text="title:N"
    ).transform_filter(
        sample_select
    )
    final_chart = alt.vconcat(title, final_chart)

    final_chart.save(os.path.join(output_dir, "index.html"))


def zscatter(
    output_dir: str,
    zscores: pd.DataFrame,
    pairs: pd.DataFrame,
    psea_tables: pd.DataFrame = None,
    splines: pd.DataFrame = None,
    colors_file: qiime2.Metadata = None,
    p_val_access: str = None,
    le_peps_access: str = None,
    taxa_access: str = None,
    highlight_threshold: float = 0.05,
) -> None:
    alt.data_transformers.disable_max_rows()

    if psea_tables is not None:
        assert p_val_access, (
            "'psea_tables' was provided, but nothing was provided for"
            " 'p_val_access'!"
        )
        assert le_peps_access, (
            "'psea_tables' was provided, but nothing was provided for"
            " 'le_peps_access'!"
        )
        assert taxa_access, (
            "'psea_tables' was provided, but nothing was provided for"
            " 'taxa_access'!"
        )

    pair_strs = pairs.apply(
        lambda row: f"{row.iloc[0]}~{row.iloc[1]}", axis=1
    )
    unsorted_pairs = pair_strs.tolist()
    pair_2_title = dict(zip(
        pair_strs,
        pairs.iloc[:, 2].astype(str) if len(pairs.columns) >= 3
        else [""] * len(pair_strs),
    ))

    pairs_list = sorted(unsorted_pairs)

    sample_dropdown = alt.binding_select(
        options=unsorted_pairs, name="Sample Select"
    )
    sample_select = alt.selection_point(
        fields=["pair"],
        bind=sample_dropdown,
        name="pair",
        value=[{"pair": unsorted_pairs[0]}],
    )

    heatmap_dict = {
        "bin_x_start": list(), "bin_x_end": list(),
        "bin_y_start": list(), "bin_y_end": list(),
        "count": list(), "pair": list(),
    }
    p = 0
    for pair in pairs_list:
        pair = pair.split("~")

        x = zscores.loc[pair[0]]
        y = zscores.loc[pair[1]]

        heatmap, x_edges, y_edges = np.histogram2d(x, y, bins=(70, 70))
        for x in range(0, heatmap.shape[0]):
            for y in range(0, heatmap.shape[1]):
                count = heatmap[x, y]
                if count == 0.0:
                    continue
                bin_x_start = x_edges[x]
                bin_x_end = x_edges[x + 1]
                bin_y_start = y_edges[y]
                bin_y_end = y_edges[y + 1]

                heatmap_dict["bin_x_start"].append(bin_x_start)
                heatmap_dict["bin_x_end"].append(bin_x_end)
                heatmap_dict["bin_y_start"].append(bin_y_start)
                heatmap_dict["bin_y_end"].append(bin_y_end)
                heatmap_dict["count"].append(count)
                heatmap_dict["pair"].append(pairs_list[p])
        p += 1
    heatmap_df = pd.DataFrame(heatmap_dict)
    xy_max = heatmap_df.loc[:, ["bin_x_end", "bin_y_end"]].max()
    ratio = (xy_max.iloc[0] / xy_max.iloc[1]) + 1
    chart_height = 500
    chart_width = chart_height + (20 * ratio)

    highlight_df = None
    if psea_tables is not None:
        highlight_dict = {
            "x": list(), "y": list(),
            "peptide": list(), "taxa": list(), "pair": list(),
        }

        for pair_str, table_df in sorted(psea_tables.items()):
            pair = pair_str.split("~")
            rows = table_df.loc[:, [p_val_access, le_peps_access, taxa_access]]
            p_vals = rows.iloc[:, 0].to_list()

            for i in range(len(p_vals)):
                if p_vals[i] < highlight_threshold:
                    le_peps = rows.iloc[i, 1].split("/")
                    sig_taxa = rows.iloc[i, 2]
                    for le_pep in le_peps:
                        highlight_dict["x"].append(
                            zscores.loc[pair[0], le_pep]
                        )
                        highlight_dict["y"].append(
                            zscores.loc[pair[1], le_pep]
                        )
                        highlight_dict["peptide"].append(le_pep)
                        highlight_dict["taxa"].append(sig_taxa)
                        highlight_dict["pair"].append(pair_str)
        highlight_df = pd.DataFrame(highlight_dict)

    heatmap_chart = alt.Chart(
        heatmap_df, width=chart_width, height=chart_height
    ).mark_rect().encode(
        alt.X("bin_x_start:Q", title="Time Point 1"),
        alt.X2("bin_x_end:Q"),
        alt.Y("bin_y_start:Q", title="Time Point 2"),
        alt.Y2("bin_y_end:Q"),
        alt.Color(
            "count:Q",
            scale=alt.Scale(scheme="greys"),
            legend=alt.Legend(title="Point Frequency"),
        ),
    ).add_params(
        sample_select
    ).transform_filter(
        sample_select
    )
    final_chart = alt.layer(heatmap_chart)

    if splines:
        for pair, spline_df in splines.items():
            spline_df.dropna(inplace=True)
            spline_df["pair"] = pair
            spline_chart = alt.Chart(spline_df).mark_square(size=20).encode(
                x=alt.X("x:Q"),
                y=alt.Y("yfit:Q"),
                color=alt.Color(
                    "x:N",
                    scale=alt.Scale(range=["#FF0000"]),
                    legend=None,
                ),
            ).transform_filter(
                sample_select
            )
            final_chart = alt.layer(final_chart, spline_chart)

    color_scale = alt.Scale(range=[
        "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00",
        "#CC79A7",
    ])
    shape = alt.Shape("taxa:N", legend=None)
    legend = alt.Legend(title="Significant Taxa")

    if colors_file is not None:
        all_species = list(set(highlight_df["taxa"].to_list()))
        num_extra_colors = len(all_species)

        color_df = colors_file.to_dataframe()
        color_df = color_df[color_df.index.isin(all_species)]
        species_list = color_df.index.to_list()
        colors_list = color_df.iloc[:, 0].to_list()
        num_extra_colors -= len(species_list)

        color_iter = iter(
            plt.cm.rainbow(np.linspace(0, 1, num_extra_colors))
        )

        for species in all_species:
            if species not in species_list:
                species_list.append(species)
                colors_list.append(clr.to_hex(next(color_iter)))

        color_scale = alt.Scale(domain=species_list, range=colors_list)
        shape = alt.Shape(
            "taxa:N",
            scale=alt.Scale(domain=species_list),
            legend=None,
        )
        legend = alt.Legend(
            title="Significant Taxa",
            columns=int(math.ceil(len(species_list) / 30)),
            symbolLimit=0,
        )

    if highlight_df is not None:
        highlight_chart = alt.Chart(highlight_df).mark_point(
            filled=True, size=60
        ).encode(
            x=alt.X("x:Q"),
            y=alt.Y("y:Q"),
            color=alt.Color("taxa:N", scale=color_scale, legend=legend),
            shape=shape,
            tooltip=["peptide", "taxa"],
        ).transform_filter(
            sample_select
        )
        final_chart = alt.layer(final_chart, highlight_chart).resolve_scale(
            color="independent",
            shape="independent",
        )

    titleDf = pd.DataFrame(pair_2_title.items(), columns=["pair", "title"])
    title = alt.Chart(titleDf).mark_text(
        size=30, dx=chart_width / 2
    ).encode(
        text="title:N"
    ).transform_filter(
        sample_select
    )
    final_chart = alt.vconcat(title, final_chart)

    final_chart.save(os.path.join(output_dir, "index.html"))


def aeplots(
    output_dir: str,
    pos_ae_counts: pd.DataFrame,
    neg_ae_counts: pd.DataFrame,
    colors_file: qiime2.Metadata = None,
    xy_access: list = ["Events", "Species"],
    xy_labels: list = ["Number of AEs in cohort", "Species"],
) -> None:
    alt.data_transformers.disable_max_rows()

    pos_df = pos_ae_counts.copy()
    neg_df = neg_ae_counts.copy()

    pos_df["NES"] = "Positive"
    neg_df["NES"] = "Negative"

    ae_df = pd.concat([pos_df, neg_df])

    color_scale = alt.Scale(range=[
        "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00",
        "#CC79A7",
    ])
    if colors_file is not None:
        all_species = list(set(ae_df["Species"].to_list()))
        num_extra_colors = len(all_species)

        color_df = colors_file.to_dataframe()
        color_df = color_df[color_df.index.isin(all_species)]
        species_list = color_df.index.to_list()
        colors_list = color_df.iloc[:, 0].to_list()
        num_extra_colors -= len(species_list)

        color_iter = iter(
            plt.cm.rainbow(np.linspace(0, 1, num_extra_colors))
        )

        for species in all_species:
            if species not in species_list:
                species_list.append(species)
                colors_list.append(clr.to_hex(next(color_iter)))

        color_scale = alt.Scale(domain=species_list, range=colors_list)

    bar_chart = alt.Chart(ae_df).mark_bar().encode(
        alt.X(f"{xy_access[0]}:Q", title=xy_labels[0]),
        alt.Y(
            f"{xy_access[1]}:N",
            axis=alt.Axis(grid=True),
            sort=alt.EncodingSortField(
                field=xy_access[1], op="count", order="ascending"
            ),
            title=None,
        ),
        alt.Color(f"{xy_access[1]}:N", scale=color_scale, legend=None),
    )

    text = bar_chart.mark_text(
        align="left", baseline="middle", dx=3
    ).encode(
        text=f"{xy_access[0]}:Q"
    )

    final_chart = (bar_chart + text).facet(
        column=alt.Column(
            "NES:N",
            sort=alt.SortField(field="NES", order="descending"),
            title=None,
            header=alt.Header(labelFontSize=15, labelFontWeight="bold"),
        )
    ).resolve_scale(y="independent")

    final_chart.save(os.path.join(output_dir, "index.html"))
