from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go

import numpy as np
from numpy.ma.core import size

from src.logs import logs
from src.Config import PROJECT_ROOT, COLOR_MAP
from src._T_typing import (
    _T_label,
    _T_landmark63,
    _T_landmarkBatch63,
    _T_landmark21xyz,
    _T_landmarkBatch21xyz,
    _T_labelBatch,
    _T_xyz,
    _T_xyzBatch,
    _T_cmap_colorHex,
    _T__converter__,
    _T_rel_path,
)

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# ============================================================
# Plot elements
# ============================================================

from abc import ABC, abstractmethod

class PlotElement(ABC):
    @abstractmethod
    def display(  self, ax: plt.Axes, zorder:int ) -> None: ...

class PE_Tag(PlotElement):
    def __init__(
        self, pt: _T_xyz, *, s: float = 3, marker: str = "o",
        color: str = "black", alpha: float = 1.0,
        linewidths:float = 1.0,
        label: str = "_tag"
    ) -> None:
        self.x = pt[0]
        self.y = pt[1]

        self.s = s
        self.marker = marker
        self.color = color
        self.alpha = alpha
        self.linewidths = linewidths
        self.label = label

    def display(self, ax: plt.Axes, zorder:int = 1) -> None:
        ax.scatter(
            self.x, self.y,
            s=self.s, marker=self.marker,
            color=self.color, alpha=self.alpha,
            linewidths=self.linewidths,
            zorder=zorder,
            label=self.label,
        )

class PE_Points(PlotElement):
    def __init__(
        self, points: _T_xyzBatch, *, s: float = 3, marker: str = "o",
        color: str = "black", alpha: float = 1.0,
        linewidths: float = 1.0,
        label: str = "_tag"
    ) -> None:
        self.points = points
        self.s = s
        self.marker = marker
        self.color = color
        self.alpha = alpha
        self.linewidths = linewidths
        self.label = label
    def display(self, ax: plt.Axes, zorder: int = 1) -> None:
        ax.scatter(
            self.points[:, 0], self.points[:, 1],
            s=self.s, marker=self.marker,
            color=self.color, alpha=self.alpha,
            linewidths=self.linewidths,
            zorder=zorder,
            label=self.label,
        )

class PE_TextLabel(PlotElement):
    def __init__(
        self, pt: _T_xyz, text: str, *,
        color: str = "black", fontsize: float = 10,
        fontweight: str = "normal", ha: str = "center",
        va: str = "center", alpha: float = 1.0,
        label ="_tag"
    ) -> None:
        self.x = pt[0]
        self.y = pt[1]

        self.text = text
        self.color = color
        self.fontsize = fontsize
        self.fontweight = fontweight
        self.ha = ha
        self.va = va
        self.alpha = alpha
        self.label = label

    def display(self, ax: plt.Axes, zorder:int) -> None:
        ax.text(
            self.x, self.y, self.text,
            color=self.color, fontsize=self.fontsize,
            fontweight=self.fontweight, ha=self.ha,
            va=self.va, alpha=self.alpha,
            zorder=zorder,
            label = self.label,
        )

class PE_Landmark(PlotElement):
    DEFAULT_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]

    def __init__(self, landmark: _T_landmark63, *,
            point_size: float = 20,
            point_marker: str = ".",
            point_color: str = "#FF0000",
            point_alpha: float = 1.0,
            point_overrides = None,
            connection_color: str = "#00FF00",
            connection_linewidth: float = 1.0,
            connection_alpha: float = 1.0,
            connection_overrides = None,
            label = "_",
        ) -> None:

        self.landmark: _T_landmark21xyz = _T__converter__.conv_landmark63_to_landmark21xyz(landmark)

        self.connections = self.DEFAULT_CONNECTIONS

        self.point_size = point_size
        self.point_marker = point_marker
        self.point_color = point_color
        self.point_alpha = point_alpha
        self.point_overrides = point_overrides or {}

        self.connection_color = connection_color
        self.connection_linewidth = connection_linewidth
        self.connection_alpha = connection_alpha
        self.connection_overrides = connection_overrides or {}

        self.label = label

    def display(self, ax: plt.Axes, zorder) -> None:
        for i, connection in enumerate(self.connections):
            idx_a, idx_b = connection
            pt_a, pt_b = self.landmark[idx_a], self.landmark[idx_b]
            kwargs = {"color": self.connection_color, "linewidth": self.connection_linewidth,
                      "alpha": self.connection_alpha, **self.connection_overrides.get(connection, {})}
            ax.plot([pt_a[0], pt_b[0]], [pt_a[1], pt_b[1]], zorder=zorder, label="{} landmark".format(self.label) if i == 0 else None,
                    **kwargs)
        for idx, pt in enumerate(self.landmark):
            kwargs = {"s": self.point_size, "marker": self.point_marker, "color": self.point_color,
                      "alpha": self.point_alpha, **self.point_overrides.get(idx, {})}
            ax.scatter(pt[0], pt[1], zorder=zorder + 1, label="{} connections".format(self.label) if idx == 0 else None, **kwargs)

# ============================================================
# Plot PE
# ============================================================

def plot_PlotElements(
    title: str,
    xlabel: str,
    ylabel: str,
    elements: list[PlotElement] = [],
    axes_label: str = "_population",

    flag_save=False,
    save_project_relative_path: _T_rel_path = Path("data/plot/landmark_colorfull_stack"),
    save_package_name = logs.create_name_by_datetime(),
    save_nametag: str = "",

) -> None:
    fig, ax = plt.subplots()

    ax.set_title( title )
    ax.set_xlabel( xlabel )
    ax.set_ylabel( ylabel )
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)

    if elements:
        for zorder, element in enumerate(elements):
            element.display(ax, zorder+10)

    # ax.legend(loc='best')
    leg = ax.get_legend()
    if leg is not None and not leg.get_texts():
        leg.remove()

    # ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    if flag_save:
        logs.storage.save.plot(
            logs.storage.nametag(title, save_nametag),
            save_project_relative_path,
            plt
        )
    else:
        plt.show()

# ============================================================
# Plot basic
# ============================================================

def plot_landmarksBatch63_by_label(
    label: _T_label,
    points: _T_landmarkBatch63,
    color_map: _T_cmap_colorHex = COLOR_MAP,
    elements: list[PlotElement] = [],
    axes_label: str = "_population",

    flag_save=False,
    save_root_path: Path = Path("data/plot/landmark_colorfull_stack"),
    save_package_name: str = logs.create_name_by_datetime(),
    name_tag: str = "",
) -> None:

    save_path = PROJECT_ROOT / save_root_path / save_package_name
    save_path.mkdir(parents=True, exist_ok=True)

    points: _T_landmarkBatch21xyz = (
        _T__converter__.conv_landmarkBatch63_to_landmarkBatch21xyz(
            points
        )
    )

    fig, ax = plt.subplots()

    ax.scatter(
        points[:, :, 0],
        points[:, :, 1],
        marker=".",
        s=1,
        c=color_map[label],
        label=axes_label,
        zorder=1
    )
    ax.set_title( f"{label} (threshold={name_tag})")   #label)
    ax.set_xlabel("Width")
    ax.set_ylabel("Hight")
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)

    if elements:
        for zorder, element in enumerate(elements):
            element.display(ax, zorder+10)

    # ax.legend(loc='best')
    leg = ax.get_legend()
    if leg is not None and not leg.get_texts():
        leg.remove()


    # ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    if flag_save:
        plt.savefig(
            str(save_path / f"{name_tag}.jpg"),
            bbox_inches="tight",
            dpi=300,
        )
    else:
        plt.show()

# ============================================================
# Plot advance
# ============================================================

def landmarkBatch63_colorfull_stack(
        data: _T_landmarkBatch63,
        label: _T_labelBatch,
        flag_save = False,
        save_root_path: Path = Path("data/plot/landmark_colorfull_stack"),
        save_package_name: str = logs.create_name_by_datetime(),
        name_tag: str = "",
        flag_block_size: bool = False,
) -> None:
    flag_mean_marker: bool = True

    save_path = PROJECT_ROOT / save_root_path / (save_package_name + "_" + name_tag)

    ## utworzenie katalogu do zapisu
    if flag_save:
        save_path.mkdir(parents=True, exist_ok=True)


    ## mapowanie kolorów
    label_to_color = {
        lb: plt.cm.tab10(i)
        for i, lb in enumerate(set(label))
    }

    ## rzutowanie wymiararu do x, y
    data_xy = data.reshape(-1, 21, 3)[:, :, :2]

    def _saving_logic(last_lb):
        ax = plt.gca()

        if flag_block_size:
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)

            ax.set_xticks([i / 10 for i in range(11)])
            ax.set_yticks([i / 10 for i in range(11)])

            # kwadratowy wykres
            ax.set_aspect("equal", adjustable="box")

        ax.grid()

        if flag_save:
            plt.savefig(
                str(save_path / f"{last_lb}.jpg"),
                bbox_inches="tight",
                dpi = 300,
            )
        else:
            plt.show()

        plt.close()

    last_lb = None
    for sample, lb in zip(data_xy, label):
        # logika na zapis wewnatrz petli
        if lb != last_lb:
            if last_lb is not None:
                _saving_logic(last_lb )
            last_lb = lb
            plt.figure()

        plt.scatter(
            sample[:, 0],
            sample[:, 1],
            marker="o",
            s=3,
            c=[label_to_color[lb]]
        )

        # plt.text(sample[0,0], sample[0,1], "Wrist")
        # plt.text(sample[4, 0], sample[4, 1], "Thumb")
        # plt.text(sample[8, 0], sample[8, 1], "Index")
        # plt.text(sample[12, 0], sample[12, 1], "Middle")
        # plt.text(sample[16, 0], sample[16, 1], "Ring")
        # plt.text(sample[20, 0], sample[20, 1], "Pinky")

        if flag_mean_marker:
            x = np.mean(sample[:, 0])
            y = np.mean(sample[:, 1])

            # biała obwódka
            plt.scatter(
                x, y,
                marker="x",
                s=60,
                c="white",
                linewidths=2,
            )

            # właściwy marker
            plt.scatter(
                x, y,
                marker="x",
                s=40,
                c=[label_to_color[lb]],
                linewidths=1,
            )

    else:
        ## ostatni wykres
        _saving_logic(last_lb)


    ## === all ===
    plt.figure()
    for sample, lb in zip(data_xy, label):
        plt.scatter(
            sample[:, 0],
            sample[:, 1],
            marker="o",
            s=3,
            c=[label_to_color[lb]]
        )
        if flag_mean_marker:
            x = np.mean(sample[:, 0])
            y = np.mean(sample[:, 1])

            # biała obwódka
            plt.scatter(
                x, y,
                marker="x",
                s=60,
                c="white",
                linewidths=2,
            )

            # właściwy marker
            plt.scatter(
                x, y,
                marker="x",
                s=40,
                c=[label_to_color[lb]],
                linewidths=1,
            )
    else:
        _saving_logic("all")

def landmark_colorfull_stack_3d(
    data: _T_landmarkBatch63,
    label: _T_labelBatch,
    flag_save=False,
    save_root_path: Path = Path("data/plot/landmark_colorfull_stack_3d"),
    save_package_name: str = logs.create_name_by_datetime(),
    name_tag: str = "",
) -> None:

    save_path = PROJECT_ROOT / save_root_path / (save_package_name + "_" + name_tag)

    ## utworzenie katalogu do zapisu
    if flag_save:
        save_path.mkdir(parents=True, exist_ok=True)

    ## mapowanie kolorów
    label_to_color = {
        lb: plt.cm.tab10(i)
        for i, lb in enumerate(set(label))
    }

    ## rzutowanie wymiaru do x, y, z
    data_xyz = data.reshape(-1, 21, 3)

    def _saving_logic(last_lb):
        ax = plt.gca()

        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_zlim(-0.5, 0.5)

        ax.set_xticks([i / 5 for i in range(6)])
        ax.set_yticks([i / 5 for i in range(6)])
        ax.set_zticks([i / 5 for i in range(-2, 3)])

        ax.grid()

        if flag_save:
            plt.savefig(str(save_path / f"{last_lb}.jpg"))
        else:
            plt.show()

        plt.close()

    last_lb = None

    for sample, lb in zip(data_xyz, label):

        # logika na zapis wewnątrz pętli
        if lb != last_lb:

            if last_lb is not None:
                _saving_logic(last_lb)

            last_lb = lb

            plt.figure()
            ax = plt.axes(projection="3d")

        ax.scatter(
            sample[:, 0],
            sample[:, 1],
            sample[:, 2],
            marker="o",
            s=3,
            c=[label_to_color[lb]]
        )

    else:
        ## ostatni wykres
        _saving_logic(last_lb)

    ## wykres ALL
    fig = go.Figure()

    for sample, lb in zip(data_xyz, label):
        fig.add_trace(
            go.Scatter3d(
                x=sample[:, 0],
                y=sample[:, 1],
                z=sample[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=mcolors.to_hex(label_to_color[lb])
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(
                range=[0, 1],
                dtick=0.2
            ),
            yaxis=dict(
                range=[1, 0],
                dtick=0.2
            ),
            zaxis=dict(
                range=[-0.5, 0.5],
                dtick=0.2
            ),
        )
    )

    if flag_save:
        fig.write_html(save_path / "all_interactive.html")

    fig.show()


# if __name__ == '__main__':
#     label: list[_T_label_name] = list(logs.load.numpy(filename="CLASSES_TEST.npy", project_path=Path("MLP/data/array")))
#     data = logs.load.numpy(filename="POINTS_TEST.npy", project_path=Path("MLP/data/array"))
#     landmarkBatch63_colorfull_stack(data, label, flag_save=True, name_tag="TEST")
#     # landmark_colorfull_stack_3d(data, label, flag_save=True, name_tag="TEST")