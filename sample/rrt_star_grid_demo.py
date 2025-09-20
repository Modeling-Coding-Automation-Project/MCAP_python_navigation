"""
File: rrt_star_grid_demo.py

This script demonstrates the use of RRT* (Rapidly-exploring Random Tree Star)
with a simplified Reeds-Shepp-like steering local planner
for path planning on a 2D occupancy grid.
"""
import os
import sys
sys.path.append(os.getcwd())

import random
import math
import numpy as np
import matplotlib.pyplot as plt

from python_navigation.occupancy_grid import OccupancyGrid2D
from python_navigation.occupancy_grid import GridMeta
from python_navigation.rapidly_exploring_random_tree import SimpleRSLocalPlanner
from python_navigation.rapidly_exploring_random_tree import RRTStar_SimpleRS


def _demo_map():
    res = 0.05
    W, H = 200, 150
    g = np.zeros((H, W), dtype=np.int16)
    g[20:120, 40:60] = 100
    g[60:140, 100:120] = 100
    g[0:80, 140:160] = 100
    meta = GridMeta(resolution=res, origin=(0, 0, 0))
    og = OccupancyGrid2D(g, meta)
    start = (0.5, 0.5, 0.0)
    goal = (9.5, 7.0, -1.2)
    return og, start, goal


def main():
    og, start, goal = _demo_map()
    # Use occupancy grid origin so world coordinates reflect the provided origin
    xr = (og.meta.origin[0], og.meta.origin[0] + og.w * og.meta.resolution)
    yr = (og.meta.origin[1], og.meta.origin[1] + og.h * og.meta.resolution)

    lp = SimpleRSLocalPlanner(rho=0.4, validation_ds=0.1, max_conn_len=0.2)

    planner = RRTStar_SimpleRS(
        grid=og, lp=lp, x_range=xr, y_range=yr,
        goal_pos_tol=1.0, goal_yaw_tol=45 * math.pi / 180.0,
        gamma_rrtstar=8.0, max_iter=7000, clearance=0.10, rng=random.Random(0),
        verbose=True, progress_interval=1000
    )

    path = planner.plan(start, goal)
    print("found:", path is not None)

    # visualize
    fig, ax = plt.subplots()
    occ = (og.grid >= 100)
    ax.imshow(occ, cmap="gray_r", origin="lower",
              extent=[xr[0], xr[1], yr[0], yr[1]], interpolation="nearest")
    for i, n in enumerate(planner.nodes):
        if n.parent is not None:
            p = planner.nodes[n.parent]
            ax.plot([p.x, n.x], [p.y, n.y], linewidth=0.3)
    if path:
        ax.plot([p[0] for p in path], [p[1] for p in path], linewidth=2)
        for (x, y, yaw) in path[::max(1, len(path) // 30)]:
            ax.arrow(x, y, 0.3 * math.cos(yaw), 0.3 * math.sin(yaw),
                     head_width=0.1, length_includes_head=True)
    ax.plot(start[0], start[1], "o")
    ax.plot(goal[0], goal[1], "x")
    ax.set_aspect("equal")
    ax.set_title("RRT* with simplified RS-like steering")
    plt.show()


if __name__ == "__main__":
    main()
