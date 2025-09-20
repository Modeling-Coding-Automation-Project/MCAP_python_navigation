"""
File: office_area_path_plan_demo.py

This script demonstrates path planning in an office area
using an occupancy grid and RRT* (Rapidly-exploring Random Tree Star)
with a simplified RS-like (Reeds-Shepp-like) steering local planner.
"""
import os
import sys
sys.path.append(os.getcwd())

import math
import random
import matplotlib.pyplot as plt
import csv
from datetime import datetime

from python_navigation.occupancy_grid import image_to_occupancy_grid
from python_navigation.occupancy_grid import plot_occupancy_grid

from python_navigation.rapidly_exploring_random_tree import SimpleRSLocalPlanner
from python_navigation.rapidly_exploring_random_tree import RRTStar_SimpleRS


def _demo_map():
    og = image_to_occupancy_grid(
        image_path="./sample/office_area.png",
        resolution=0.05,
        origin=(-10.0, -15.0, 0.0),
        free_threshold=205,
        occ_threshold=50,
        # If the map is white=free, black=occupied, set to False.
        invert_intensity=False,
        flip_y_up=True,          # If True, flip the Y-axis to have +Y up
        dilate_radius_px=1       # 1 pixel safety margin (optional)
    )

    # plot_occupancy_grid(og)

    start = (0.0, 0.0, 0.0)
    goal = (23.0, -3.0, 0.0)

    return og, start, goal


def main():
    og, start, goal = _demo_map()

    # Use occupancy grid origin so world coordinates reflect the provided origin
    xr = (og.meta.origin[0], og.meta.origin[0] + og.w * og.meta.resolution)
    yr = (og.meta.origin[1], og.meta.origin[1] + og.h * og.meta.resolution)

    lp = SimpleRSLocalPlanner(rho=0.4, validation_ds=0.1, max_conn_len=0.8)

    planner = RRTStar_SimpleRS(
        grid=og, lp=lp, x_range=xr, y_range=yr,
        goal_pos_tol=1.0, goal_yaw_tol=45 * math.pi / 180.0,
        gamma_rrtstar=8.0, max_iter=7000, clearance=0.10, rng=random.Random(0),
        verbose=True, progress_interval=1000
    )

    path = planner.plan(start, goal)
    print("found:", path is not None)

    # Save found path to CSV (index,x,y,yaw) in the sample/ directory with timestamp
    if path:
        try:
            out_dir = os.path.dirname(__file__)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(out_dir, f"path_{timestamp}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "x", "y", "yaw"])
                for i, (x, y, yaw) in enumerate(path):
                    writer.writerow([i, f"{x:.6f}", f"{y:.6f}", f"{yaw:.6f}"])
            print(f"Saved path CSV to: {csv_path}")
        except Exception as e:
            print("Failed to save path CSV:", e)

    # visualize
    fig, ax = plt.subplots()
    occ = (og.grid >= 100)
    # extent uses (xmin, xmax, ymin, ymax) in world coordinates
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
