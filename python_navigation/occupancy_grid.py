"""
File: occupancy_grid.py

This module provides classes and functions for handling 2D occupancy grids,
commonly used in robotics and mapping applications.
It supports loading occupancy grids from images, querying cell occupancy,
applying dilation for safety margins, and visualizing grids.
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


@dataclass
class GridMeta:
    resolution: float
    origin: Tuple[float, float, float]


class OccupancyGrid2D:
    """
    Represents a 2D occupancy grid for navigation and path planning.
    Args:
        grid (np.ndarray): 2D numpy array representing occupancy values of the grid.
        meta (GridMeta): Metadata containing grid origin and resolution.
        occ_threshold (int, optional): Threshold above which a cell is considered occupied. Defaults to 50.
        treat_unknown_as_occupied (bool, optional): If True,
          unknown cells (value < 0) are treated as occupied. Defaults to True.
    Attributes:
        grid (np.ndarray): The occupancy grid data.
        meta (GridMeta): Metadata for the grid.
        h (int): Height of the grid (number of rows).
        w (int): Width of the grid (number of columns).
        occ_th (int): Occupancy threshold.
        treat_unknown (bool): Flag for treating unknown cells as occupied.
    Methods:
        world_to_grid(x: float, y: float) -> Tuple[int, int]:
            Converts world coordinates (x, y) to grid cell indices (gx, gy).
        in_bounds(gx: int, gy: int) -> bool:
            Checks if the grid cell indices (gx, gy) are within the grid bounds.
        is_occupied_cell(gx: int, gy: int) -> bool:
            Determines if the cell at (gx, gy) is occupied or unknown
              (based on threshold and unknown policy).
        point_is_free(x: float, y: float, clearance_cells: int = 0) -> bool:
            Checks if the world point (x, y) is free, optionally considering
              a clearance (number of cells) around the point.
    """

    def __init__(self, grid: np.ndarray, meta: GridMeta,
                 occ_threshold: int = 50, treat_unknown_as_occupied: bool = True):
        self.grid = grid
        self.meta = meta
        self.h, self.w = grid.shape
        self.occ_th = occ_threshold
        self.treat_unknown = treat_unknown_as_occupied

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Converts world coordinates (x, y) to grid coordinates (gx, gy).
        Args:
            x (float): The x-coordinate in world space.
            y (float): The y-coordinate in world space.
        Returns:
            Tuple[int, int]: The corresponding grid coordinates (gx, gy).
        Notes:
            The conversion uses the origin and resolution defined in the occupancy grid metadata.
        """
        x0, y0, _ = self.meta.origin
        gx = int(round((x - x0) / self.meta.resolution))
        gy = int(round((y - y0) / self.meta.resolution))
        return gx, gy

    def in_bounds(self, gx: int, gy: int) -> bool:
        """
        Check if the given grid coordinates (gx, gy) are within the bounds of the occupancy grid.
        Args:
            gx (int): The x-coordinate (column index) to check.
            gy (int): The y-coordinate (row index) to check.
        Returns:
            bool: True if the coordinates are within the grid bounds, False otherwise.
        """
        return 0 <= gx < self.w and 0 <= gy < self.h

    def is_occupied_cell(self, gx: int, gy: int) -> bool:
        """
        Determines whether the cell at grid coordinates (gx, gy) is considered occupied.
        Args:
            gx (int): The x-coordinate of the cell in the grid.
            gy (int): The y-coordinate of the cell in the grid.
        Returns:
            bool: True if the cell is occupied or out of bounds, False otherwise.
        Notes:
            - If the cell is out of bounds, it is treated as occupied.
            - If the cell value is negative, the method returns the value of self.treat_unknown.
            - Otherwise, the cell is considered occupied
              if its value is greater than or equal to self.occ_th.
        """
        if not self.in_bounds(gx, gy):
            return True
        v = int(self.grid[gy, gx])
        if v < 0:
            return self.treat_unknown
        return v >= self.occ_th

    def point_is_free(self, x: float, y: float, clearance_cells: int = 0) -> bool:
        """
        Determines whether a point in the world coordinates (x, y)
        is free (not occupied) in the occupancy grid.
        Args:
            x (float): The x-coordinate in world space.
            y (float): The y-coordinate in world space.
            clearance_cells (int, optional): The number of grid cells around the point to check for clearance.
                If greater than 0, the function checks a square region of size
                  (2 * clearance_cells + 1) centered at (x, y).
                Defaults to 0 (only the cell containing the point is checked).
        Returns:
            bool: True if the point (and optionally its surrounding region)
              is free of obstacles, False otherwise.
        """
        gx, gy = self.world_to_grid(x, y)
        if clearance_cells <= 0:
            return not self.is_occupied_cell(gx, gy)
        for yy in range(gy - clearance_cells, gy + clearance_cells + 1):
            for xx in range(gx - clearance_cells, gx + clearance_cells + 1):
                if self.is_occupied_cell(xx, yy):
                    return False
        return True


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Simple binary dilation without dependencies (square neighborhood).
      radius=1 means 3x3.
    """
    if radius <= 0:
        return mask
    h, w = mask.shape
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            # シフト貼り付け
            ys = slice(max(0, -dy), h - max(0, dy))
            xs = slice(max(0, -dx), w - max(0, dx))
            yd = slice(max(0, dy), h - max(0, -dy))
            xd = slice(max(0, dx), w - max(0, -dx))
            shifted = np.zeros_like(mask)
            shifted[yd, xd] = mask[ys, xs]
            out |= shifted
    return out


def image_to_occupancy_grid(
    image_path: str,
    resolution: float,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    free_threshold: int = 205,    # 0..255：above this is free
    occ_threshold: int = 50,      # 0..255：below this is occupied
    # A<this -> unknown（only for alpha images）
    alpha_unknown_threshold: Optional[int] = 1,
    invert_intensity: bool = False,  # If True, invert "white=occupied / black=free"
    # If True, flip Y axis to make +Y upward in world coordinates
    flip_y_up: bool = True,
    dilate_radius_px: int = 0     # Dilation of occupied pixels (safety margin)
) -> OccupancyGrid2D:
    """
    Convert an image (white=free, black=occupied, gray=unknown) to an occupancy grid with values 0/100/-1,
    and return as an OccupancyGrid2D instance.
    - resolution: [m/cell]
    - origin: (x0, y0, theta) in world coordinates. If you want the image bottom-left as origin, set flip_y_up=True.
    - free_threshold and occ_threshold are intensity thresholds (L mode 0..255).
    - If alpha_unknown_threshold is None, alpha channel is ignored.
    """
    img = Image.open(image_path)
    has_alpha = "A" in img.getbands()
    gray = img.convert("L")
    I = np.asarray(gray, dtype=np.uint8)

    # If the image is white=occupied/black=free, invert it.
    if invert_intensity:
        I = 255 - I

    # If you want +Y to be upward in world coordinates, flip the image vertically.
    if flip_y_up:
        I = np.flipud(I)

    H, W = I.shape

    grid = np.full((H, W), -1, dtype=np.int16)

    if has_alpha and alpha_unknown_threshold is not None:
        A = np.asarray(img.split()[-1], dtype=np.uint8)
        if flip_y_up:
            A = np.flipud(A)
        known_mask = A >= alpha_unknown_threshold
    else:
        known_mask = np.ones_like(I, dtype=bool)

    occ_mask = (I <= occ_threshold) & known_mask
    free_mask = (I >= free_threshold) & known_mask

    if dilate_radius_px > 0:
        occ_mask = _binary_dilate(occ_mask, dilate_radius_px)

    grid[occ_mask] = 100
    grid[free_mask & ~occ_mask] = 0

    meta = GridMeta(resolution=resolution, origin=origin)
    return OccupancyGrid2D(grid, meta)


def plot_occupancy_grid(og: OccupancyGrid2D, cmap="gray"):
    """Visualize an OccupancyGrid2D using matplotlib.
    - og.grid: 2D numpy array (values: -1=unknown, 0=free, 100=occupied)
    - og.meta.resolution: float, cell size [m]
    - og.meta.origin: (x, y, yaw), origin coordinates
    """
    grid = og.grid

    img = np.ones_like(grid, dtype=float)
    img[grid == -1] = 0.5
    img[grid == 0] = 1.0
    img[grid == 100] = 0.0

    height, width = grid.shape
    extent = [
        og.meta.origin[0], og.meta.origin[0] + width * og.meta.resolution,
        og.meta.origin[1], og.meta.origin[1] + height * og.meta.resolution,
    ]

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap, origin="lower", extent=extent)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Occupancy Grid Map")
    plt.colorbar(label="Occupancy (0=occupied, 1=free, 0.5=unknown)")
    plt.grid(False)
    plt.show()
