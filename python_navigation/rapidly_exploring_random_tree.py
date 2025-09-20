"""
File: rapidly_exploring_random_tree.py

This module implements a simple Rapidly-exploring Random Tree Star (RRT*) planner
for non-holonomic vehicles using a greedy local planner inspired by Reeds-Shepp curves.

Usage:
------
- Instantiate OccupancyGrid2D and SimpleRSLocalPlanner.
- Create RRTStar_SimpleRS with appropriate parameters.
- Call plan(start, goal) to compute a collision-free path.

Note:
-----
- The local planner uses a greedy feedback approach and does not compute exact Reeds-Shepp paths.
- The planner supports both forward and backward movement and rewiring for optimality.
"""
import math
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

from python_navigation.occupancy_grid import OccupancyGrid2D


@dataclass
class Node:
    x: float
    y: float
    yaw: float
    parent: Optional[int]
    cost: float


class SimpleRSLocalPlanner:
    """
    Instead of exact Reeds-Shepp (RS) analytical solutions,
    this local planner generates a short connecting trajectory from q0 to q1 by numerical integration,
    using the following simple method:
    - Curvature limit: |kappa| <= 1/rho
    - Forward/backward movement allowed (assume v = ±1 and integrate along arc length)
    - "Goal-directed" greedy feedback to determine steering angle (= curvature)
    """

    def __init__(self, rho: float = 0.4, validation_ds: float = 0.05, max_conn_len: float = 2.0):
        self.rho = float(rho)
        self.validation_ds = float(validation_ds)
        self.max_conn_len = float(max_conn_len)
        self.kappa_max = 1.0 / max(self.rho, 1e-6)

        self.k_heading = 2.0     # Heading weight
        self.k_yaw_goal = 0.5    # Yaw error weight
        # If the angle error is larger than this, allow reversing once.
        self.switch_back_ang = 100.0 * math.pi / 180.0

    @staticmethod
    def _wrap(a: float) -> float:
        """
        Wraps the input angle to the range [-pi, pi].
        Args:
            a (float): The angle in radians to be wrapped.
        Returns:
            float: The wrapped angle in radians within [-pi, pi].
        """
        return math.atan2(math.sin(a), math.cos(a))

    def _greedy_control(self, q: Tuple[float, float, float], q_goal: Tuple[float, float, float]) -> Tuple[int, float]:
        """
        q=(x,y,yaw), q_goal=(xg,yg,yawg)
        Returns the direction of movement v_sign ∈ {+1,-1} and curvature kappa.
        """
        x, y, yaw = q
        xg, yg, yawg = q_goal
        dx, dy = xg - x, yg - y
        dist = math.hypot(dx, dy)
        dir_to_goal = math.atan2(dy, dx)

        ang_err_forward = abs(self._wrap(dir_to_goal - yaw))
        ang_err_backward = abs(self._wrap(
            dir_to_goal - self._wrap(yaw + math.pi)))
        v_sign = +1 if ang_err_forward <= ang_err_backward else -1

        if v_sign > 0:
            heading_err = self._wrap(dir_to_goal - yaw)
        else:
            heading_err = self._wrap(dir_to_goal - self._wrap(yaw + math.pi))

        yaw_goal_err = self._wrap(yawg - yaw)

        w1 = 1.0 if dist > 1.0 else max(0.2, dist / 1.0)
        w2 = 1.0 - w1
        target_turn = self.k_heading * w1 * heading_err + \
            self.k_yaw_goal * w2 * yaw_goal_err

        kappa = max(-self.kappa_max, min(self.kappa_max, target_turn))
        return v_sign, kappa

    def steer_and_truncate(self, q0: Tuple[float, float, float], q1: Tuple[float, float, float]):
        """
        Follow from q0 to q1 using greedy control.
        Truncate if arc length L exceeds max_conn_len.
        Returns: states [(x,y,theta)...], used_length
        """
        x, y, th = q0
        states = [(x, y, th)]
        L = 0.0
        ds = self.validation_ds
        for _ in range(int(math.ceil(self.max_conn_len / ds))):
            v_sign, kappa = self._greedy_control((x, y, th), q1)

            dth = v_sign * kappa * ds
            if abs(kappa) < 1e-6:
                x += v_sign * ds * math.cos(th)
                y += v_sign * ds * math.sin(th)
            else:
                R = 1.0 / kappa

                x += v_sign * R * (math.sin(th + dth) - math.sin(th))
                y += v_sign * R * (-math.cos(th + dth) + math.cos(th))
            th = self._wrap(th + dth)
            L += ds
            states.append((x, y, th))

            dx, dy = q1[0] - x, q1[1] - y
            dpos = math.hypot(dx, dy)
            dyaw = abs(self._wrap(q1[2] - th))
            if dpos < 0.05 and dyaw < 5.0 * math.pi / 180.0:
                break
        return states, L


class RRTStar_SimpleRS:
    """
    Rapidly-exploring Random Tree Star (RRT*) implementation with a simple Reeds-Shepp local planner for 2D navigation.
    This class plans a collision-free path from a start to a goal pose in a 2D occupancy grid, considering robot orientation.
    It uses the RRT* algorithm, which incrementally builds a tree of feasible states and rewires it to optimize path cost.
    Args:
        grid (OccupancyGrid2D): The occupancy grid map for collision checking.
        lp (SimpleRSLocalPlanner): Local planner for generating feasible paths between states.
        x_range (Tuple[float, float]): Allowed range for x-coordinate sampling.
        y_range (Tuple[float, float]): Allowed range for y-coordinate sampling.
        theta_range (Tuple[float, float], optional): Allowed range for orientation (yaw) sampling. Defaults to (-pi, pi).
        goal_pos_tol (float, optional): Position tolerance for considering the goal reached. Defaults to 0.3.
        goal_yaw_tol (float, optional): Orientation tolerance (radians) for goal. Defaults to 30 degrees.
        gamma_rrtstar (float, optional): RRT* radius scaling parameter. Defaults to 3.0.
        max_iter (int, optional): Maximum number of iterations for planning. Defaults to 30000.
        clearance (float, optional): Minimum clearance from obstacles (meters). Defaults to 0.0.
        rng (Optional[random.Random], optional): Random number generator instance. Defaults to None.
        verbose (bool, optional): If True, prints progress information. Defaults to False.
        progress_interval (int, optional): Iteration interval for progress reporting. Defaults to 1000.
    Methods:
        plan(start, goal):
            Plans a path from start to goal pose.
            Returns a list of (x, y, yaw) tuples representing the path, or None if no path is found.
    Internal Methods:
        _angdiff(a, b): Computes the shortest angular difference between a and b.
        _sample(goal, goal_bias): Samples a random state or the goal with given bias.
        _nearest(q): Finds the index of the nearest node to state q.
        _near_radius(n): Computes the radius for neighbor search based on node count.
        _near_indices(q, radius): Returns indices of nodes within a given radius of q.
        _collision_free_states(states): Checks if all states are within bounds and collision-free.
        _connect(i_from, q_to): Uses the local planner to connect two states.
        _goal_reached(q, goal): Checks if state q is within goal tolerance.
    Attributes:
        nodes (List[Node]): List of tree nodes, each with position, orientation, parent, and cost.
    """

    def __init__(self, grid: OccupancyGrid2D, lp: SimpleRSLocalPlanner,
                 x_range: Tuple[float, float], y_range: Tuple[float, float],
                 theta_range: Tuple[float, float] = (-math.pi, math.pi),
                 goal_pos_tol: float = 0.3, goal_yaw_tol: float = 30 * math.pi / 180.0,
                 gamma_rrtstar: float = 3.0, max_iter: int = 30000,
                 clearance: float = 0.0, rng: Optional[random.Random] = None,
                 verbose: bool = False, progress_interval: int = 1000):
        self.grid = grid
        self.lp = lp
        self.xr, self.yr, self.tr = x_range, y_range, theta_range
        self.gpos_tol, self.gyaw_tol = goal_pos_tol, goal_yaw_tol
        self.gamma = gamma_rrtstar
        self.max_iter = max_iter
        self.clearance_cells = int(round(clearance / grid.meta.resolution))
        self.rng = rng or random.Random()
        self.nodes: List[Node] = []

        # progress/logging
        self.verbose = bool(verbose)
        self.progress_interval = int(progress_interval)

    @staticmethod
    def _angdiff(a, b):  # a-b
        """
        Compute the minimal angular difference between two angles.
        Parameters
        ----------
        a : float, array-like
            First angle(s) in radians.
        b : float, array-like
            Second angle(s) in radians.
        Returns
        -------
        float or ndarray
            The minimal difference(s) between angles `a` and `b`, in radians,
            wrapped to the range [-pi, pi]. Returns a float if inputs are scalars,
            otherwise returns a NumPy array.
        Notes
        -----
        This function supports both scalar and array inputs using NumPy.
        """
        # Supports NumPy: a,b can be either scalars or arrays.
        da = np.asarray(a) - np.asarray(b)
        res = np.arctan2(np.sin(da), np.cos(da))
        # If the result is a scalar, return a Python float
        if np.ndim(res) == 0:
            return float(res)
        return res

    def _sample(self, goal, goal_bias=0.2):
        """
        Samples a random point in the search space with a specified probability of returning the goal.
        Args:
            goal (tuple): The goal point to sample towards, typically a tuple of coordinates.
            goal_bias (float, optional): Probability of sampling
              the goal point instead of a random point. Default is 0.2.
        Returns:
            tuple: A sampled point in the search space. With probability `goal_bias`,
            returns the goal point; otherwise, returns a randomly sampled point
            within the defined ranges for x, y, and theta.
        """

        if self.rng.random() < goal_bias:
            return goal
        return (self.rng.uniform(*self.xr), self.rng.uniform(*self.yr), self.rng.uniform(*self.tr))

    def _nearest(self, q):
        """
        Finds the index of the node in self.nodes that is nearest to the given query point q.
        The distance metric combines Euclidean distance in (x, y) and
        a weighted angular difference in yaw.
        Args:
            q (array-like): A query point represented as [x, y, yaw].
        Returns:
            int: The index of the nearest node in self.nodes.
        """
        pts = np.array([[n.x, n.y, n.yaw] for n in self.nodes])
        d2 = (pts[:, 0] - q[0])**2 + (pts[:, 1] - q[1])**2 + \
            (0.2 * self._angdiff(pts[:, 2], q[2]))**2
        return int(np.argmin(d2))

    def _near_radius(self, n: int) -> float:
        """
        Computes the radius for the 'near' neighborhood in the RRT* algorithm.
        The radius is determined based on the number of nodes `n` in the tree,
        following the formula: gamma * sqrt(log(n + 1) / (n + 1)), where `gamma`
        is a scaling parameter. For n <= 1, returns `gamma` directly.
        Args:
            n (int): The current number of nodes in the tree.
        Returns:
            float: The computed near radius for connecting nodes.
        """
        if n <= 1:
            return self.gamma
        return self.gamma * math.sqrt(math.log(n + 1) / (n + 1))

    def _near_indices(self, q, radius) -> List[int]:
        """
        Finds the indices of nodes within a specified radius of a given query point.
        Args:
            q (Tuple[float, float]): The query point as a tuple (x, y).
            radius (float): The radius within which to search for nearby nodes.
        Returns:
            List[int]: A list of indices of nodes whose Euclidean distance
            from the query point is less than or equal to the specified radius.
        """

        pts = np.array([[n.x, n.y] for n in self.nodes])
        d2 = (pts[:, 0] - q[0])**2 + (pts[:, 1] - q[1])**2
        return list(np.where(d2 <= radius * radius)[0])

    def _collision_free_states(self, states) -> bool:
        """
        Checks whether all given states are collision-free within the defined boundaries and grid.
        Args:
            states (Iterable[Tuple[float, float, float]]):
                An iterable of states, where each state is a tuple (x, y, _th) representing
                the position (x, y) and orientation (_th).
        Returns:
            bool: True if all states are within the allowed boundaries and do not collide
            with obstacles in the grid (considering clearance), False otherwise.
        """
        for (x, y, _th) in states:
            if not (self.xr[0] <= x <= self.xr[1] and self.yr[0] <= y <= self.yr[1]):
                return False
            if not self.grid.point_is_free(x, y, self.clearance_cells):
                return False
        return True

    def _connect(self, i_from: int, q_to):
        """
        Attempts to connect a node at index `i_from` to a target configuration
        `q_to` using the local planner.
        Parameters:
            i_from (int): Index of the starting node in the tree.
            q_to: Target configuration to connect to, typically a tuple (x, y, yaw).
        Returns:
            Result of the local planner's steer_and_truncate method,
            representing the path or connection between `q_from` and `q_to`.
        """
        q_from = (self.nodes[i_from].x,
                  self.nodes[i_from].y, self.nodes[i_from].yaw)
        return self.lp.steer_and_truncate(q_from, q_to)

    def _goal_reached(self, q, goal) -> bool:
        """
        Determines whether the given configuration `q` has reached the goal configuration `goal`.
        The function checks if both the positional distance and the yaw (orientation) difference
        between `q` and `goal` are within specified tolerances.
        Args:
            q (tuple): The current configuration as a tuple (x, y, yaw).
            goal (tuple): The goal configuration as a tuple (x, y, yaw).
        Returns:
            bool: True if the position and yaw are within their respective tolerances,
              False otherwise.
        """
        dpos = math.hypot(q[0] - goal[0], q[1] - goal[1])
        dyaw = abs(self._angdiff(q[2], goal[2]))
        return (dpos <= self.gpos_tol) and (dyaw <= self.gyaw_tol)

    def plan(self, start, goal):
        """
        Plans a path from the start to the goal using the
        Rapidly-exploring Random Tree (RRT*) algorithm.
        Args:
            start (tuple): The starting state as a tuple (x, y, yaw).
            goal (tuple): The goal state as a tuple (x, y, yaw).
        Returns:
            list or None: A list of states representing the path from start to goal,
            where each state is a tuple (x, y, yaw).
            Returns None if no valid path is found.
        """

        if not self.grid.point_is_free(start[0], start[1]):
            return None
        if not self.grid.point_is_free(goal[0], goal[1]):
            return None
        self.nodes = [Node(*start, parent=None, cost=0.0)]
        best_goal_idx, best_cost = None, float("inf")

        for it in range(self.max_iter):
            if self.verbose and (it % max(1, self.progress_interval) == 0):
                best_str = f"{best_cost:.3f}" if best_cost < float(
                    "inf") else "inf"
                print(
                    f"[RRT*] iter {it}/{self.max_iter}  nodes {len(self.nodes)}  best_cost {best_str}", flush=True)

            qs = self._sample(goal)
            i_near = self._nearest(qs)
            s1, L1 = self._connect(i_near, qs)
            if L1 <= 1e-9 or not self._collision_free_states(s1):
                continue
            q_new = s1[-1]

            r = self._near_radius(len(self.nodes))
            near_ids = self._near_indices(q_new, r)
            if i_near not in near_ids:
                near_ids.append(i_near)

            best_parent, best_new_cost = i_near, self.nodes[i_near].cost + L1
            for j in near_ids:
                s2, L2 = self._connect(j, q_new)
                if L2 <= 1e-9 or not self._collision_free_states(s2):
                    continue
                c2 = self.nodes[j].cost + L2
                if c2 < best_new_cost:
                    best_parent, best_new_cost = j, c2

            new_idx = len(self.nodes)
            self.nodes.append(
                Node(*q_new, parent=best_parent, cost=best_new_cost))

            # Rewire
            for j in near_ids:
                if j == best_parent or j == new_idx:
                    continue
                s3, L3 = self._connect(
                    new_idx, (self.nodes[j].x, self.nodes[j].y, self.nodes[j].yaw))
                if L3 <= 1e-9 or not self._collision_free_states(s3):
                    continue
                new_cost = self.nodes[new_idx].cost + L3
                if new_cost + 1e-9 < self.nodes[j].cost:
                    self.nodes[j].parent = new_idx
                    self.nodes[j].cost = new_cost

            # Goal check
            if self._goal_reached(q_new, goal):
                s_goal, Lg = self._connect(new_idx, goal)
                if Lg > 1e-9 and self._collision_free_states(s_goal):
                    tot = self.nodes[new_idx].cost + Lg
                    if tot < best_cost:
                        best_cost, best_goal_idx = tot, new_idx

        if best_goal_idx is None:
            return None
        path = []
        idx = best_goal_idx
        while idx is not None:
            n = self.nodes[idx]
            path.append((n.x, n.y, n.yaw))
            idx = n.parent
        path.reverse()
        if not self._goal_reached(path[-1], goal):
            path.append(goal)
        return path
