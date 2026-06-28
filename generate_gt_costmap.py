"""
Stage 3 – Ground-Truth Costmap Generation
==========================================
For every saved frame, transforms per-pixel 3D points (camera frame) into
world coordinates, computes a scalar cost to the final goal, and saves both
a coloured PNG visualization and a raw float32 .npy cost map.

Three cost types are available:

  1) euclidean3d   – full 3D Euclidean distance  sqrt(dx²+dy²+dz²)
  2) groundplane   – 2D ground-plane distance     sqrt(dx²+dy²)   (recommended for driving)
  3) geodesic      – road-following distance via CARLA's GlobalRoutePlanner

Usage:
    # Defaults to euclidean3d
    python generate_costmaps.py --scene my_scene

    # Choose cost type
    python generate_costmaps.py --scene my_scene --cost groundplane
    python generate_costmaps.py --scene my_scene --cost geodesic --host 127.0.0.1 --port 2000

Note: geodesic requires a live CARLA connection to query the route planner.
      euclidean3d and groundplane are fully offline.

Output:
    out/<scene_name>/
        costmaps/          000000.png …   RGB visualisation
        costmaps_raw/      000000.npy …   (H, W) float32, actual metres
"""

import argparse
import os
import glob

import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Color mapping  black → green → red
# ──────────────────────────────────────────────────────────────────────────────

def cost_to_bgr(t: np.ndarray) -> np.ndarray:
    """
    Map normalised cost t ∈ [0, 1] to BGR colour array.

    t = 0.0  →  black   (0,   0,   0)
    t = 0.5  →  green   (0,   255, 0)
    t = 1.0  →  red     (0,   0,   255)

    Parameters
    ----------
    t : (H, W) float32, values in [0, 1]

    Returns
    -------
    bgr : (H, W, 3) uint8
    """
    H, W = t.shape
    bgr  = np.zeros((H, W, 3), dtype=np.uint8)

    lo = t < 0.5                         # black → green
    hi = ~lo                             # green → red

    alpha_lo = t[lo] / 0.5              # 0 → 1
    alpha_hi = (t[hi] - 0.5) / 0.5     # 0 → 1

    # black → green: G ramps up, R and B stay 0
    bgr[lo, 1] = (255 * alpha_lo).astype(np.uint8)

    # green → red: R ramps up, G ramps down, B stays 0
    bgr[hi, 2] = (255 * alpha_hi).astype(np.uint8)   # R  (OpenCV is BGR)
    bgr[hi, 1] = (255 * (1 - alpha_hi)).astype(np.uint8)  # G

    return bgr


# ──────────────────────────────────────────────────────────────────────────────
# Cost functions
# ──────────────────────────────────────────────────────────────────────────────

def cost_euclidean3d(pts_world: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """
    Full 3D Euclidean distance from every world point to the goal.

    Parameters
    ----------
    pts_world : (H, W, 3)  XYZ in world frame
    goal      : (3,)       goal position in world frame

    Returns
    -------
    cost : (H, W) float32
    """
    diff = pts_world - goal[np.newaxis, np.newaxis, :]   # (H, W, 3)
    return np.linalg.norm(diff, axis=-1).astype(np.float32)


def cost_groundplane(pts_world: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """
    2D ground-plane distance (XY only). Ignores elevation.

    Parameters
    ----------
    pts_world : (H, W, 3)
    goal      : (3,)

    Returns
    -------
    cost : (H, W) float32
    """
    diff_xy = pts_world[:, :, :2] - goal[np.newaxis, np.newaxis, :2]
    return np.linalg.norm(diff_xy, axis=-1).astype(np.float32)


def cost_geodesic(pts_world: np.ndarray,
                  goal: np.ndarray,
                  grp,                    # GlobalRoutePlanner instance
                  carla_module) -> np.ndarray:
    """
    Approximate geodesic (road-following) distance from every pixel to the goal.

    The GlobalRoutePlanner returns a list of waypoints along the road network.
    We measure the total arc length of that route as the cost.

    Because querying the planner for every pixel of a 320×240 image (~76 800
    times) would be extremely slow, we:

      1. Downsample the pixel grid to a coarser resolution (GEODESIC_STRIDE).
      2. Query the planner for each downsampled point.
      3. Bicubically upsample the cost map back to full resolution.

    Parameters
    ----------
    pts_world    : (H, W, 3)
    goal         : (3,)
    grp          : carla.GlobalRoutePlanner
    carla_module : the carla Python module

    Returns
    -------
    cost : (H, W) float32
    """
    H, W = pts_world.shape[:2]
    STRIDE = 8     # query 1 in every 8×8 pixels → 40×30 queries per frame

    rows = np.arange(0, H, STRIDE)
    cols = np.arange(0, W, STRIDE)

    goal_loc = carla_module.Location(x=float(goal[0]),
                                     y=float(goal[1]),
                                     z=float(goal[2]))

    sparse_H = len(rows)
    sparse_W = len(cols)
    cost_sparse = np.zeros((sparse_H, sparse_W), dtype=np.float32)

    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            px = pts_world[r, c]
            src_loc = carla_module.Location(x=float(px[0]),
                                            y=float(px[1]),
                                            z=float(px[2]))
            try:
                route = grp.trace_route(src_loc, goal_loc)
                if len(route) < 2:
                    # Same waypoint as goal – distance 0
                    cost_sparse[ri, ci] = 0.0
                else:
                    # Arc length of the route
                    total = 0.0
                    for k in range(1, len(route)):
                        a = route[k-1][0].transform.location
                        b = route[k  ][0].transform.location
                        total += np.sqrt((a.x-b.x)**2 +
                                         (a.y-b.y)**2 +
                                         (a.z-b.z)**2)
                    cost_sparse[ri, ci] = total
            except Exception:
                # Off-road point – fall back to Euclidean
                cost_sparse[ri, ci] = float(np.linalg.norm(px - goal))

    # Upsample back to (H, W)
    cost_full = cv2.resize(cost_sparse, (W, H),
                           interpolation=cv2.INTER_LINEAR)
    return cost_full.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# World-frame transform
# ──────────────────────────────────────────────────────────────────────────────

def cam_to_world(pts_cam: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform (H, W, 3) camera-frame points to world frame using a (4,4) matrix.

    Parameters
    ----------
    pts_cam : (H, W, 3) float32  points in camera frame
    T       : (4, 4)   float64  camera-to-world homogeneous transform

    Returns
    -------
    pts_world : (H, W, 3) float32
    """
    H, W, _ = pts_cam.shape
    pts_flat = pts_cam.reshape(-1, 3)                     # (H*W, 3)
    ones     = np.ones((pts_flat.shape[0], 1), dtype=np.float64)
    pts_h    = np.hstack([pts_flat.astype(np.float64), ones])  # (H*W, 4)
    pts_w    = (T @ pts_h.T).T                            # (H*W, 4)
    return pts_w[:, :3].reshape(H, W, 3).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    root           = os.path.join(args.out, args.scene)
    points_dir     = os.path.join(root, "3d_points")
    extrinsics_path = os.path.join(root, "camera_extrinsics.npy")
    trajectory_path = os.path.join(root, "trajectory.npy")
    costmap_dir    = os.path.join(root, "costmaps")
    costraw_dir    = os.path.join(root, "costmaps_raw")

    os.makedirs(costmap_dir, exist_ok=True)
    os.makedirs(costraw_dir, exist_ok=True)

    # ── Load shared data ──────────────────────────────────────────────────────
    if not os.path.isfile(trajectory_path):
        raise FileNotFoundError(f"trajectory.npy not found at {trajectory_path}")
    if not os.path.isfile(extrinsics_path):
        raise FileNotFoundError(f"camera_extrinsics.npy not found at {extrinsics_path}")

    trajectory   = np.load(trajectory_path)    # (K, 3)
    extrinsics   = np.load(extrinsics_path)    # (N, 4, 4)

    # Final goal = last anchor waypoint from the collection script
    goal = trajectory[-1]                       # (3,)
    print(f"[INFO] Goal (world): x={goal[0]:.2f}  y={goal[1]:.2f}  z={goal[2]:.2f}")

    point_files = sorted(glob.glob(os.path.join(points_dir, "*.npy")))
    if not point_files:
        raise FileNotFoundError(f"No 3D point .npy files found in {points_dir}")
    if len(point_files) != len(extrinsics):
        raise ValueError(
            f"Mismatch: {len(point_files)} point files vs "
            f"{len(extrinsics)} extrinsic matrices."
        )

    print(f"[INFO] Cost type : {args.cost}")
    print(f"[INFO] Frames    : {len(point_files)}")

    # ── Geodesic: set up CARLA connection ─────────────────────────────────────
    grp          = None
    carla_module = None
    if args.cost == "geodesic":
        try:
            import carla
            from agents.navigation.global_route_planner import GlobalRoutePlanner
        except ImportError:
            raise ImportError(
                "geodesic cost requires the 'carla' package and CARLA agents. "
                "Make sure CARLA's PythonAPI is on your PYTHONPATH."
            )
        carla_module = carla
        client = carla.Client(args.host, args.port)
        client.set_timeout(15.0)
        world      = client.get_world()
        world_map  = world.get_map()
        grp        = GlobalRoutePlanner(world_map, sampling_resolution=2.0)
        print(f"[INFO] Connected to CARLA at {args.host}:{args.port} for geodesic routing.")

    # ── Per-frame processing ──────────────────────────────────────────────────
    all_costs = []     # collect to compute global normalisation

    # Pass 1: compute raw costs
    raw_costs_list = []
    for idx, pf in enumerate(point_files):
        pts_cam   = np.load(pf)                 # (H, W, 3)
        T         = extrinsics[idx]             # (4, 4)
        pts_world = cam_to_world(pts_cam, T)    # (H, W, 3)

        if args.cost == "euclidean3d":
            cost = cost_euclidean3d(pts_world, goal)
        elif args.cost == "groundplane":
            cost = cost_groundplane(pts_world, goal)
        elif args.cost == "geodesic":
            cost = cost_geodesic(pts_world, goal, grp, carla_module)
        else:
            raise ValueError(f"Unknown cost type: {args.cost}")

        raw_costs_list.append(cost)
        all_costs.append(cost)

    # ── Global normalisation ──────────────────────────────────────────────────
    # Use the max distance observed across the entire trajectory so that
    # the colour scale is consistent across all frames.
    global_max = float(np.max(np.concatenate([c.ravel() for c in all_costs])))
    if global_max == 0.0:
        global_max = 1.0   # guard against degenerate case
    print(f"[INFO] Global max cost : {global_max:.2f} m")

    # Pass 2: normalise, colour, save
    for idx, (pf, cost) in enumerate(zip(point_files, raw_costs_list)):
        stem = os.path.splitext(os.path.basename(pf))[0]

        # Save raw cost map (metres)
        np.save(os.path.join(costraw_dir, f"{stem}.npy"), cost)

        # Normalise to [0, 1]
        cost_norm = np.clip(cost / global_max, 0.0, 1.0).astype(np.float32)

        # Colour map
        bgr = cost_to_bgr(cost_norm)
        cv2.imwrite(os.path.join(costmap_dir, f"{stem}.png"), bgr)

    print(f"[INFO] Costmaps saved to  {costmap_dir}")
    print(f"[INFO] Raw costs saved to {costraw_dir}")


if __name__ == "__main__":
    COST_TYPES = ["euclidean3d", "groundplane", "geodesic"]

    parser = argparse.ArgumentParser(
        description="Generate GT costmaps from saved 3D point clouds."
    )
    parser.add_argument("--out",   default="out",
                        help="Root output directory (same as collect_dataset.py)")
    parser.add_argument("--scene", default="scene_00",
                        help="Scene name (subfolder under --out)")
    parser.add_argument("--cost",  default="euclidean3d",
                        choices=COST_TYPES,
                        help=(
                            "Cost type to use:\n"
                            "  euclidean3d  – full 3D straight-line distance (default)\n"
                            "  groundplane  – XY-only distance, ignores elevation\n"
                            "  geodesic     – road-following distance via CARLA route planner"
                            "                 (requires live CARLA connection)"
                        ))
    # Only needed for geodesic
    parser.add_argument("--host",  default="127.0.0.1",
                        help="CARLA server host (geodesic only)")
    parser.add_argument("--port",  default=2000, type=int,
                        help="CARLA server port (geodesic only)")

    main(parser.parse_args())