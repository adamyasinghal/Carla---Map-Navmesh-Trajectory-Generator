"""
Stage 2 – Offline 3D Point Cloud Generation
=============================================
Loads the metric depth and camera intrinsics saved by collect_dataset.py
and back-projects every pixel to a 3D point in the camera frame.

Run after Stage 1:
    python compute_3d_points.py --scene my_scene

Output:
    out/<scene_name>/3d_points/000000.npy  …   shape (H, W, 3), float32
    X = right, Y = down, Z = forward  (standard pinhole / camera frame)
"""

import argparse
import os
import glob

import numpy as np


def backproject(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Back-project a metric depth map to 3D points in the camera frame.

    Parameters
    ----------
    depth_m : (H, W) float32   metric depth in metres
    K       : (3, 3) float64   camera intrinsic matrix

    Returns
    -------
    points  : (H, W, 3) float32
              X = right, Y = down, Z = forward (camera frame)
    """
    H, W = depth_m.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel coordinate grids
    u = np.arange(W, dtype=np.float32)   # (W,)
    v = np.arange(H, dtype=np.float32)   # (H,)
    uu, vv = np.meshgrid(u, v)            # (H, W)

    X = (uu - cx) / fx * depth_m
    Y = (vv - cy) / fy * depth_m
    Z = depth_m

    return np.stack([X, Y, Z], axis=-1)  # (H, W, 3)


def main(args):
    root        = os.path.join(args.out, args.scene)
    depth_dir   = os.path.join(root, "images_depth")
    points_dir  = os.path.join(root, "3d_points")
    intrinsics_path = os.path.join(root, "camera_intrinsics.npy")

    os.makedirs(root, exist_ok=True)
    if os.path.exists(points_dir):
        import shutil
        shutil.rmtree(points_dir)
    os.makedirs(points_dir)

    if not os.path.isfile(intrinsics_path):
        raise FileNotFoundError(f"Intrinsics not found: {intrinsics_path}")
    K = np.load(intrinsics_path)

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
    if not depth_files:
        raise FileNotFoundError(f"No depth .npy files found in {depth_dir}")

    print(f"[INFO] Processing {len(depth_files)} depth maps …")

    for depth_path in depth_files:
        stem = os.path.splitext(os.path.basename(depth_path))[0]   # e.g. "000042"
        depth_m = np.load(depth_path)                               # (H, W)

        points  = backproject(depth_m, K)                           # (H, W, 3)

        out_path = os.path.join(points_dir, f"{stem}.npy")
        np.save(out_path, points)

    print(f"[INFO] Done. 3D point clouds saved to {points_dir}")
    print(f"       Shape per file: {points.shape}  dtype: {points.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",   default="out")
    parser.add_argument("--scene", default="scene_00")
    main(parser.parse_args())