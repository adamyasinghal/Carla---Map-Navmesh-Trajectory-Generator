"""
Mass Dataset Generation Orchestrator
=======================================
Generates a full multi-world, multi-trajectory CARLA dataset by calling the
pipeline scripts in order, for every (world, trajectory) combination:

    1. generate_world.py      — spawns static traffic + pedestrians
    2. generate_dataset.py    — OR generate_dataset_grp.py (toggle)
    3. generate_3d_points.py  — offline 3D point back-projection
    4. generate_gt_costmap.py — ground-truth costmap rendering

Since every pipeline script is deterministic given a seed, "different worlds"
and "different trajectories within a world" are both produced purely by
varying the --seed argument passed to each stage.

Seeding scheme
--------------
For world index w (0-indexed) and trajectory index t (0-indexed):

    world_seed       = world_seed_base + w
    trajectory_seed   = traj_seed_base  + w * trajectories_per_world + t

world_seed controls generate_world.py (static actor placement).
trajectory_seed controls generate_dataset(.py|_grp.py) (route + spawn point).

Scene naming
------------
    out/world{w:02d}_traj{t:02d}/

Usage:
    # Defaults: 1 world, 10 trajectories per world, random-walk collection
    python generate_mass_dataset.py

    # 10 worlds, 15 trajectories each, GRP-based collection
    python generate_mass_dataset.py --num_worlds 10 --trajectories_per_world 15 --use_grp

    # Custom cost type and CARLA connection
    python generate_mass_dataset.py --cost groundplane --host 127.0.0.1 --port 2000
"""

import argparse
import subprocess
import sys
import time
import os


def run_step(cmd: list[str], step_name: str, log_prefix: str) -> bool:
    """
    Run one pipeline stage as a subprocess. Returns True on success.
    Streams the subprocess output live, prefixed for readability.
    """
    print(f"\n{'='*80}")
    print(f"[{log_prefix}] {step_name}")
    print(f"[{log_prefix}] CMD: {' '.join(cmd)}")
    print(f"{'='*80}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[{log_prefix}] ✗ {step_name} FAILED "
              f"(exit code {result.returncode})")
        return False

    print(f"[{log_prefix}] ✓ {step_name} completed.")
    return True


def main(args):
    collect_script = "generate_dataset_grp.py" if args.use_grp else "generate_dataset.py"

    print(f"[CONFIG] Worlds: {args.num_worlds}")
    print(f"[CONFIG] Trajectories per world: {args.trajectories_per_world}")
    print(f"[CONFIG] Total scenes to generate: "
          f"{args.num_worlds * args.trajectories_per_world}")
    print(f"[CONFIG] Collection script: {collect_script}")
    print(f"[CONFIG] Cost type: {args.cost}")
    print(f"[CONFIG] Output root: {args.out}")

    total_scenes   = args.num_worlds * args.trajectories_per_world
    completed      = 0
    failed_scenes  = []

    t_start = time.time()

    for w in range(args.num_worlds):
        world_seed = args.world_seed_base + w

        # ── Stage 1: generate the world (static traffic + pedestrians) ────────
        world_cmd = [
            sys.executable, "generate_world.py",
            "--host", args.host,
            "--port", str(args.port),
            "--seed", str(world_seed),
            "--walker_seed", str(world_seed),
        ]
        ok = run_step(world_cmd,
                      f"World {w+1}/{args.num_worlds} generation (seed={world_seed})",
                      log_prefix=f"world{w:02d}")
        if not ok:
            print(f"[ERROR] World {w} generation failed — skipping all "
                  f"{args.trajectories_per_world} trajectories for this world.")
            failed_scenes.extend(
                [f"world{w:02d}_traj{t:02d}"
                 for t in range(args.trajectories_per_world)]
            )
            continue

        for t in range(args.trajectories_per_world):
            traj_seed = (args.traj_seed_base
                        + w * args.trajectories_per_world + t)
            scene_name = f"world{w:02d}_traj{t:02d}"
            log_prefix = scene_name

            print(f"\n[PROGRESS] Scene {completed + 1}/{total_scenes}  "
                  f"({scene_name}, world_seed={world_seed}, traj_seed={traj_seed})")

            # ── Stage 2: collect dataset (drive + capture) ─────────────────────
            collect_cmd = [
                sys.executable, collect_script,
                "--host",  args.host,
                "--port",  str(args.port),
                "--out",   args.out,
                "--scene", scene_name,
                "--seed",  str(traj_seed),
            ]
            ok = run_step(collect_cmd,
                          f"Collect dataset ({collect_script})",
                          log_prefix)
            if not ok:
                failed_scenes.append(scene_name)
                continue

            # ── Stage 3: offline 3D point computation ──────────────────────────
            points_cmd = [
                sys.executable, "generate_3d_points.py",
                "--out",   args.out,
                "--scene", scene_name,
            ]
            ok = run_step(points_cmd, "3D point cloud generation", log_prefix)
            if not ok:
                failed_scenes.append(scene_name)
                continue

            # ── Stage 4: ground-truth costmap generation ────────────────────────
            costmap_cmd = [
                sys.executable, "generate_gt_costmap.py",
                "--out",   args.out,
                "--scene", scene_name,
                "--cost",  args.cost,
            ]
            if args.cost == "geodesic":
                costmap_cmd += ["--host", args.host, "--port", str(args.port)]
            ok = run_step(costmap_cmd, "GT costmap generation", log_prefix)
            if not ok:
                failed_scenes.append(scene_name)
                continue

            completed += 1
            print(f"[PROGRESS] ✓ {scene_name} fully completed "
                  f"({completed}/{total_scenes} done)")

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"[SUMMARY] {completed}/{total_scenes} scenes completed successfully.")
    print(f"[SUMMARY] Elapsed time: {elapsed/60:.1f} minutes.")
    if failed_scenes:
        print(f"[SUMMARY] {len(failed_scenes)} scene(s) FAILED:")
        for s in failed_scenes:
            print(f"    - {s}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mass-generate a CARLA navigation dataset across multiple "
                    "worlds and trajectories."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--out",  default="out",
                        help="Root output directory for all generated scenes.")

    parser.add_argument("--num_worlds", default=1, type=int,
                        help="Number of distinct static worlds to generate. "
                             "Default: 1.")
    parser.add_argument("--trajectories_per_world", default=10, type=int,
                        help="Number of trajectories to collect within each "
                             "world. Default: 10.")

    parser.add_argument("--use_grp", action="store_true",
                        help="Use generate_dataset_grp.py (GRP-planned route) "
                             "instead of generate_dataset.py (random-walk route).")

    parser.add_argument("--cost", default="groundplane",
                        choices=["euclidean3d", "groundplane", "geodesic"],
                        help="Cost type passed to generate_gt_costmap.py. "
                             "Default: groundplane.")

    parser.add_argument("--world_seed_base", default=1000, type=int,
                        help="Base seed for world generation. "
                             "world_seed = world_seed_base + world_index.")
    parser.add_argument("--traj_seed_base", default=0, type=int,
                        help="Base seed for trajectory generation. "
                             "traj_seed = traj_seed_base + world_index * "
                             "trajectories_per_world + traj_index.")

    main(parser.parse_args())
