"""
Stage 1 – CARLA Dataset Collection (GRP version)
=================================================
Drives an ego vehicle through N anchor waypoints, where BOTH anchor selection
AND the dense driving path are generated entirely by GlobalRoutePlanner.

Key difference from collect_dataset.py (random-walk version):
  - Anchors are picked by sampling random road waypoints and measuring their
    GRP route arc-length from the current position. Only waypoints whose
    route distance falls in [MIN_WP_DIST, MAX_WP_DIST] are accepted.
  - The same GRP route used to measure that distance is immediately reused
    as the dense driving path — no second planning call, no inconsistency.
  - The dense path is sampled at exactly GRP_RESOLUTION metres per step,
    which defines the timestep size.

Usage:
    python collect_dataset_grp.py --scene my_scene --host 127.0.0.1 --port 2000

Output layout:
    out/<scene_name>/
        images/               000000.png …
        images_depth/         000000.npy …   (float32, metres, 320×240)
        trajectory.npy        (K, 3)  anchor waypoints in world XYZ
        agent_states.npy      (N, 6)  x y z roll pitch yaw  per saved frame
        camera_intrinsics.npy (3, 3)
        camera_extrinsics.npy (N, 4, 4) camera-to-world per saved frame
"""

import argparse
import queue
import os
import random
import shutil

import numpy as np
import cv2

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
IMG_W = 320
IMG_H = 240
FOV   = 90.0            # degrees

FIXED_DELTA = 0.1       # seconds per simulation tick

GRP_RESOLUTION = 1.0    # metres between consecutive path waypoints (= timestep size)

MIN_WP_DIST = 30.0      # min GRP route arc-length between consecutive anchors (metres)
MAX_WP_DIST = 80.0      # max GRP route arc-length between consecutive anchors (metres)
NUM_ANCHORS = 4         # number of anchor waypoints after the start

CANDIDATE_POOL  = 200   # how many random road waypoints to sample when searching for
                        # an anchor — increase if finding anchors is slow or fails often
MAX_SEARCH_ITER = 5     # how many times to resample the pool before giving up

DEPTH_FAR = 1000.0      # CARLA default far-plane (metres)

# Camera offset relative to ego centre (metres / degrees)
CAM_X, CAM_Y, CAM_Z         = 1.5, 0.0, 2.4
CAM_ROLL, CAM_PITCH, CAM_YAW = 0.0, 0.0, 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers  (identical to collect_dataset.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_output_dirs(base: str, scene: str) -> dict[str, str]:
    paths = {
        "root":         os.path.join(base, scene),
        "images":       os.path.join(base, scene, "images"),
        "images_depth": os.path.join(base, scene, "images_depth"),
    }
    os.makedirs(paths["root"], exist_ok=True)
    for key in ("images", "images_depth"):
        if os.path.exists(paths[key]):
            shutil.rmtree(paths[key])
        os.makedirs(paths[key])
    return paths


def make_camera_intrinsics(w: int, h: int, fov: float) -> np.ndarray:
    fx = w / (2.0 * np.tan(np.radians(fov) / 2.0))
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0,  cx],
                     [0,  fx, cy],
                     [0,  0,  1 ]], dtype=np.float64)


def decode_depth_meters(depth_img: carla.Image) -> np.ndarray:
    """Decode CARLA's RGB-encoded depth image to float32 metres."""
    arr = np.frombuffer(depth_img.raw_data, dtype=np.uint8)
    arr = arr.reshape((depth_img.height, depth_img.width, 4))  # BGRA
    B = arr[:, :, 0].astype(np.float32)
    G = arr[:, :, 1].astype(np.float32)
    R = arr[:, :, 2].astype(np.float32)
    normalized = (R + G * 256.0 + B * 65536.0) / 16777215.0
    return (normalized * DEPTH_FAR).astype(np.float32)


def carla_transform_to_matrix(t: carla.Transform) -> np.ndarray:
    """Return 4×4 camera-to-world homogeneous matrix for a carla.Transform."""
    loc = t.location
    rot = t.rotation
    cy = np.cos(np.radians(rot.yaw));   sy = np.sin(np.radians(rot.yaw))
    cp = np.cos(np.radians(rot.pitch)); sp = np.sin(np.radians(rot.pitch))
    cr = np.cos(np.radians(rot.roll));  sr = np.sin(np.radians(rot.roll))
    R = np.array([
        [cp*cy,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [cp*sy,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ], dtype=np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3,  3] = [loc.x, loc.y, loc.z]
    return M


def route_arc_length(route: list) -> float:
    """Sum straight-line distances between consecutive GRP route waypoints."""
    total = 0.0
    for i in range(1, len(route)):
        a = route[i-1][0].transform.location
        b = route[i  ][0].transform.location
        total += a.distance(b)
    return total


# ──────────────────────────────────────────────────────────────────────────────
# GRP-based anchor selection
# ──────────────────────────────────────────────────────────────────────────────

def pick_anchors_and_routes(
        world_map: carla.Map,
        grp: GlobalRoutePlanner,
        start_wp: carla.Waypoint,
        n: int,
        min_d: float,
        max_d: float,
        rng: random.Random,
) -> tuple[list[carla.Waypoint], list[list]]:
    """
    Select n anchor waypoints and return both the anchors and the GRP routes
    that connect them.

    For each anchor:
      1. Sample CANDIDATE_POOL random road waypoints.
      2. Apply a cheap Euclidean pre-filter (avoids obvious rejects).
      3. Call grp.trace_route() and measure arc-length.
      4. Accept the first candidate whose arc-length is in [min_d, max_d].
      5. Reuse that exact route for driving — no second planning call.

    Returns
    -------
    anchors : list of carla.Waypoint   (length n)
    routes  : list of GRP route lists  (length n)
              routes[i] connects anchors[i-1] → anchors[i]
              (routes[0] connects start_wp → anchors[0])
    """
    # Pre-generate the pool of all drivable road waypoints once
    all_road_wps = world_map.generate_waypoints(GRP_RESOLUTION)

    anchors = []
    routes  = []
    current = start_wp

    for anchor_idx in range(n):
        found_wp    = None
        found_route = None

        for attempt in range(MAX_SEARCH_ITER):
            candidates = rng.sample(all_road_wps,
                                    min(CANDIDATE_POOL, len(all_road_wps)))

            for wp in candidates:
                # ── Euclidean pre-filter ──────────────────────────────────────
                # The road distance is always >= Euclidean distance, so if the
                # straight-line distance is already > max_d * 1.5 or < min_d * 0.4
                # it can never satisfy [min_d, max_d] and we skip the GRP call.
                euc = current.transform.location.distance(wp.transform.location)
                if euc < min_d * 0.4 or euc > max_d * 1.5:
                    continue

                # ── GRP route + arc-length check ──────────────────────────────
                try:
                    route = grp.trace_route(current.transform.location,
                                            wp.transform.location)
                except Exception:
                    continue

                if len(route) < 2:
                    continue

                arc = route_arc_length(route)
                if min_d <= arc <= max_d:
                    found_wp    = wp
                    found_route = route
                    break           # accept first valid candidate

            if found_wp is not None:
                break
            else:
                print(f"[WARN] Anchor {anchor_idx + 1}: no candidate found in "
                      f"attempt {attempt + 1}/{MAX_SEARCH_ITER}, resampling pool …")

        if found_wp is None:
            raise RuntimeError(
                f"Could not find anchor {anchor_idx + 1} with route distance "
                f"in [{min_d}, {max_d}] m after {MAX_SEARCH_ITER} attempts. "
                f"Try increasing CANDIDATE_POOL or MAX_SEARCH_ITER, or "
                f"widening MIN_WP_DIST / MAX_WP_DIST."
            )

        arc = route_arc_length(found_route)
        print(f"[INFO] Anchor {anchor_idx + 1}: "
              f"({found_wp.transform.location.x:.1f}, "
              f"{found_wp.transform.location.y:.1f})  "
              f"route_arc={arc:.1f} m  waypoints_in_segment={len(found_route)}")

        anchors.append(found_wp)
        routes.append(found_route)
        current = found_wp

    return anchors, routes


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    dirs = build_output_dirs(args.out, args.scene)
    rng  = random.Random(args.seed)
    print(f"[INFO] RNG seed: {args.seed}")

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world  = client.get_world()

    # ── Synchronous mode ──────────────────────────────────────────────────────
    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = FIXED_DELTA
    world.apply_settings(settings)

    actor_list     = []
    rgb_q:   queue.Queue = queue.Queue()
    depth_q: queue.Queue = queue.Queue()
    collision_flag = [False]

    try:
        bp_lib = world.get_blueprint_library()

        # ── Ego vehicle ───────────────────────────────────────────────────────
        ego_bp = bp_lib.find("vehicle.lincoln.mkz")
        ego_bp.set_attribute("role_name", "hero")

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points in this map.")
        spawn_tf = rng.choice(spawn_points)
        ego = world.spawn_actor(ego_bp, spawn_tf)
        actor_list.append(ego)
        print(f"[INFO] Ego spawned at {spawn_tf.location}")

        # ── RGB camera ────────────────────────────────────────────────────────
        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(IMG_W))
        rgb_bp.set_attribute("image_size_y", str(IMG_H))
        rgb_bp.set_attribute("fov",          str(FOV))
        rgb_bp.set_attribute("sensor_tick",  "0.0")

        cam_tf = carla.Transform(
            carla.Location(x=CAM_X, y=CAM_Y, z=CAM_Z),
            carla.Rotation(roll=CAM_ROLL, pitch=CAM_PITCH, yaw=CAM_YAW),
        )
        rgb_cam = world.spawn_actor(rgb_bp, cam_tf, attach_to=ego)
        actor_list.append(rgb_cam)
        rgb_cam.listen(lambda d: rgb_q.put(d))

        # ── Depth camera ──────────────────────────────────────────────────────
        depth_bp = bp_lib.find("sensor.camera.depth")
        depth_bp.set_attribute("image_size_x", str(IMG_W))
        depth_bp.set_attribute("image_size_y", str(IMG_H))
        depth_bp.set_attribute("fov",          str(FOV))
        depth_bp.set_attribute("sensor_tick",  "0.0")
        depth_cam = world.spawn_actor(depth_bp, cam_tf, attach_to=ego)
        actor_list.append(depth_cam)
        depth_cam.listen(lambda d: depth_q.put(d))

        # ── Collision sensor ──────────────────────────────────────────────────
        col_bp = bp_lib.find("sensor.other.collision")
        col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
        actor_list.append(col_sensor)
        col_sensor.listen(lambda _: collision_flag.__setitem__(0, True))

        # Warm up — give sensors one tick to initialise
        world.tick()

        # ── Route planning ────────────────────────────────────────────────────
        world_map = world.get_map()
        grp       = GlobalRoutePlanner(world_map, sampling_resolution=GRP_RESOLUTION)

        start_wp  = world_map.get_waypoint(
            ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        print(f"[INFO] Start: ({start_wp.transform.location.x:.1f}, "
              f"{start_wp.transform.location.y:.1f})  "
              f"yaw={start_wp.transform.rotation.yaw:.1f}°")

        # Pick anchors + reuse their routes directly for driving
        anchors, segment_routes = pick_anchors_and_routes(
            world_map, grp, start_wp,
            NUM_ANCHORS, MIN_WP_DIST, MAX_WP_DIST, rng,
        )

        # Save anchor positions as trajectory
        trajectory = np.array(
            [[w.transform.location.x,
              w.transform.location.y,
              w.transform.location.z] for w in anchors],
            dtype=np.float32,
        )
        np.save(os.path.join(dirs["root"], "trajectory.npy"), trajectory)
        print(f"[INFO] Trajectory anchor XYs:\n{trajectory[:, :2]}")

        # Flatten all segment routes into one ordered waypoint list
        # Each segment_routes[i] is a list of (carla.Waypoint, RoadOption) tuples
        path_waypoints = []
        for route in segment_routes:
            for wp, _ in route:
                path_waypoints.append(wp)
        print(f"[INFO] Dense path: {len(path_waypoints)} waypoints "
              f"at {GRP_RESOLUTION} m resolution.")

        # ── Camera intrinsics (constant for the whole scene) ──────────────────
        K = make_camera_intrinsics(IMG_W, IMG_H, FOV)
        np.save(os.path.join(dirs["root"], "camera_intrinsics.npy"), K)

        # ── Drive loop ────────────────────────────────────────────────────────
        agent_states:   list[np.ndarray] = []
        cam_extrinsics: list[np.ndarray] = []
        frame_idx = 0

        for wp in path_waypoints:

            # Reset collision flag and drain stale queue items BEFORE moving
            collision_flag[0] = False
            while not rgb_q.empty():
                try: rgb_q.get_nowait()
                except queue.Empty: break
            while not depth_q.empty():
                try: depth_q.get_nowait()
                except queue.Empty: break

            # Kinematic teleport to next waypoint + advance simulation
            ego.set_transform(wp.transform)
            world.tick()
            # Collision sensor fires its callback during world.tick() if hit

            # ── Collision check ───────────────────────────────────────────────
            if collision_flag[0]:
                try: rgb_q.get(timeout=1.0)
                except queue.Empty: pass
                try: depth_q.get(timeout=1.0)
                except queue.Empty: pass
                print(f"[WARN] Collision at frame {frame_idx} – skipping.")
                continue

            # ── Retrieve sensor data ──────────────────────────────────────────
            try:
                rgb_data   = rgb_q.get(timeout=2.0)
                depth_data = depth_q.get(timeout=2.0)
            except queue.Empty:
                print(f"[WARN] Sensor timeout at frame {frame_idx} – skipping.")
                continue

            # ── Save RGB ──────────────────────────────────────────────────────
            rgb_arr = np.frombuffer(rgb_data.raw_data, dtype=np.uint8)
            rgb_arr = rgb_arr.reshape((IMG_H, IMG_W, 4))[:, :, :3]
            cv2.imwrite(
                os.path.join(dirs["images"], f"{frame_idx:06d}.png"),
                rgb_arr[:, :, ::-1],   # RGB → BGR for OpenCV
            )

            # ── Save metric depth ─────────────────────────────────────────────
            depth_m = decode_depth_meters(depth_data)
            np.save(
                os.path.join(dirs["images_depth"], f"{frame_idx:06d}.npy"),
                depth_m,
            )

            # ── Ego state: x y z roll pitch yaw ──────────────────────────────
            tf  = ego.get_transform()
            loc, rot = tf.location, tf.rotation
            agent_states.append(np.array(
                [loc.x, loc.y, loc.z, rot.roll, rot.pitch, rot.yaw],
                dtype=np.float32,
            ))

            # ── Camera extrinsic: camera-to-world 4×4 ────────────────────────
            cam_extrinsics.append(
                carla_transform_to_matrix(rgb_cam.get_transform())
            )

            frame_idx += 1

        # ── Persist per-frame arrays ──────────────────────────────────────────
        if agent_states:
            np.save(os.path.join(dirs["root"], "agent_states.npy"),
                    np.stack(agent_states))
            np.save(os.path.join(dirs["root"], "camera_extrinsics.npy"),
                    np.stack(cam_extrinsics))
        else:
            print("[WARN] No valid frames were saved (all collisions?).")

        print(f"[INFO] Collection complete — {frame_idx} frames saved.")

    finally:
        settings = world.get_settings()
        settings.synchronous_mode    = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        for actor in reversed(actor_list):
            actor.destroy()
        print("[INFO] Actors destroyed. World restored.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CARLA dataset collection — GRP-planned trajectory version."
    )
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  default=2000, type=int)
    parser.add_argument("--out",   default="out")
    parser.add_argument("--scene", default="scene_00")
    parser.add_argument("--seed",  default=42, type=int,
                        help="RNG seed (default: 42).")
    main(parser.parse_args())
