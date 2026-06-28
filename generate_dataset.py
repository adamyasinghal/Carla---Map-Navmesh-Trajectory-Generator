"""
Stage 1 – CARLA Dataset Collection
====================================
Drives an ego vehicle through 4 random road waypoints (30-80 m apart),
capturing synchronized RGB + metric depth + ego states at 10 Hz (0.1 s ticks,
target speed 5 m/s).

Anchor waypoints and dense paths are both generated via GlobalRoutePlanner
so the ego follows real road geometry including turns and junctions.

Usage:
    python collect_dataset.py --scene my_scene --host 127.0.0.1 --port 2000

Output layout:
    out/<scene_name>/
        images/              000000.png …
        images_depth/        000000.npy …   (float32, metres, 320×240)
        trajectory.npy       (K, 3)  anchor waypoints in world XYZ
        agent_states.npy     (N, 6)  x y z roll pitch yaw  per saved frame
        camera_intrinsics.npy (3, 3)
        camera_extrinsics.npy (N, 4, 4) camera-to-world per saved frame
"""

import argparse
import queue
import os
import time
import random

import numpy as np
import cv2

import sys

sys.path.append(
    "/scratch2/adamya.singhal/carla/Carla-0.10.0-Linux-Shipping/PythonAPI/carla"
)

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
IMG_W = 320
IMG_H = 240
FOV   = 90.0          # degrees

FIXED_DELTA = 0.1     # seconds per tick  → 10 Hz
TARGET_SPEED = 5.0    # m/s

MIN_WP_DIST = 30.0    # metres  (road-distance between consecutive anchors)
MAX_WP_DIST = 80.0
NUM_ANCHORS = 4       # intermediate waypoints (plus ego start = 5 stops total)

DEPTH_FAR   = 1000.0  # CARLA default far-plane (metres)

# Camera offset relative to ego (metres / degrees)
CAM_X, CAM_Y, CAM_Z     = 1.5, 0.0, 2.4   # forward, right, up
CAM_ROLL, CAM_PITCH, CAM_YAW = 0.0, 0.0, 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_output_dirs(base: str, scene: str) -> dict[str, str]:
    paths = {
        "root":         os.path.join(base, scene),
        "images":       os.path.join(base, scene, "images"),
        "images_depth": os.path.join(base, scene, "images_depth"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def make_camera_intrinsics(w: int, h: int, fov: float) -> np.ndarray:
    fx = w / (2.0 * np.tan(np.radians(fov) / 2.0))
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[fx, 0,  cx],
                     [0,  fy, cy],
                     [0,  0,  1 ]], dtype=np.float64)


def decode_depth_meters(depth_img: carla.Image) -> np.ndarray:
    """
    CARLA encodes metric depth in the R, G, B channels of a 'sensor.camera.depth'
    image saved in raw format (not converted to logarithmic).

    Formula (from CARLA docs):
        normalized = (R + G*256 + B*256²) / (256³ - 1)
        depth_m    = normalized * 1000.0
    """
    array = np.frombuffer(depth_img.raw_data, dtype=np.uint8)
    array = array.reshape((depth_img.height, depth_img.width, 4))  # BGRA
    B = array[:, :, 0].astype(np.float32)
    G = array[:, :, 1].astype(np.float32)
    R = array[:, :, 2].astype(np.float32)
    normalized = (R + G * 256.0 + B * 65536.0) / 16777215.0       # 256³-1
    return (normalized * DEPTH_FAR).astype(np.float32)


def carla_transform_to_matrix(t: carla.Transform) -> np.ndarray:
    """Return 4×4 homogeneous matrix for a carla.Transform (world convention)."""
    loc = t.location
    rot = t.rotation          # degrees, Unreal convention
    cy = np.cos(np.radians(rot.yaw))
    sy = np.sin(np.radians(rot.yaw))
    cp = np.cos(np.radians(rot.pitch))
    sp = np.sin(np.radians(rot.pitch))
    cr = np.cos(np.radians(rot.roll))
    sr = np.sin(np.radians(rot.roll))

    # Standard ZYX rotation (Unreal / CARLA world axes)
    R = np.array([
        [cp*cy,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [cp*sy,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ], dtype=np.float64)

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3,  3] = [loc.x, loc.y, loc.z]
    return M


def camera_to_world_matrix(ego_tf: carla.Transform,
                            cam_tf: carla.Transform) -> np.ndarray:
    """Compose ego-to-world and camera-to-ego to get camera-to-world."""
    M_ego  = carla_transform_to_matrix(ego_tf)
    M_cam  = carla_transform_to_matrix(cam_tf)   # cam relative to ego
    return M_ego @ M_cam


def sensor_callback(data, q: queue.Queue):
    q.put(data)


def pick_anchor_waypoints(start_wp: carla.Waypoint,
                          n: int,
                          min_d: float,
                          max_d: float,
                          rng: random.Random) -> list[carla.Waypoint]:
    """
    Select n consecutive road anchors by walking the waypoint graph forward,
    taking a RANDOM branch whenever a junction is reached.

    This guarantees:
      - the ego always stays on drivable road surfaces
      - turns are naturally taken whenever the road network branches
      - arc-length between consecutive anchors is in [min_d, max_d]

    Strategy
    --------
    From the current waypoint, walk forward 1 m at a time.
    At every junction (len(nexts) > 1) randomly pick a branch.
    When accumulated distance reaches a random target in [min_d, max_d],
    record the current waypoint as the next anchor and reset the counter.
    """
    anchors  = []
    current  = start_wp
    STEP     = 1.0   # metres per graph step

    for _ in range(n):
        target_d = rng.uniform(min_d, max_d)
        walked   = 0.0
        wp       = current

        while walked < target_d:
            nexts = wp.next(STEP)
            if not nexts:
                print("[WARN] Road ended before reaching target distance.")
                break

            if len(nexts) > 1:
                # Junction — pick a random branch so we actually turn
                wp = rng.choice(nexts)
                print(f"[INFO] Junction with {len(nexts)} branches — "
                      f"took branch yaw={wp.transform.rotation.yaw:.1f}°")
            else:
                wp = nexts[0]

            walked += STEP

        anchors.append(wp)
        current = wp
        print(f"[INFO] Anchor {len(anchors)}: "
              f"({wp.transform.location.x:.1f}, {wp.transform.location.y:.1f}) "
              f"walked={walked:.1f} m  target={target_d:.1f} m")

    return anchors


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    dirs = build_output_dirs(args.out, args.scene)
    rng  = random.Random(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world  = client.get_world()

    # ── Synchronous mode ──────────────────────────────────────────────────────
    settings = world.get_settings()
    settings.synchronous_mode   = True
    settings.fixed_delta_seconds = FIXED_DELTA
    world.apply_settings(settings)

    actor_list  = []
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
            raise RuntimeError("No spawn points available in this map.")
        spawn_tf = rng.choice(spawn_points)
        ego = world.spawn_actor(ego_bp, spawn_tf)
        actor_list.append(ego)
        print(f"[INFO] Ego spawned at {spawn_tf.location}")

        # ── RGB camera ────────────────────────────────────────────────────────
        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x",  str(IMG_W))
        rgb_bp.set_attribute("image_size_y",  str(IMG_H))
        rgb_bp.set_attribute("fov",           str(FOV))
        rgb_bp.set_attribute("sensor_tick",   "0.0")   # every tick in sync mode

        cam_tf = carla.Transform(
            carla.Location(x=CAM_X, y=CAM_Y, z=CAM_Z),
            carla.Rotation(roll=CAM_ROLL, pitch=CAM_PITCH, yaw=CAM_YAW),
        )
        rgb_cam = world.spawn_actor(rgb_bp, cam_tf, attach_to=ego)
        actor_list.append(rgb_cam)
        rgb_cam.listen(lambda d: sensor_callback(d, rgb_q))

        # ── Depth camera ──────────────────────────────────────────────────────
        depth_bp = bp_lib.find("sensor.camera.depth")
        depth_bp.set_attribute("image_size_x",  str(IMG_W))
        depth_bp.set_attribute("image_size_y",  str(IMG_H))
        depth_bp.set_attribute("fov",           str(FOV))
        depth_bp.set_attribute("sensor_tick",   "0.0")
        depth_cam = world.spawn_actor(depth_bp, cam_tf, attach_to=ego)
        actor_list.append(depth_cam)
        depth_cam.listen(lambda d: sensor_callback(d, depth_q))

        # ── Collision sensor ──────────────────────────────────────────────────
        col_bp = bp_lib.find("sensor.other.collision")
        col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
        actor_list.append(col_sensor)
        col_sensor.listen(lambda _: collision_flag.__setitem__(0, True))

        # Give sensors one tick to initialise
        world.tick()

        # ── Waypoints & route ─────────────────────────────────────────────────
        world_map  = world.get_map()
        grp        = GlobalRoutePlanner(world_map, sampling_resolution=1.0)
        start_wp   = world_map.get_waypoint(ego.get_location(),
                                            project_to_road=True,
                                            lane_type=carla.LaneType.Driving)
        print(f"[INFO] Start waypoint: "
              f"({start_wp.transform.location.x:.1f}, {start_wp.transform.location.y:.1f}) "
              f"yaw={start_wp.transform.rotation.yaw:.1f}°")

        anchors    = pick_anchor_waypoints(start_wp,
                                           NUM_ANCHORS, MIN_WP_DIST, MAX_WP_DIST, rng)
        trajectory = np.array([[w.transform.location.x,
                                 w.transform.location.y,
                                 w.transform.location.z] for w in anchors],
                               dtype=np.float32)
        np.save(os.path.join(dirs["root"], "trajectory.npy"), trajectory)
        print(f"[INFO] Trajectory anchor XYs: {trajectory[:, :2].tolist()}")

        # ── Camera intrinsics (constant) ──────────────────────────────────────
        K = make_camera_intrinsics(IMG_W, IMG_H, FOV)
        np.save(os.path.join(dirs["root"], "camera_intrinsics.npy"), K)

        # ── Build dense path via GRP: start → a1 → a2 → a3 → a4 ─────────────
        # GRP returns (waypoint, RoadOption) tuples spaced ~1 m apart,
        # following true road geometry through junctions.
        path_waypoints = []
        stops = [start_wp] + anchors
        for i in range(len(stops) - 1):
            src = stops[i].transform.location
            dst = stops[i+1].transform.location
            segment = grp.trace_route(src, dst)
            for wp, road_opt in segment:
                path_waypoints.append(wp)

        print(f"[INFO] Dense path: {len(path_waypoints)} waypoints total.")

        # ── Drive loop ────────────────────────────────────────────────────────
        agent_states:    list[np.ndarray] = []
        cam_extrinsics:  list[np.ndarray] = []
        frame_idx = 0

        for wp in path_waypoints:
            # Teleport kinematically to each waypoint on the dense 0.5 m grid.
            # Because the grid is dense the motion appears perfectly continuous.
            ego.set_transform(wp.transform)
            collision_flag[0] = False   # reset before tick

            world.tick()

            # ── Collision check ───────────────────────────────────────────────
            if collision_flag[0]:
                # Drain sensors so queues don't fill up
                try:
                    rgb_q.get(timeout=1.0)
                    depth_q.get(timeout=1.0)
                except queue.Empty:
                    pass
                print(f"[WARN] Collision at waypoint {frame_idx} – skipping frame.")
                continue

            # ── Retrieve sensor data ──────────────────────────────────────────
            try:
                rgb_data   = rgb_q.get(timeout=2.0)
                depth_data = depth_q.get(timeout=2.0)
            except queue.Empty:
                print(f"[WARN] Sensor timeout at frame {frame_idx} – skipping.")
                continue

            # ── Save RGB ──────────────────────────────────────────────────────
            rgb_array = np.frombuffer(rgb_data.raw_data, dtype=np.uint8)
            rgb_array = rgb_array.reshape((IMG_H, IMG_W, 4))[:, :, :3]  # drop alpha
            rgb_bgr   = rgb_array[:, :, ::-1]                            # RGB→BGR for cv2
            img_path  = os.path.join(dirs["images"], f"{frame_idx:06d}.png")
            cv2.imwrite(img_path, rgb_bgr)

            # ── Save metric depth ─────────────────────────────────────────────
            depth_m   = decode_depth_meters(depth_data)
            depth_path = os.path.join(dirs["images_depth"], f"{frame_idx:06d}.npy")
            np.save(depth_path, depth_m)

            # ── Ego state: x y z roll pitch yaw ──────────────────────────────
            tf  = ego.get_transform()
            rot = tf.rotation
            loc = tf.location
            agent_states.append(np.array([
                loc.x, loc.y, loc.z,
                rot.roll, rot.pitch, rot.yaw,
            ], dtype=np.float32))

            # ── Camera extrinsic (camera-to-world, 4×4) ───────────────────────
            # The camera's absolute world transform is what CARLA tracks for us.
            cam_world_tf = rgb_cam.get_transform()
            M = carla_transform_to_matrix(cam_world_tf)
            cam_extrinsics.append(M)

            frame_idx += 1

        # ── Persist per-frame arrays ───────────────────────────────────────────
        np.save(os.path.join(dirs["root"], "agent_states.npy"),
                np.stack(agent_states, axis=0))
        np.save(os.path.join(dirs["root"], "camera_extrinsics.npy"),
                np.stack(cam_extrinsics, axis=0))

        print(f"[INFO] Collection complete. {frame_idx} frames saved.")

    finally:
        # ── Restore async mode and destroy actors ─────────────────────────────
        settings = world.get_settings()
        settings.synchronous_mode    = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        for actor in reversed(actor_list):
            actor.destroy()
        print("[INFO] Actors destroyed. World restored.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  default=2000, type=int)
    parser.add_argument("--out",   default="out")
    parser.add_argument("--scene", default="scene_00")
    parser.add_argument("--seed",  default=42, type=int)
    main(parser.parse_args())