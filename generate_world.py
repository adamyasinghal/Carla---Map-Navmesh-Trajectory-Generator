import argparse
import carla
import random

# TODO : check if the pedestrian seed function exists and works in this carla version (also remove any comments regarding that)
# TODO : check for collision of ego while moving in the lane and make it sidestep
# TODO : restrict the world to a specific region
# NOTE : do not enable traffic manager in any way as it will lead to dynamic environment and will break the controller

NUMBER_OF_TRAFFIC_CARS = 20  # can be modified as per use
NUMBER_OF_PEDESTRIANS  = 50  # can be modified as per use
MINIMUM_DISTANCE_BETWEEN_WAYPOINTS = 50  # minimum distance between two waypoints in meters
TIMESTEP_DISTANCE      = 0.5   # distance travelled by ego vehicle to be classified as one timestep
MAXIMUM_TIMESTEPS      = 1000  # maximum number of timesteps that can be present in one trajectory
MAXIMUM_TRAJECTORY_LENGTH = MAXIMUM_TIMESTEPS * TIMESTEP_DISTANCE  # max length of one trajectory


def destroy_existing_actors(world: carla.World) -> None:
    """
    Destroy all previously spawned vehicles and pedestrians (walkers) before
    spawning a new world. Without this, calling this script multiple times
    (e.g. from the mass generation orchestrator) accumulates actors instead
    of replacing them, and eventually exhausts spawn points.

    Does NOT touch sensors, the ego vehicle (if already spawned by a separate
    collection script), or any actor not matching 'vehicle.*' / 'walker.*'.

    Note: this script only ever runs BEFORE collect_dataset.py spawns the ego,
    so in normal pipeline order there is no ego to accidentally destroy here.
    If you run this manually while an ego is already in the world, it WILL
    also be destroyed since it matches 'vehicle.*' — be aware of that.
    """
    actors = world.get_actors()
    to_destroy = [a for a in actors if a.type_id.startswith("vehicle.")
                                     or a.type_id.startswith("walker.")]

    if not to_destroy:
        print("[INFO] No existing vehicles/walkers found — world is clean.")
        return

    print(f"[INFO] Destroying {len(to_destroy)} existing actor(s) "
          f"(vehicles + walkers) before respawning …")

    for actor in to_destroy:
        try:
            actor.destroy()
        except RuntimeError as e:
            print(f"[WARN] Failed to destroy actor {actor.id} ({actor.type_id}): {e}")

    print("[INFO] Cleanup complete.")


def main(args):
    random.seed(args.seed)  # controls vehicle/pedestrian spawn point selection — reproducible per seed

    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world = client.get_world()

    # ── Clean slate before spawning a new world ───────────────────────────────
    destroy_existing_actors(world)

    # world.set_pedestrians_seed(args.walker_seed) — not confirmed available in this
    # CARLA UE5 version; if it exists, it would make pedestrian nav-mesh spawn
    # points reproducible. Left available via --walker_seed but not assumed working.
    try:
        world.set_pedestrians_seed(args.walker_seed)
    except AttributeError:
        print("[WARN] world.set_pedestrians_seed() not available in this CARLA "
              "version — pedestrian spawn locations may not be fully reproducible.")

    # ── Traffic vehicles ───────────────────────────────────────────────────────
    traffic_vehicle_blueprints = sorted(
        world.get_blueprint_library().filter('*vehicle*'),
        key=lambda bp: bp.id
    )

    traffic_car_spawn_points = world.get_map().get_spawn_points()
    random.shuffle(traffic_car_spawn_points)  # reproducible per seed

    traffic_vehicles = []
    for i in range(args.num_vehicles):
        blueprint = traffic_vehicle_blueprints[i % len(traffic_vehicle_blueprints)]
        # TODO : could randomize blueprint selection using the seed as well
        transform = traffic_car_spawn_points[i % len(traffic_car_spawn_points)]
        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            traffic_vehicles.append(vehicle)
            print(f"Spawned {blueprint.id} at "
                  f"({transform.location.x:.2f}, {transform.location.y:.2f})")
        else:
            print(f"Failed to spawn vehicle at spawn point {i}")

    print(f"Spawned {len(traffic_vehicles)} vehicles")

    # ── Pedestrians (walkers) ─────────────────────────────────────────────────
    walker_blueprints = sorted(
        world.get_blueprint_library().filter('walker.pedestrian.*'),
        key=lambda bp: bp.id
    )

    walker_spawn_points = []
    attempts = 0
    max_attempts = args.num_pedestrians * 20  # safety cap against infinite loop
    while len(walker_spawn_points) < args.num_pedestrians and attempts < max_attempts:
        loc = world.get_random_location_from_navigation()
        # get_random_location_from_navigation() is not directly seedable;
        # set_pedestrians_seed() above may or may not influence it depending
        # on CARLA version. random.shuffle() below reduces deviation regardless.
        if loc:
            walker_spawn_points.append(carla.Transform(loc))
        attempts += 1

    if len(walker_spawn_points) < args.num_pedestrians:
        print(f"[WARN] Only found {len(walker_spawn_points)}/"
              f"{args.num_pedestrians} valid pedestrian spawn points "
              f"after {attempts} attempts.")

    random.shuffle(walker_spawn_points)  # minimizes deviation across runs with same seed

    walkers = []
    for i in range(len(walker_spawn_points)):
        bp = walker_blueprints[i % len(walker_blueprints)]
        walker = world.try_spawn_actor(bp, walker_spawn_points[i])
        if walker:
            walkers.append(walker)

    print(f"Spawned {len(walkers)} pedestrians")
    print(f"[INFO] World generation complete. seed={args.seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spawn static traffic vehicles and pedestrians in an "
                    "already-running CARLA world."
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--seed", default=42, type=int,
                        help="Seed controlling vehicle/pedestrian spawn point "
                             "selection (default: 42).")
    parser.add_argument("--walker_seed", default=42, type=int,
                        help="Seed passed to world.set_pedestrians_seed() if "
                             "available in this CARLA version (default: 42).")
    parser.add_argument("--num_vehicles", default=NUMBER_OF_TRAFFIC_CARS, type=int,
                        help=f"Number of traffic vehicles to spawn "
                             f"(default: {NUMBER_OF_TRAFFIC_CARS}).")
    parser.add_argument("--num_pedestrians", default=NUMBER_OF_PEDESTRIANS, type=int,
                        help=f"Number of pedestrians to spawn "
                             f"(default: {NUMBER_OF_PEDESTRIANS}).")

    main(parser.parse_args())
