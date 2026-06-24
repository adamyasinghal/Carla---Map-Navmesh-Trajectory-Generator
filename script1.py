import carla
import random

#TODO : take the seeds as input using argument parser
#TODO : check if the pedestrian seed function exists and works in this carla version (also remove any comments regarding that)
#TODO : check for collision of ego while moving in the lane and make it sidestep
#TODO : restrict the world to a specific region

#NOTE : do not enable traffic manager in any way as it will lead to dynamic environment and will break the controller

random.seed(42) #can be changed to generate different looking world
NUMBER_OF_TRAFFIC_CARS = 20 #can be modified as per use
NUMBER_OF_PEDESTRIANS = 50 #can be modified as per use
CARLA_WALKER_SEED = 42 #?? needs more investigation as to how this will work
MINIMUM_DISTANCE_BETWEEN_WAYPOINTS = 50 #minimum distance between two waypoints in meters
TIMESTEP_DISTANCE = 0.5 #distance travelled by ego vehicle to be classified as one timestep
MAXIMUM_TIMESTEPS = 1000 #maximum number of timesteps that can be present in on trajectory
MAXIMUM_TRAJECTORY_LENGTH = MAXIMUM_TIMESTEPS * TIMESTEP_DISTANCE #maximum length of one trajectory that ego vehicle follows

client = carla.Client('localhost', 2000)
world = client.get_world()

world.set_pedestrians_seed(CARLA_WALKER_SEED) #not sure if available in carla unreal engine 5
#world.set_pedestrians_seed could make spawning of pedestrians reproducible if it exists in this version


traffic_vehicle_blueprints = sorted(
    world.get_blueprint_library().filter('*vehicle*'), 
    key=lambda bp: bp.id
)

traffic_car_spawn_points = world.get_map().get_spawn_points()

random.shuffle(traffic_car_spawn_points) #shuffles spawn points depending on the seed to generate a reproducible world

traffic_vehicles = []

for i in range(NUMBER_OF_TRAFFIC_CARS):
    blueprint = traffic_vehicle_blueprints[i % len(traffic_vehicle_blueprints)]
    #TODO : could randomize the selection of blueprint using a random seed
    transform = traffic_car_spawn_points[i]

    vehicle = world.try_spawn_actor(blueprint, transform)

    if vehicle is not None:
        traffic_vehicles.append(vehicle)
        print(
            f"Spawned {blueprint.id}"
            f"at ({transform.location.x:.2f}, "
            f"{transform.location.y:.2f})"
        )
    else:
        print(f"Failed to spawn vehicle at spawn point {i}")
 
print(f"Spawned {len(traffic_vehicles)} vehicles")

#here walker is synonym to pedestrian

walker_blueprints = sorted(
    world.get_blueprint_library().filter('walker.pedestrian.*'),
    key=lambda bp: bp.id
)

walker_spawn_points = []

while len(walker_spawn_points) < NUMBER_OF_PEDESTRIANS :
    loc = world.get_random_location_from_navigation()
    #unfortunately this function is random and cannot be seeded, thus leading to breakage in determinism and reproducibility
    #the above comment might not be entirely due and this function might be reproducible because of pedestrians seed defined earlier
    #we can still try to minimize devation by sorting or using seeded random.shuffle later before spawning pedestrians
    if loc:
        walker_spawn_points.append(carla.Transform(loc))

random.shuffle(walker_spawn_points) #minimizing devation in pedestrians

walkers = []

for i in range(0, NUMBER_OF_PEDESTRIANS) :
    bp = walker_blueprints[i % len(walker_blueprints)]

    walker = world.try_spawn_actor(
        bp,
        walker_spawn_points[i]
    )

    if walker:
        walkers.append(walker)

print(f"Spawned {len(walkers)} pedestrians")

