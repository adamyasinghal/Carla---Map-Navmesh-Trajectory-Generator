import carla
import random
import numpy as np

resolution = 0.5 #in meters, can tweaked for a tighter grid but will be more computationally expensive

#grid size, can be tweaked
xmin = -300 
xmax = 300
ymin = -300
ymax = 300

#ray parameters
ray_start_height = 100.0
ray_end_height = -20.0

ALLOWED_LABELS = {
    carla.CityObjectLabel.Road,
    carla.CityObjectLabel.Sidewalk,
    carla.CityObjectLabel.Crosswalk,
    carla.CityObjectLabel.Parking,
    carla.CityObjectLabel.Shoulder,
}

carla = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.get_world()

print("Connected to : ", world.get_map.name)

valid_hits = []

num_x = len(np.arange(xmin, xmax, resolution))
num_y = len(np.arange(ymin, ymax, resolution))

print(f"Sampling {num_x * num_y} grid cells...")

for x in np.arange(xmin, xmax, resolution):
    for y in np.arange(ymin, ymax, resolution):

        start = carla.Location(
            x = float(x),
            y = float(y),
            z = ray_start_height
        )

        end = carla.Location(
            x = float(x),
            y = float(y),
            z = ray_end_height
        )

        hits = world.cast_ray(start, end)

        if len(hits) == 0:
            continue

        hit = hits[0]
        
        label = hit.label
        if label not in ALLOWED_LABELS:
            continue

        valid_hits.append([
            hit.location.x,
            hit.location.y,
            hit.location.z
        ])

print(f"Total ray hists : {len(valid_hits)}")