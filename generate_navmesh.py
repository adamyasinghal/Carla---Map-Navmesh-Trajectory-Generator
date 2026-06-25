import carla
import numpy as np

resolution = 0.5 #in meters, can tweaked for a tighter grid but will be more computationally expensive

#grid size, can be tweaked
xmin = -200 
xmax = 200
ymin = -200
ymax = 200

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

client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.get_world()

print("Connected to : ", world.get_map().name)

#valid_hits = [] #commented out for now
grid = {}

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

        ix = int(round((x - xmin) / resolution))
        iy = int(round((y - ymin) / resolution))

        #if label not in ALLOWED_LABELS:
        #    continue
        #uncomment later

        grid[(ix, iy)] = {
        "location": np.array([hit.location.x, hit.location.y, hit.location.z], dtype=np.float32),
        "label": label,
        "normal": np.array([hit.normal.x, hit.normal.y, hit.normal.z], dtype=np.float32),
}

        #commented out for now
        '''
            valid_hits.append([
            hit.location.x,
            hit.location.y,
            hit.location.z
        ])
        '''

print(f"Total ray hits : {len(grid)}")

clearance = 1.5 #in meters, can be tweaked depending on size of ego vehicle
radius = int(np.ceil(clearance / resolution))

safe_points = []

for (ix, iy), cell in grid.items():

    if cell["label"] not in ALLOWED_LABELS:
        continue

    keep = True

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):

            if dx*dx + dy*dy > radius*radius:
                continue

            neighbor = grid.get((ix + dx, iy + dy))

            if neighbor is None:
                keep = False
                break

            if neighbor["label"] not in ALLOWED_LABELS:
                keep = False
                break

        if not keep:
            break

    if keep:
        normal = cell["normal"]
        loc = cell["location"]

        safe_points.append(np.concatenate((loc, normal)))

print(f"Safe points: {len(safe_points)}")

safe_points = np.array(safe_points, dtype=np.float32)

np.save("navmesh.npy", safe_points)

for p in safe_points:
    world.debug.draw_point(
        carla.Location(x=p[0], y=p[1], z=p[2]+0.05),
        size=0.05,
        color=carla.Color(0, 255, 0),
        life_time=60.0
    )