import carla
import random
import numpy as np

client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.get_world()
