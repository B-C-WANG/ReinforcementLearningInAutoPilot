# Type: Private Author: Baochuan Wang

import sys

carla_egg_path="/home/wang/Desktop/carla/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg",
carla_pythonAPI_path="/home/wang/Desktop/carla/PythonAPI/carla",

sys.path.append(carla_egg_path)
sys.path.append(carla_pythonAPI_path)

#from carla.driving_benchmark.experiment_suites import CoRL2017, BasicExperimentSuite
import carla
from carla import driving_benchmark
