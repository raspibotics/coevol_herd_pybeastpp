import numpy as np

from core.world.world_object import WorldObject
from core.agent.ffn_agent import EvolvableFFNAgent
from core.utils import ColourPalette, ColourType as CT
from core.evolve.evolver import Evolver
from core.sensor.implementation import cluster_sensor, proximity_sensor, nearest_angle_sensor, state_sensor
from core.sensor.function.evaluate import connected_component_size
from core.simulation import Simulation
from core.evolve.genetic_algorithm import GeneticAlgorithm
from core.utils import GA_SELECTION_TYPE
from core.evolve.population import Population
from core.evolve.base import Group

import csv
import os

IS_DEMO = True
DEMO_NAME = "Herd"
CLASS_NAME = "HerdSimulation"

# baseline_no_cluster_no_fear
# test_1_no_cluster_fear
# test_2_cluster_no_fear
# test_3_cluster_fear_unsafe
# test_4_cluster_fear_wolf_and_safety

# SCENARIO_NAME = "baseline_no_cluster_no_fear"
# USE_CLUSTER_PROTECTION = False
# USE_IS_AFRAID_SENSOR = False
# FEAR_MODE = "none"

# Test 1
# SCENARIO_NAME = "test_1_no_cluster_fear_wolf_seen"
# USE_CLUSTER_PROTECTION = False
# USE_IS_AFRAID_SENSOR = True
# FEAR_MODE = "wolf_seen"

# Test 2
# SCENARIO_NAME = "test_2_cluster_no_fear"
# USE_CLUSTER_PROTECTION = True
# USE_IS_AFRAID_SENSOR = False
# FEAR_MODE = "none"

# Test 3
# SCENARIO_NAME = "test_3_cluster_fear_wolf_seen"
# USE_CLUSTER_PROTECTION = True
# USE_IS_AFRAID_SENSOR = True
# FEAR_MODE = "wolf_seen"

# Test 4
# SCENARIO_NAME = "test_4_cluster_fear_unsafe"
# USE_CLUSTER_PROTECTION = True
# USE_IS_AFRAID_SENSOR = True
# FEAR_MODE = "unsafe"

# Test 5
SCENARIO_NAME = "test_5_cluster_fear_wolf_and_unsafe"
USE_CLUSTER_PROTECTION = True
USE_IS_AFRAID_SENSOR = True
FEAR_MODE = "wolf_and_unsafe"

class Grass(WorldObject):
    def __init__(self):
        super().__init__()
        self.radius = 5.0
        self.colour = ColourPalette[CT.GREEN]
    
    def eaten(self):
        self.location = self.world.random_location()

class Sheep(EvolvableFFNAgent, Evolver):
    def __init__(self):
        EvolvableFFNAgent.__init__(self)
        Evolver.__init__(self)

        self.grass_found = 0
        self.timer = 0
        self.weighted_grass = 0
        self.times_eaten = 0
        self.clustered_steps = 0
        self.age = 0

        # Fitness function parameters
        self.sheep_cluster_radius = 65.0
        self.min_sheep_cluster_size = 3
        self.sheep_fitness_floor = 0.001
        self.sheep_grass_reward = 8.0
        self.sheep_quick_grass_reward = 6.0
        self.sheep_cluster_reward = 0.03
        self.sheep_survival_reward = 10.0
        self.sheep_death_multiplier = 0.2
        self.sheep_recency_timesteps = 60.0

        grass_range = 300
        vision_range = 140

        self.add_sensor("grass_angle", nearest_angle_sensor(Grass, grass_range))
        self.add_sensor("grass_left", proximity_sensor(Grass, (0.805*np.pi), vision_range, (0.2916*np.pi), True))
        self.add_sensor("grass_right", proximity_sensor(Grass, (0.805*np.pi), vision_range, -(0.2916*np.pi), True))
        self.add_sensor("sheep_angle", nearest_angle_sensor(Sheep, grass_range))
        #self.add_sensor("sheep_nearby", proximity_sensor(Sheep, (np.pi * 2), awareness_range, 0, True))
        #self.add_sensor("sheep_nearby", density_sensor(Sheep,(np.pi * 2), awareness_range, 0))
        self.add_sensor("sheep_cluster", cluster_sensor(Sheep, self.sheep_cluster_radius, self.min_sheep_cluster_size))
        self.add_sensor("sheep_left", proximity_sensor(Sheep, (0.4028*np.pi), vision_range, (0.4028*np.pi), True))
        self.add_sensor("sheep_right", proximity_sensor(Sheep, (0.4028*np.pi), vision_range, -(0.4028*np.pi), True))
        self.add_sensor("left_wolf", proximity_sensor(Wolf, (0.805*np.pi), vision_range, (0.2916*np.pi), True))
        self.add_sensor("right_wolf", proximity_sensor(Wolf, (0.805*np.pi), vision_range, -(0.2916*np.pi), True))
        # custom built state sensor
        self.add_sensor("is_afraid", state_sensor(Sheep))

        self.add_brain(10)

        self.solid = True
        self._min_speed = 10.0
        self._max_speed = 80.0
        self.radius = 10.0
        self.colour = ColourPalette[CT.BLUE]
    
    # Need some way to have these internal values count as sensor
    # def toggleAfraid(self): 
    #     if (self.sensors["left_wolf"].output() > 0) or (self.sensors["right_wolf"].output()):
    #         self.sensors["is_afraid"].updateState(1.0) 
    #     else:
    #         self.sensors["is_afraid"].updateState(0.0) 
    #     if (self.protected_by_cluster()):
    #         self.sensors["is_afraid"].updateState(0.0)
    #     else: 
    #         self.sensors["is_afraid"].updateState(1.0)

    def toggleAfraid(self): 
        if not USE_IS_AFRAID_SENSOR:
            return

        wolf_seen = (
            self.sensors["left_wolf"].output() > 0
            or self.sensors["right_wolf"].output() > 0
        )

        unsafe = not self.protected_by_cluster()

        if FEAR_MODE == "wolf_seen":
            afraid = wolf_seen

        elif FEAR_MODE == "unsafe":
            afraid = unsafe

        elif FEAR_MODE == "wolf_and_unsafe":
            afraid = wolf_seen and unsafe

        else:
            afraid = False

        self.sensors["is_afraid"].updateState(1.0 if afraid else 0.0)
        

    def control(self):
        super().control()
        self.toggleAfraid()
        self.timer += 1
        self.age += 1
        if self.protected_by_cluster():
            self.clustered_steps += 1
        for k in self.controls.keys():
            self.controls[k] = self.controls[k] + 0.5

    def cluster_size(self) -> int:
        if self.world is None:
            return 1
        sheep = [
            agent for agent in self.world._agents
            if isinstance(agent, Sheep) and not getattr(agent, "dead", False)
        ]
        return connected_component_size(self, sheep, self.sheep_cluster_radius, self.world)

    # def protected_by_cluster(self) -> bool:
    #     return self.cluster_size() >= self.min_sheep_cluster_size

    def protected_by_cluster(self) -> bool:
        if not USE_CLUSTER_PROTECTION:
            return False
        return self.cluster_size() >= self.min_sheep_cluster_size

    def on_collision(self, obj):
        if isinstance(obj, Grass):
            self.grass_found += 1
            self.weighted_grass += 1.0 / (1.0 + self.timer / self.sheep_recency_timesteps)
            self.timer = 0
            obj.eaten()
    
    def eaten(self):
        self.times_eaten += 1
        # From Agent, allows it to no longer appear
        self.dead = True
    
    def get_fitness(self, ):
        if self.grass_found == 0:
            return self.sheep_fitness_floor

        forage_reward = self.sheep_grass_reward * self.grass_found
        speed_reward = self.sheep_quick_grass_reward * self.weighted_grass
        #cluster_reward = self.sheep_cluster_reward * self.clustered_steps
        #recency_multiplier = 1.0 / (1.0 + (self.timer / self.sheep_recency_timesteps) ** 2)
        #survival_reward = 0.0 if self.times_eaten > 0 else self.sheep_survival_reward
        death_multiplier = self.sheep_death_multiplier if self.times_eaten > 0 else 1.0
        return max(
            self.sheep_fitness_floor,
            ((forage_reward + speed_reward) ) * death_multiplier
        )
    
    def reset(self):
        self.grass_found= 0
        self.times_eaten = 0
        self.weighted_grass = 0
        self.clustered_steps = 0
        self.timer = 0
        self.age = 0


# Just copy pase of predator from chase
class Wolf(EvolvableFFNAgent, Evolver):
    def __init__(self):
        EvolvableFFNAgent.__init__(self)
        Evolver.__init__(self)
        self.prey_eaten = 0
        self.visual_tracking_score = 0.0
        self.age = 0
        range = 300.0

        self.wolf_kill_reward = 8.0
        
        self.add_sensor("angle", nearest_angle_sensor(Sheep, range))
        self.add_sensor("left", proximity_sensor(Sheep, (0.875*np.pi), range, (0.3403*np.pi), True))
        self.add_sensor("right", proximity_sensor(Sheep, (0.875*np.pi), range, -(0.3403*np.pi), True))
        self._interaction_range = range
        
        self.add_brain(6)
        self.solid = True
        self.colour = ColourPalette[CT.RED]
        self._min_speed = 10.0
        self._max_speed = 110.0
        self.radius = 15.0
        
    def control(self):
        super().control()
        left = self.sensors["left"].output()
        right = self.sensors["right"].output()
        self.visual_tracking_score += max(left, right)
        self.age += 1
        for k in self.controls.keys():
            self.controls[k] = 0.5 * (self.controls[k] + 1.0)
    
    def on_collision(self, other):
        if isinstance(other, Sheep):
            if not other.dead and not other.protected_by_cluster():
                self.prey_eaten += 1
                other.eaten()
        super().on_collision(other)
    
    def get_fitness(self):
        return (
            self.wolf_kill_reward * self.prey_eaten + 0.01
        )

    def reset(self):
        self.prey_eaten = 0
        self.visual_tracking_score = 0.0
        self.age = 0
        

class HerdSimulation(Simulation):
    def __init__(self):
        super().__init__("Herd")
        
        self.generations = 100
        self.assessments = 1
        self.timesteps = 500
        
        population_sheep, population_wolf, population_grass = 24, 4, 130
        ga_sheep = GeneticAlgorithm(0.25, 0.1, selection=GA_SELECTION_TYPE.ROULETTE)
        ga_wolf = GeneticAlgorithm(0.25, 0.1, selection=GA_SELECTION_TYPE.ROULETTE)

        self.add("grass", Group(population_grass, Grass))
        self.add("sheep", Population(population_sheep, Sheep, ga_sheep, team_size=population_sheep))
        self.add("wolf", Population(population_wolf, Wolf, ga_wolf, team_size=population_wolf))

        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        self.csv_path = os.path.join(self.results_dir, f"{SCENARIO_NAME}.csv")

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "generation",
                "scenario",
                "avg_sheep_fitness",
                "avg_wolf_fitness"
            ])
        

    def log_end_generation(self):
        sheep_averages = self.contents["sheep"].average_member_fitness()
        sheep_average = sum(sheep_averages) / len(sheep_averages)
        self.log.info(f"Average sheep Fitness: {sheep_average:.3f} Count: {self._generation}")

        wolf_averages = self.contents["wolf"].average_member_fitness()
        wolf_average = sum(wolf_averages) / len(wolf_averages)
        self.log.info(f"Average wolf Fitness: {wolf_average:.3f} Count: {self._generation}")

        self.log.info(
            f"Scenario: {SCENARIO_NAME} | "
            f"Generation: {self._generation} | "
            f"Average sheep fitness: {sheep_average:.3f} | "
            f"Average wolf fitness: {wolf_average:.3f}"
        )

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self._generation,
                SCENARIO_NAME,
                sheep_average,
                wolf_average
            ])
