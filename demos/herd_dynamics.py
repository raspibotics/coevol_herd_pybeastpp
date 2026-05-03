import numpy as np

from core.world.world_object import WorldObject
from core.agent.ffn_agent import EvolvableFFNAgent
from core.utils import ColourPalette, ColourType as CT
from core.evolve.evolver import Evolver
from core.sensor.implementation import proximity_sensor, nearest_angle_sensor, state_sensor, density_sensor
from core.simulation import Simulation
from core.evolve.genetic_algorithm import GeneticAlgorithm
from core.utils import GA_SELECTION_TYPE
from core.evolve.population import Population
from core.evolve.base import Group

IS_DEMO = True
DEMO_NAME = "Herd"
CLASS_NAME = "HerdSimulation"

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

        grass_range = 300
        vision_range = 100
        awareness_range = 50

        self.add_sensor("grass_angle", nearest_angle_sensor(Grass, grass_range))
        self.add_sensor("grass_left", proximity_sensor(Grass, (0.805*np.pi), vision_range, (0.2916*np.pi), True))
        self.add_sensor("grass_right", proximity_sensor(Grass, (0.805*np.pi), vision_range, -(0.2916*np.pi), True))
        self.add_sensor("sheep_angle", nearest_angle_sensor(Sheep, grass_range))
        #self.add_sensor("sheep_nearby", proximity_sensor(Sheep, (np.pi * 2), awareness_range, 0, True))
        #self.add_sensor("sheep_nearby", density_sensor(Sheep,(np.pi * 2), awareness_range, 0))
        self.add_sensor("sheep_left", proximity_sensor(Sheep, (0.805*np.pi), vision_range, (0.2916*np.pi), True))
        self.add_sensor("sheep_right", proximity_sensor(Sheep, (0.805*np.pi), vision_range, -(0.2916*np.pi), True))
        self.add_sensor("left_wolf", proximity_sensor(Wolf, (0.805*np.pi), vision_range, (0.2916*np.pi), True))
        self.add_sensor("right_wolf", proximity_sensor(Wolf, (0.805*np.pi), vision_range, -(0.2916*np.pi), True))
        # custom built state sensor
        self.add_sensor("is_afraid", state_sensor(Sheep))

        self.add_brain(10)

        self.solid = True
        self._min_speed = 10.0
        self._max_speed = 100.0
        self.radius = 10.0
        self.colour = ColourPalette[CT.BLUE]
    
    # Need some way to have these internal values count as sensor
    def toggleAfraid(self): 
        if (self.sensors["left_wolf"].output() > 0) or (self.sensors["right_wolf"].output()):
            self.sensors["is_afraid"].updateState(1.0) 
        else:
            self.sensors["is_afraid"].updateState(0.0) 
        

    def control(self):
        super().control()
        self.toggleAfraid()
        self.timer += 1
        for k in self.controls.keys():
            self.controls[k] = self.controls[k] + 0.5

    # how to implement immunity to being eaten when nearby?
    # even if we have sensor to track count of nearby sheep, how do we handle the eating logic if
    # wolf is what triggers the eaten and wolf wont be able to see out "security?"
    #Leave as is? could be interesting if wolves "push out" sheep till they can be eaten
    def on_collision(self, obj):
        if isinstance(obj, Grass):
            self.grass_found += 1
            self.weighted_grass += 500/(self.timer+1)
            self.timer = 0
            obj.eaten()
    
    def eaten(self):
        self.times_eaten += 1
        # From Agent, allows it to no longer appear
        self.dead = True
    
    # TODO: find more sophisticated fitness if not eaten (try time based solution to encourage active foraging?)
    def get_fitness(self):
        if self.times_eaten > 0:
            return 0
        else: 
            return self.weighted_grass / (self.grass_found + 1)
    
    def reset(self):
        self.grass_found= 0
        self.times_eaten = 0
        self.weighted_grass = 0
        self.timer = 0


# Just copy pase of predator from chase
class Wolf(EvolvableFFNAgent, Evolver):
    def __init__(self):
        EvolvableFFNAgent.__init__(self)
        Evolver.__init__(self)
        self.prey_eaten = 0
        range = 300.0
        
        self.add_sensor("left", proximity_sensor(Sheep, np.pi/4, range, np.pi/8, True))
        self.add_sensor("right", proximity_sensor(Sheep, np.pi/4, range, -np.pi/8, True))
        self._interaction_range = range
        
        self.add_brain(4)
        self.solid = True
        self.colour = ColourPalette[CT.RED]
        self._min_speed = 10.0
        self._max_speed = 110.0
        self.radius = 15.0
        
    def control(self):
        super().control()
        for k in self.controls.keys():
            self.controls[k] = 0.5 * (self.controls[k] + 1.0)
    
    def on_collision(self, other):
        if isinstance(other, Sheep):
            self.prey_eaten += 1
            other.eaten()
        super().on_collision(other)
    
    def get_fitness(self):
        return self.prey_eaten + 0.01
        

class HerdSimulation(Simulation):
    def __init__(self):
        super().__init__("Herd")
        
        self.generations = 200
        self.assessments = 1
        self.timesteps = 500
        
        population_sheep, population_wolf, population_grass = 10, 2, 40
        ga_sheep = GeneticAlgorithm(0.25, 0.1, selection=GA_SELECTION_TYPE.ROULETTE)
        ga_wolf = GeneticAlgorithm(0.25, 0.1, selection=GA_SELECTION_TYPE.ROULETTE)

        self.add("grass", Group(population_grass, Grass))
        self.add("sheep", Population(population_sheep, Sheep, ga_sheep, team_size=population_sheep))
        self.add("wolf", Population(population_wolf, Wolf, ga_wolf, team_size=population_wolf))
        

    def log_end_generation(self):
        sheep_averages = self.contents["sheep"].average_member_fitness()
        sheep_average = sum(sheep_averages) / len(sheep_averages)
        self.log.info(f"Average sheep Fitness: {sheep_average:.3f}")

        wolf_averages = self.contents["wolf"].average_member_fitness()
        wolf_average = sum(wolf_averages) / len(wolf_averages)
        self.log.info(f"Average wolf Fitness: {wolf_average:.3f}")