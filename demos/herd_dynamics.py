import numpy as np

from dataclasses import dataclass

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
from concurrent.futures import ProcessPoolExecutor, as_completed

IS_DEMO = True
DEMO_NAME = "Herd"
CLASS_NAME = "HerdSimulation"

@dataclass(frozen=True)
class HerdTestScenario:
    key: str
    scenario_name: str
    use_cluster_protection: bool
    use_is_afraid_sensor: bool
    fear_mode: str
    purpose: str


HERD_TEST_SCENARIOS = {
    "baseline": HerdTestScenario(
        key="baseline",
        scenario_name="baseline_no_cluster_no_fear",
        use_cluster_protection=False,
        use_is_afraid_sensor=False,
        fear_mode="none",
        purpose="Pure foraging + wolf hunting baseline",
    ),
    "test_1": HerdTestScenario(
        key="test_1",
        scenario_name="test_1_no_cluster_fear_wolf_seen",
        use_cluster_protection=False,
        use_is_afraid_sensor=True,
        fear_mode="wolf_seen",
        purpose="Tests whether fear alone improves survival",
    ),
    "test_2": HerdTestScenario(
        key="test_2",
        scenario_name="test_2_cluster_no_fear",
        use_cluster_protection=True,
        use_is_afraid_sensor=False,
        fear_mode="none",
        purpose="Tests whether cluster immunity alone creates survival advantage",
    ),
    "test_3": HerdTestScenario(
        key="test_3",
        scenario_name="test_3_cluster_fear_wolf_seen",
        use_cluster_protection=True,
        use_is_afraid_sensor=True,
        fear_mode="wolf_seen",
        purpose="Tests if fear of wolves and herd protection implies safety in numbers",
    ),
    "test_4": HerdTestScenario(
        key="test_4",
        scenario_name="test_4_cluster_fear_unsafe",
        use_cluster_protection=True,
        use_is_afraid_sensor=True,
        fear_mode="unsafe",
        purpose="Tests whether sheep cluster due to safety-seeking",
    ),
    "test_5": HerdTestScenario(
        key="test_5",
        scenario_name="test_5_cluster_fear_wolf_and_unsafe",
        use_cluster_protection=True,
        use_is_afraid_sensor=True,
        fear_mode="wolf_and_unsafe",
        purpose="Most biologically realistic condition",
    ),
}

DEFAULT_HERD_TEST = "test_5"
DEFAULT_SCENARIO = HERD_TEST_SCENARIOS[DEFAULT_HERD_TEST]
HERD_EXPERIMENT_TESTS = ["test_1", "test_2", "test_3", "test_4", "test_5"]

SELECTED_HERD_TESTS = [
    "baseline",  # no cluster protection, no is_afraid sensor, no fear trigger
    "test_1",    # no cluster protection, is_afraid sensor, wolf seen -> afraid
    "test_2",    # cluster protection, no is_afraid sensor, no fear trigger
    "test_3",    # cluster protection, is_afraid sensor, wolf seen -> afraid
    "test_4",    # cluster protection, is_afraid sensor, not protected -> afraid
    "test_5",    # cluster protection, is_afraid sensor, wolf seen and not protected -> afraid
]

SCENARIO_NAME = DEFAULT_SCENARIO.scenario_name
USE_CLUSTER_PROTECTION = DEFAULT_SCENARIO.use_cluster_protection
USE_IS_AFRAID_SENSOR = DEFAULT_SCENARIO.use_is_afraid_sensor
FEAR_MODE = DEFAULT_SCENARIO.fear_mode


def set_herd_test_scenario(scenario: HerdTestScenario | str) -> HerdTestScenario:
    if isinstance(scenario, str):
        try:
            scenario = HERD_TEST_SCENARIOS[scenario]
        except KeyError as exc:
            known_tests = ", ".join(HERD_TEST_SCENARIOS)
            raise ValueError(f"Unknown herd test '{scenario}'. Choose from: {known_tests}") from exc

    global SCENARIO_NAME, USE_CLUSTER_PROTECTION, USE_IS_AFRAID_SENSOR, FEAR_MODE
    SCENARIO_NAME = scenario.scenario_name
    USE_CLUSTER_PROTECTION = scenario.use_cluster_protection
    USE_IS_AFRAID_SENSOR = scenario.use_is_afraid_sensor
    FEAR_MODE = scenario.fear_mode

    return scenario


def herd_test_keys(selected_tests: list[HerdTestScenario | str] | None) -> list[str]:
    tests = SELECTED_HERD_TESTS if selected_tests is None else selected_tests
    return [set_herd_test_scenario(test).key for test in tests]

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
        if USE_IS_AFRAID_SENSOR:
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
    def __init__(self, scenario: HerdTestScenario | str = DEFAULT_HERD_TEST):
        self.scenario = set_herd_test_scenario(scenario)
        self.scenario_name = self.scenario.scenario_name
        self.display_name = f"Herd - {self.scenario.key.replace('_', ' ').title()}"

        super().__init__(self.display_name)
        
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

        self.csv_path = os.path.join(self.results_dir, f"{self.scenario_name}.csv")

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "generation",
                "scenario",
                "avg_sheep_fitness",
                "avg_wolf_fitness"
            ])
        
    def log_begin_assessment(self):
        self.log.info(
            f"Scenario: {self.scenario_name} | "
            f"Generation: {self._generation + 1}/{self.generations} | "
            f"Assessment: {self._assessment + 1}/{self.assessments}"
        )

    def log_end_generation(self):
        sheep_averages = self.contents["sheep"].average_member_fitness()
        sheep_average = sum(sheep_averages) / len(sheep_averages)
        self.log.info(f"Average sheep Fitness: {sheep_average:.3f} Count: {self._generation}")

        wolf_averages = self.contents["wolf"].average_member_fitness()
        wolf_average = sum(wolf_averages) / len(wolf_averages)
        self.log.info(f"Average wolf Fitness: {wolf_average:.3f} Count: {self._generation}")

        self.log.info(
            f"Scenario: {self.scenario_name} | "
            f"Generation: {self._generation + 1}/{self.generations} | "
            f"Average sheep fitness: {sheep_average:.3f} | "
            f"Average wolf fitness: {wolf_average:.3f}"
        )

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self._generation,
                self.scenario_name,
                sheep_average,
                wolf_average
            ])


def herd_test_demo_name(scenario: HerdTestScenario) -> str:
    return f"Herd - {scenario.key.replace('_', ' ').title()}"


def make_herd_simulation_class(scenario_key: str):
    scenario = HERD_TEST_SCENARIOS[scenario_key]

    class ConfiguredHerdSimulation(HerdSimulation):
        def __init__(self):
            super().__init__(scenario)

    ConfiguredHerdSimulation.__name__ = f"{scenario.key}_HerdSimulation"
    return ConfiguredHerdSimulation


DEMO_CLASSES = [
    (herd_test_demo_name(HERD_TEST_SCENARIOS[test]), make_herd_simulation_class(test))
    for test in SELECTED_HERD_TESTS
]


def _run_herd_test_no_render(
    test: str,
    runs: int | None,
    generations: int | None,
    assessments: int | None,
    timesteps: int | None,
) -> str:
    sim = HerdSimulation(test)
    scenario = sim.scenario

    if runs is not None:
        sim.runs = runs
    if generations is not None:
        sim.generations = generations
    if assessments is not None:
        sim.assessments = assessments
    if timesteps is not None:
        sim.timesteps = timesteps

    sim.log.info(f"Running herd test: {scenario.key} ({scenario.scenario_name})")
    sim.log.info(f"Purpose: {scenario.purpose}")
    sim.run_simulation(render=False, parallel=False)
    return scenario.scenario_name


def run_herd_tests_parallel(
    selected_tests: list[HerdTestScenario | str] | None = None,
    *,
    max_workers: int | None = None,
    runs: int | None = None,
    generations: int | None = None,
    assessments: int | None = None,
    timesteps: int | None = None,
) -> None:
    tests_to_run = herd_test_keys(selected_tests)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_herd_test_no_render,
                test,
                runs,
                generations,
                assessments,
                timesteps,
            ): test
            for test in tests_to_run
        }

        for future in as_completed(futures):
            test = futures[future]
            scenario_name = future.result()
            print(f"Finished herd test: {test} ({scenario_name})")


def run_herd_tests(
    selected_tests: list[HerdTestScenario | str] | None = None,
    *,
    render: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
    runs: int | None = None,
    generations: int | None = None,
    assessments: int | None = None,
    timesteps: int | None = None,
) -> None:
    tests_to_run = herd_test_keys(selected_tests)

    if render and len(tests_to_run) > 1:
        raise ValueError("Render mode can only run one herd test at a time.")

    if parallel and not render and len(tests_to_run) > 1:
        run_herd_tests_parallel(
            tests_to_run,
            max_workers=max_workers,
            runs=runs,
            generations=generations,
            assessments=assessments,
            timesteps=timesteps,
        )
        return

    for test in tests_to_run:
        sim = HerdSimulation(test)
        scenario = sim.scenario

        if runs is not None:
            sim.runs = runs
        if generations is not None:
            sim.generations = generations
        if assessments is not None:
            sim.assessments = assessments
        if timesteps is not None:
            sim.timesteps = timesteps

        sim.log.info(f"Running herd test: {scenario.key} ({scenario.scenario_name})")
        sim.log.info(f"Purpose: {scenario.purpose}")
        sim.run_simulation(render=render, parallel=parallel)
