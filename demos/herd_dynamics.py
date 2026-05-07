import numpy as np

from dataclasses import dataclass

from core.world.world_object import WorldObject
from core.agent.ffn_agent import EvolvableFFNAgent
from core.utils import ColourPalette, ColourType as CT
from core.evolve.evolver import Evolver
from core.sensor.implementation import cluster_sensor, proximity_sensor, nearest_angle_sensor, state_sensor
from core.sensor.function.evaluate import connected_component_size
from core.simulation import DEFAULT_RANDOM_SEED, Simulation
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


@dataclass(frozen=True)
class HerdFitnessConfig:
    key: str
    slug: str
    label: str
    grass_reward: float
    quick_grass_reward: float
    survival_reward: float
    death_multiplier: float
    purpose: str


@dataclass(frozen=True)
class HerdExperimentCase:
    scenario_key: str
    fitness_key: str


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

HERD_FITNESS_CONFIGS = {
    "current": HerdFitnessConfig(
        key="current",
        slug="fit_current",
        label="Current fitness",
        grass_reward=8.0,
        quick_grass_reward=6.0,
        survival_reward=0.0,
        death_multiplier=0.2,
        purpose="Current forage-first fitness with a death multiplier.",
    ),
    "survival_bonus": HerdFitnessConfig(
        key="survival_bonus",
        slug="fit_survival_bonus",
        label="Survival bonus",
        grass_reward=8.0,
        quick_grass_reward=6.0,
        survival_reward=40.0,
        death_multiplier=0.2,
        purpose="Adds reward for surviving without directly rewarding clustering.",
    ),
    "strong_survival": HerdFitnessConfig(
        key="strong_survival",
        slug="fit_strong_survival",
        label="Strong survival pressure",
        grass_reward=6.0,
        quick_grass_reward=4.0,
        survival_reward=80.0,
        death_multiplier=0.05,
        purpose="Reduces pure foraging pressure and strongly penalises predation.",
    ),
}

DEFAULT_HERD_TEST = "test_5"
DEFAULT_SCENARIO = HERD_TEST_SCENARIOS[DEFAULT_HERD_TEST]
DEFAULT_FITNESS = "current"
DEFAULT_FITNESS_CONFIG = HERD_FITNESS_CONFIGS[DEFAULT_FITNESS]
HERD_EXPERIMENT_TESTS = ["test_1", "test_2", "test_3", "test_4", "test_5"]
HERD_EXPERIMENT_FITNESS_CONFIGS = ["current", "survival_bonus", "strong_survival"]
HERD_EXPERIMENT_CASES = [
    HerdExperimentCase(scenario_key, fitness_key)
    for scenario_key in HERD_EXPERIMENT_TESTS
    for fitness_key in HERD_EXPERIMENT_FITNESS_CONFIGS
]

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
FITNESS_CONFIG = DEFAULT_FITNESS_CONFIG


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


def set_herd_fitness_config(fitness: HerdFitnessConfig | str) -> HerdFitnessConfig:
    if isinstance(fitness, str):
        try:
            fitness = HERD_FITNESS_CONFIGS[fitness]
        except KeyError as exc:
            known_configs = ", ".join(HERD_FITNESS_CONFIGS)
            raise ValueError(f"Unknown herd fitness '{fitness}'. Choose from: {known_configs}") from exc

    global FITNESS_CONFIG
    FITNESS_CONFIG = fitness

    return fitness


def herd_experiment_cases(
    selected_tests: list[HerdExperimentCase | HerdTestScenario | str | tuple[str, str]] | None
) -> list[HerdExperimentCase]:
    tests = HERD_EXPERIMENT_CASES if selected_tests is None else selected_tests
    cases = []

    for test in tests:
        if isinstance(test, HerdExperimentCase):
            scenario = set_herd_test_scenario(test.scenario_key)
            fitness = set_herd_fitness_config(test.fitness_key)
        elif isinstance(test, tuple):
            scenario = set_herd_test_scenario(test[0])
            fitness = set_herd_fitness_config(test[1])
        else:
            scenario = set_herd_test_scenario(test)
            fitness = set_herd_fitness_config(DEFAULT_FITNESS)

        cases.append(HerdExperimentCase(scenario.key, fitness.key))

    return cases

# Grass Class. Simple World Object
class Grass(WorldObject):
    def __init__(self):
        super().__init__()
        self.radius = 5.0
        self.colour = ColourPalette[CT.GREEN]
    
    def eaten(self):
        self.location = self.world.random_location()

# Sheep Agent Class
class Sheep(EvolvableFFNAgent, Evolver):
    def __init__(self):
        EvolvableFFNAgent.__init__(self)
        Evolver.__init__(self)

        # Initialisation of tracked variables
        self.grass_found = 0
        self.timer = 0
        self.weighted_grass = 0
        self.times_eaten = 0
        self.clustered_steps = 0
        self.wolf_seen_steps = 0
        self.afraid_steps = 0
        self.age = 0

        # Fitness function parameters
        self.sheep_cluster_radius = 65.0
        self.min_sheep_cluster_size = 3
        self.sheep_fitness_floor = 0.001
        self.sheep_recency_timesteps = 60.0

        # Range parameters
        grass_range = 300
        vision_range = 140
        vision_cone_angle = 0.805*np.pi
        vision_cone_pos = 0.2916*np.pi

        # Sensor Initialisation
        self.add_sensor("grass_angle", nearest_angle_sensor(Grass, grass_range))
        self.add_sensor("grass_left", proximity_sensor(Grass, vision_cone_angle, vision_range, vision_cone_pos, True))
        self.add_sensor("grass_right", proximity_sensor(Grass, vision_cone_angle, vision_range, -vision_cone_pos, True))
        self.add_sensor("sheep_angle", nearest_angle_sensor(Sheep, grass_range))
        self.add_sensor("sheep_cluster", cluster_sensor(Sheep, self.sheep_cluster_radius, self.min_sheep_cluster_size))
        self.add_sensor("sheep_left", proximity_sensor(Sheep, vision_cone_angle, vision_range, vision_cone_pos, True))
        self.add_sensor("sheep_right", proximity_sensor(Sheep, vision_cone_angle, vision_range, -vision_cone_pos, True))
        self.add_sensor("left_wolf", proximity_sensor(Wolf, vision_cone_angle, vision_range, vision_cone_pos, True))
        self.add_sensor("right_wolf", proximity_sensor(Wolf, vision_cone_angle, vision_range, -vision_cone_pos, True))
        # custom built state sensor
        if USE_IS_AFRAID_SENSOR:
            self.add_sensor("is_afraid", state_sensor(Sheep))

        self.add_brain(10)

        self.solid = True
        self._min_speed = 10.0
        self._max_speed = 80.0
        self.radius = 10.0
        self.colour = ColourPalette[CT.BLUE]

    def wolf_seen(self) -> bool:
        return (
            self.sensors["left_wolf"].output() > 0
            or self.sensors["right_wolf"].output() > 0
        )

    # Toggles the state sensor based on determined cases
    def toggleAfraid(self): 
        if not USE_IS_AFRAID_SENSOR:
            return

        wolf_seen = self.wolf_seen()
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
        
    # Added state sensor toggling and time tracking for fitness, integrates with main cycle of code
    def control(self):
        super().control()
        self.toggleAfraid()
        self.timer += 1
        self.age += 1
        if self.wolf_seen():
            self.wolf_seen_steps += 1
        if USE_IS_AFRAID_SENSOR and self.sensors["is_afraid"].output() > 0:
            self.afraid_steps += 1
        if self.in_cluster():
            self.clustered_steps += 1
        for k in self.controls.keys():
            self.controls[k] = self.controls[k] + 0.5

    # Finds cluster size of alive sheep
    def cluster_size(self) -> int:
        if self.world is None:
            return 1
        sheep = [
            agent for agent in self.world._agents
            if isinstance(agent, Sheep) and not getattr(agent, "dead", False)
        ]
        return connected_component_size(self, sheep, self.sheep_cluster_radius, self.world)

    # Sheep is only in cluster if the group its in is bigger than the number defined as cluster size
    def in_cluster(self) -> bool:
        return self.cluster_size() >= self.min_sheep_cluster_size

    def protected_by_cluster(self) -> bool:
        if not USE_CLUSTER_PROTECTION:
            return False
        return self.in_cluster()
    
    # Collision behaviour, only for grass eating
    def on_collision(self, obj):
        if isinstance(obj, Grass):
            self.grass_found += 1
            self.weighted_grass += 1.0 / (1.0 + self.timer / self.sheep_recency_timesteps)
            self.timer = 0
            obj.eaten()
    
    # Behaviour if eaten, function triggered by wolf on collision
    def eaten(self):
        self.times_eaten += 1
        # From Agent class, allows it to no longer appear
        self.dead = True
    
    def get_fitness(self, ):
        if self.grass_found == 0:
            return self.sheep_fitness_floor

        # Variant fitness weighting depending on test config
        forage_reward = FITNESS_CONFIG.grass_reward * self.grass_found
        speed_reward = FITNESS_CONFIG.quick_grass_reward * self.weighted_grass
        survival_reward = FITNESS_CONFIG.survival_reward if self.times_eaten == 0 else 0.0
        death_multiplier = FITNESS_CONFIG.death_multiplier if self.times_eaten > 0 else 1.0
        return max(
            self.sheep_fitness_floor,
            (forage_reward + speed_reward + survival_reward) * death_multiplier
        )
    
    def reset(self):
        self.grass_found= 0
        self.times_eaten = 0
        self.weighted_grass = 0
        self.clustered_steps = 0
        self.wolf_seen_steps = 0
        self.afraid_steps = 0
        self.timer = 0
        self.age = 0


# Wolf agent class
class Wolf(EvolvableFFNAgent, Evolver):
    def __init__(self):
        EvolvableFFNAgent.__init__(self)
        Evolver.__init__(self)

        # Initialisation of tracked variables and range
        self.prey_eaten = 0
        self.age = 0
        vision_range = 300.0
        vision_cone_angle = 0.875*np.pi
        vision_cone_pos = 0.3403*np.pi


        # Init of fitness reward baseline
        self.wolf_kill_reward = 8.0
        
        # Initalisation of sensors
        self.add_sensor("angle", nearest_angle_sensor(Sheep, vision_range))
        self.add_sensor("left", proximity_sensor(Sheep, vision_cone_angle, vision_range, vision_cone_pos, True))
        self.add_sensor("right", proximity_sensor(Sheep, vision_cone_angle, vision_range, -vision_cone_pos, True))
        
        self.add_brain(6)
        self.solid = True
        self.colour = ColourPalette[CT.RED]
        self._min_speed = 10.0
        self._max_speed = 110.0
        self.radius = 15.0
    
    # Addition of age for time element for fitness
    def control(self):
        super().control()
        self.age += 1
        for k in self.controls.keys():
            self.controls[k] = 0.5 * (self.controls[k] + 1.0)
    
    # On collision, checks if sheep agent is clustered before it can eat them
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
        self.age = 0
        

class HerdSimulation(Simulation):
    def __init__(
        self,
        scenario: HerdTestScenario | str = DEFAULT_HERD_TEST,
        fitness: HerdFitnessConfig | str = DEFAULT_FITNESS,
        random_seed: int | None = DEFAULT_RANDOM_SEED,
    ):
        self.scenario = set_herd_test_scenario(scenario)
        self.fitness_config = set_herd_fitness_config(fitness)
        self.scenario_name = self.scenario.scenario_name
        self.result_slug = f"{self.scenario_name}__{self.fitness_config.slug}"
        self.display_name = (
            f"Herd - {self.scenario.key.replace('_', ' ').title()} "
            f"({self.fitness_config.label})"
        )
        self._assessment_metrics = []

        super().__init__(self.display_name, random_seed=random_seed)
        
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

        self.csv_path = os.path.join(self.results_dir, f"{self.result_slug}.csv")

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "generation",
                "scenario",
                "test",
                "fitness",
                "random_seed",
                "avg_sheep_fitness",
                "avg_wolf_fitness",
                "survival_rate",
                "sheep_alive",
                "sheep_deaths",
                "grass_total",
                "grass_per_sheep",
                "protected_time_fraction",
                "afraid_time_fraction",
                "wolf_seen_time_fraction",
                "protected_alive_fraction_end",
                "mean_alive_cluster_size_end",
                "largest_alive_cluster_size_end",
                "wolf_kills",
                "wolf_kills_per_wolf",
                "objective_survival_foraging_score",
                "objective_cluster_survival_score",
            ])

    def begin_generation(self) -> None:
        self._assessment_metrics = []
        super().begin_generation()

    def log_begin_assessment(self):
        self.log.info(
            f"Scenario: {self.scenario_name} | "
            f"Fitness: {self.fitness_config.key} | "
            f"Generation: {self._generation + 1}/{self.generations} | "
            f"Assessment: {self._assessment + 1}/{self.assessments} | "
            f"Seed: {self.random_seed}"
        )

    def end_assessment(self) -> None:
        self._assessment_metrics.append(self.assessment_metrics())
        super().end_assessment()

    def assessment_metrics(self) -> dict[str, float]:
        sheep = list(self.contents["sheep"].team)
        wolves = list(self.contents["wolf"].team)
        sheep_count = len(sheep)
        wolf_count = len(wolves)
        alive_sheep = [agent for agent in sheep if not getattr(agent, "dead", False)]

        sheep_deaths = sum(1 for agent in sheep if agent.times_eaten > 0)
        sheep_alive = sheep_count - sheep_deaths
        survival_rate = sheep_alive / sheep_count if sheep_count else 0.0

        grass_total = sum(agent.grass_found for agent in sheep)
        grass_per_sheep = grass_total / sheep_count if sheep_count else 0.0

        total_sheep_steps = self.timesteps * sheep_count
        protected_time_fraction = (
            sum(agent.clustered_steps for agent in sheep) / total_sheep_steps
            if total_sheep_steps
            else 0.0
        )
        afraid_time_fraction = (
            sum(agent.afraid_steps for agent in sheep) / total_sheep_steps
            if total_sheep_steps
            else 0.0
        )
        wolf_seen_time_fraction = (
            sum(agent.wolf_seen_steps for agent in sheep) / total_sheep_steps
            if total_sheep_steps
            else 0.0
        )

        cluster_sizes = [agent.cluster_size() for agent in alive_sheep]
        protected_alive = [
            size >= alive_sheep[0].min_sheep_cluster_size
            for size in cluster_sizes
        ] if alive_sheep else []
        protected_alive_fraction_end = (
            sum(protected_alive) / len(protected_alive)
            if protected_alive
            else 0.0
        )
        mean_alive_cluster_size_end = (
            sum(cluster_sizes) / len(cluster_sizes)
            if cluster_sizes
            else 0.0
        )
        largest_alive_cluster_size_end = max(cluster_sizes) if cluster_sizes else 0.0

        wolf_kills = sum(wolf.prey_eaten for wolf in wolves)
        wolf_kills_per_wolf = wolf_kills / wolf_count if wolf_count else 0.0

        return {
            "survival_rate": survival_rate,
            "sheep_alive": sheep_alive,
            "sheep_deaths": sheep_deaths,
            "grass_total": grass_total,
            "grass_per_sheep": grass_per_sheep,
            "protected_time_fraction": protected_time_fraction,
            "afraid_time_fraction": afraid_time_fraction,
            "wolf_seen_time_fraction": wolf_seen_time_fraction,
            "protected_alive_fraction_end": protected_alive_fraction_end,
            "mean_alive_cluster_size_end": mean_alive_cluster_size_end,
            "largest_alive_cluster_size_end": largest_alive_cluster_size_end,
            "wolf_kills": wolf_kills,
            "wolf_kills_per_wolf": wolf_kills_per_wolf,
            "objective_survival_foraging_score": survival_rate * grass_per_sheep,
            "objective_cluster_survival_score": survival_rate * protected_time_fraction,
        }

    def generation_metrics(self) -> dict[str, float]:
        if not self._assessment_metrics:
            return {}

        keys = self._assessment_metrics[0].keys()
        return {
            key: sum(metrics[key] for metrics in self._assessment_metrics) / len(self._assessment_metrics)
            for key in keys
        }

    def log_end_generation(self):
        sheep_averages = self.contents["sheep"].average_member_fitness()
        sheep_average = sum(sheep_averages) / len(sheep_averages)
        self.log.info(f"Average sheep Fitness: {sheep_average:.3f} Count: {self._generation}")

        wolf_averages = self.contents["wolf"].average_member_fitness()
        wolf_average = sum(wolf_averages) / len(wolf_averages)
        self.log.info(f"Average wolf Fitness: {wolf_average:.3f} Count: {self._generation}")
        metrics = self.generation_metrics()

        self.log.info(
            f"Scenario: {self.scenario_name} | "
            f"Fitness: {self.fitness_config.key} | "
            f"Generation: {self._generation + 1}/{self.generations} | "
            f"Average sheep fitness: {sheep_average:.3f} | "
            f"Average wolf fitness: {wolf_average:.3f} | "
            f"Survival: {metrics.get('survival_rate', 0.0):.3f} | "
            f"Protected-time: {metrics.get('protected_time_fraction', 0.0):.3f} | "
            f"Grass/sheep: {metrics.get('grass_per_sheep', 0.0):.3f}"
        )

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self._generation,
                self.scenario_name,
                self.scenario.key,
                self.fitness_config.key,
                self.random_seed,
                sheep_average,
                wolf_average,
                metrics.get("survival_rate", 0.0),
                metrics.get("sheep_alive", 0.0),
                metrics.get("sheep_deaths", 0.0),
                metrics.get("grass_total", 0.0),
                metrics.get("grass_per_sheep", 0.0),
                metrics.get("protected_time_fraction", 0.0),
                metrics.get("afraid_time_fraction", 0.0),
                metrics.get("wolf_seen_time_fraction", 0.0),
                metrics.get("protected_alive_fraction_end", 0.0),
                metrics.get("mean_alive_cluster_size_end", 0.0),
                metrics.get("largest_alive_cluster_size_end", 0.0),
                metrics.get("wolf_kills", 0.0),
                metrics.get("wolf_kills_per_wolf", 0.0),
                metrics.get("objective_survival_foraging_score", 0.0),
                metrics.get("objective_cluster_survival_score", 0.0),
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
    fitness: str,
    runs: int | None,
    generations: int | None,
    assessments: int | None,
    timesteps: int | None,
    random_seed: int | None,
) -> str:
    sim = HerdSimulation(test, fitness, random_seed=random_seed)
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
    sim.log.info(f"Fitness config: {sim.fitness_config.key} ({sim.fitness_config.label})")
    sim.log.info(f"Purpose: {scenario.purpose}")
    sim.log.info(f"Fitness purpose: {sim.fitness_config.purpose}")
    sim.run_simulation(render=False, parallel=False)
    return sim.result_slug


def run_herd_tests_parallel(
    selected_tests: list[HerdExperimentCase | HerdTestScenario | str | tuple[str, str]] | None = None,
    *,
    max_workers: int | None = None,
    runs: int | None = None,
    generations: int | None = None,
    assessments: int | None = None,
    timesteps: int | None = None,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
) -> None:
    tests_to_run = herd_experiment_cases(selected_tests)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_herd_test_no_render,
                case.scenario_key,
                case.fitness_key,
                runs,
                generations,
                assessments,
                timesteps,
                random_seed,
            ): case
            for case in tests_to_run
        }

        for future in as_completed(futures):
            case = futures[future]
            result_slug = future.result()
            print(f"Finished herd test: {case.scenario_key} / {case.fitness_key} ({result_slug})")


def run_herd_tests(
    selected_tests: list[HerdExperimentCase | HerdTestScenario | str | tuple[str, str]] | None = None,
    *,
    render: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
    runs: int | None = None,
    generations: int | None = None,
    assessments: int | None = None,
    timesteps: int | None = None,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
) -> None:
    tests_to_run = herd_experiment_cases(selected_tests)

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
            random_seed=random_seed,
        )
        return

    for case in tests_to_run:
        sim = HerdSimulation(case.scenario_key, case.fitness_key, random_seed=random_seed)
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
        sim.log.info(f"Fitness config: {sim.fitness_config.key} ({sim.fitness_config.label})")
        sim.log.info(f"Purpose: {scenario.purpose}")
        sim.log.info(f"Fitness purpose: {sim.fitness_config.purpose}")
        sim.run_simulation(render=render, parallel=parallel)
