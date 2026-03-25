"""
Metaheuristic Baseline: Genetic Algorithm for Transit Route Network Design.

Route-Set Genetic Algorithm (GA) for TRNDP:
- Chromosome = set of transit routes (sequences of stops on road graph)
- Fitness = transit performance via simulation
- Operators = route-set operators (route exchange crossover, path regeneration mutation)
"""

import os
import json
import shutil
import numpy as np
import random
import torch
from datetime import datetime
import multiprocessing as mp
mp_ctx = mp.get_context('spawn')

from rl.env_utils import (
    plot_network_and_demand,
    plot_network_demand_and_path,
    initialize_route,
    aggregate_results,
    write_results_summary,
    ensure_eval_step_update_dir,
    make_seed_output_dir,
    save_routes_json,
)
from baselines.utils import (
    set_global_seeds,
    create_main_save_dir,
    simulate_baseline_routes,
    create_initial_network_plot,
    create_path_visualization,
    create_fancy_animations,
    execute_runs,
    print_results,
)

def _ga_evaluate_individual(args):
    """
    Evaluate fitness of a single individual in a worker process.

    This function is standalone (no class dependencies) so it can be
    pickled and sent to worker processes via multiprocessing.Pool.

    Args:
        args: Tuple of (individual, config_dict)
            - individual: List of routes, each route is list of node IDs
            - config_dict: Configuration dictionary for TransitEnv

    Returns:
        Scalar fitness value
    """
    individual, config_dict = args

    # Import here to avoid circular imports in worker processes
    from rl.env import TransitEnv

    # Create fresh environment
    env = TransitEnv(config_dict)

    # Set environment state to this individual
    env.all_routes = [[str(n) for n in route] for route in individual]
    env.current_route = []
    env.current_route_index = len(individual)
    env.is_baseline = True

    # Build world and run simulation
    env.world = env.build_world(config_dict.get("network"))
    env._apply_action()
    sim_result = env._step_until(env.horizon, print_metrics=False)

    # Compute reward (same function as RL methods)
    sim_result['route_completed'] = True
    sim_result['route_forced_end'] = False
    fitness = env.compute_reward(sim_result, is_route_end=True, is_forced_end=False)

    return fitness

class GeneticAlgorithm:
    """
    Route-Set Genetic Algorithm for Transit Route Network Design (TRNDP).
    - Chromosome = set of transit routes (sequences of stops on road graph)
    - Fitness = transit performance via simulation (coverage/service/waiting/operator costs)
    - Operators = route-set operators (route exchange crossover, path regeneration mutation)

    Uses same compute_reward() as PPO/MCTS for fair comparison.
    Budget matching: total simulations ≈ population × generations.

    Notes:
    - All routes satisfy: MIN_ROUTE_LENGTH <= len(route) <= MAX_ROUTE_LENGTH
    - All routes are valid paths on the road graph (adjacent nodes, no cycles)
    - Fitness evaluation uses identical reward function as RL methods
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, env, config, num_runs, base_seed):
        self.env = env
        self.config = config
        self.world = env.build_world(config.get("network"))
        self.num_runs = num_runs
        self.base_seed = base_seed

        now = datetime.now()
        self.main_save_dir = os.path.join(
            config.get("save_dir"),
            f"genetic_{now.strftime('%b')}_{now.strftime('%d')}_{now.strftime('%H')}_{now.strftime('%M')}_{now.strftime('%S')}"
        )
        os.makedirs(self.main_save_dir, exist_ok=True)

        # GA hyperparameters (from config, defaults defined in config.py)
        self.population_size = config["ga_population"]
        self.generations = config["ga_generations"]
        self.mutation_rate = config["ga_mutation_rate"]
        self.crossover_rate = config["ga_crossover_rate"]
        self.tournament_size = config["ga_tournament_size"]
        self.elitism_count = config["ga_elitism"]
        self.num_workers = config["ga_num_workers"]

        # Route constraints (from env)
        self.min_route_len = getattr(env, 'MIN_ROUTE_LENGTH', 2)
        self.max_route_len = env.MAX_ROUTE_LENGTH
        self.num_routes = env.NUM_ROUTES

        # Fitness cache for memoization
        self._fitness_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def run(self):
        """
        Execute GA baseline. Evolution runs once, then full evaluation
        on the final best solution.
        """
        state, _ = self.env.reset(seed=self.base_seed)
        best_solution = self.construct_path(state)
        self._evaluate_final_best(best_solution)
        return None, None

    # ─────────────────────────────────────────────────────────────────────────
    # Main Evolution Loop
    # ─────────────────────────────────────────────────────────────────────────

    def construct_path(self, state):
        """
        Run the genetic algorithm to find optimal transit network.

        Returns:
            List of routes (best solution found)
        """
        print(f"\n=== Route-Set Genetic Algorithm (TRNDP) ===")
        print(f"Population: {self.population_size}, Generations: {self.generations}")
        print(f"Mutation: {self.mutation_rate}, Crossover: {self.crossover_rate}, Elitism: {self.elitism_count}")
        print(f"Route constraints: {self.min_route_len} <= len <= {self.max_route_len}")
        print(f"Parallel workers: {self.num_workers}")

        # Reset state
        self._fitness_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

        # Initialize and evolve
        print(f"\nInitializing population...")
        population = self._initialize_population()
        self._save_initial_population(population)
        print(f"Population initialized. Starting evolution...\n")
        best_solution, best_fitness = self._evolve(population)

        # Report results
        print(f"\nGA complete. Best fitness: {best_fitness:.2f}")
        print(f"Simulations: {self._cache_misses}, Cache hits: {self._cache_hits}")

        self.env.all_routes = best_solution
        return best_solution

    def _evolve(self, population):
        """
        Run the evolution loop for all generations.

        Returns:
            (best_solution, best_fitness) tuple
        """
        best_solution = None
        best_fitness = -np.inf

        for gen in range(self.generations):
            # Evaluate all individuals (parallel or sequential)
            fitness_scores = self._evaluate_population(population)

            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_solution = self._copy_individual(population[gen_best_idx])
                self._save_best_solution(best_solution, best_fitness, gen + 1)

            # Log progress
            print(f"Gen {gen + 1}/{self.generations}: "
                  f"Best={best_fitness:.2f}, Avg={np.mean(fitness_scores):.2f}, "
                  f"Cache={self._cache_hits}/{self._cache_hits + self._cache_misses}")

            # Create next generation
            population = self._create_next_generation(population, fitness_scores)

        return best_solution, best_fitness

    def _create_next_generation(self, population, fitness_scores):
        """
        Create the next generation via selection, crossover, mutation.

        Returns:
            New population (list of individuals)
        """
        next_pop = []

        # Elitism: preserve top individuals
        elite_indices = np.argsort(fitness_scores)[::-1][:self.elitism_count]
        for idx in elite_indices:
            next_pop.append(self._copy_individual(population[idx]))

        # Fill rest with offspring
        while len(next_pop) < self.population_size:
            # Select parents
            parent1 = self._tournament_select(population, fitness_scores)
            parent2 = self._tournament_select(population, fitness_scores)

            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1 = self._copy_individual(parent1)
                child2 = self._copy_individual(parent2)

            # Mutate
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2)

            # Repair and add (repair guarantees feasibility)
            next_pop.append(self._repair(child1))
            if len(next_pop) < self.population_size:
                next_pop.append(self._repair(child2))

        return next_pop

    # ─────────────────────────────────────────────────────────────────────────
    # Population Initialization
    # ─────────────────────────────────────────────────────────────────────────

    def _initialize_population(self):
        """
        Create initial population with diverse solutions.

        Strategy mix:
        - 1 warm-start individual from real-world routes (if available)
        - 1/3 demand-guided construction (greedy OD coverage)
        - 2/3 random feasible routes (exploration)

        Returns:
            Population where every individual is guaranteed feasible.
        """
        population = []

        # Try to add warm-start from real-world routes
        print("  Loading warm-start from real-world routes...")
        warm_start = self._load_warm_start()
        if warm_start is not None:
            population.append(warm_start)
            print("  [1/{0}] Added warm-start individual".format(self.population_size))
        else:
            print("  No warm-start available, using generated routes only")

        # Fill with generated individuals
        while len(population) < self.population_size:
            use_demand_guided = (len(population) % 3 == 0)
            strategy = "demand-guided" if use_demand_guided else "random"
            individual = self._build_individual(demand_guided=use_demand_guided)
            population.append(individual)
            print(f"  [{len(population)}/{self.population_size}] Built individual ({strategy})")

        return population

    def _build_individual(self, demand_guided=False):
        """
        Build a complete individual (set of routes).

        Args:
            demand_guided: If True, use greedy demand coverage; else random.

        Returns:
            Individual guaranteed to satisfy all constraints.
        """
        self.env.all_routes = []
        self.env.current_route = []
        self.env.current_route_index = 0

        individual = []
        for _ in range(self.num_routes):
            route = self._build_route(demand_guided=demand_guided)
            individual.append(route)

        return self._repair(individual)

    def _load_warm_start(self):
        """
        Load real-world routes as warm-start individual.

        Returns:
            Feasible individual, or None if routes unavailable.
        """
        routes_file = self.env.network_dir / f"{self.config.get('network')}_existing_routes.json"
        if not routes_file.exists():
            return None

        try:
            with open(routes_file, "r") as f:
                all_routes = json.load(f)

            # Check if we need to filter by transit center
            route_init = self.config["route_init"]
            transit_center = str(self.env.transit_center_node) if route_init == "transit_center" else None

            # Score routes by demand coverage, filter by start node if needed
            scored = []
            for route in all_routes:
                nodes = route.get('nodes', [])
                if not nodes:
                    continue
                # Filter by transit center start if route_init is "transit_center"
                if transit_center and str(nodes[0]) != transit_center:
                    continue
                demand = sum(self._get_node_demand(n) for n in nodes)
                score = len(nodes) * (demand / len(nodes))
                scored.append((score, nodes))

            scored.sort(key=lambda x: x[0], reverse=True)
            selected = [nodes for _, nodes in scored[:self.num_routes]]

            # Convert to string node IDs and truncate to max length
            individual = []
            for nodes in selected:
                route = [str(n) for n in nodes[:self.max_route_len]]
                individual.append(route)

            # Pad with random routes if needed
            while len(individual) < self.num_routes:
                individual.append(self._build_route(demand_guided=False))

            return self._repair(individual)

        except Exception as e:
            print(f"  Warning: Could not load warm-start: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Route Building Primitives
    # ─────────────────────────────────────────────────────────────────────────

    def _build_route(self, demand_guided=False):
        """
        Build a single route via incremental extension.

        Args:
            demand_guided: If True, extend toward high-demand nodes.

        Returns:
            Route (list of node IDs). May be shorter than min_route_len;
            caller should use _repair() to guarantee feasibility.
        """
        from rl.env_utils import initialize_route
        route = initialize_route(self.env)

        while len(route) < self.max_route_len:
            neighbors = self._get_valid_neighbors(route[-1], exclude=set(route))
            if not neighbors:
                break

            if demand_guided:
                next_node = self._select_by_demand(neighbors, route)
            else:
                next_node = random.choice(neighbors)

            route.append(next_node)

        return route

    def _get_valid_neighbors(self, node, exclude):
        """
        Get adjacent nodes that aren't in the exclude set.

        Args:
            node: Current node ID
            exclude: Set of node IDs to exclude (e.g., already in route)

        Returns:
            List of valid neighbor node IDs
        """
        return [n for n in self.env.adj[node] if n not in exclude]

    def _select_by_demand(self, candidates, current_route):
        """
        Select the candidate with highest demand interaction with current route.

        Args:
            candidates: List of candidate node IDs
            current_route: Current route being built

        Returns:
            Best candidate node ID
        """
        route_indices = np.array([self.env.node_to_idx[n] for n in current_route])

        best_node = candidates[0]
        best_score = -np.inf

        for node in candidates:
            idx = self.env.node_to_idx[node]
            d_out = self.env.od_matrix[idx, route_indices].sum()
            d_in = self.env.od_matrix[route_indices, idx].sum()
            score = d_out + d_in

            if score > best_score:
                best_score = score
                best_node = node

        return best_node

    def _get_node_demand(self, node_id):
        """Get total demand (in + out) for a node."""
        idx = self.env.node_to_idx.get(str(node_id))
        if idx is not None:
            return self.env.demand_out[idx] + self.env.demand_in[idx]
        return 0

    # ─────────────────────────────────────────────────────────────────────────
    # Genetic Operators
    # ─────────────────────────────────────────────────────────────────────────

    def _tournament_select(self, population, fitness_scores):
        """
        Select an individual via tournament selection.

        Returns:
            Copy of selected individual
        """
        k = min(self.tournament_size, len(population))
        indices = random.sample(range(len(population)), k)
        winner_idx = max(indices, key=lambda i: fitness_scores[i])
        return self._copy_individual(population[winner_idx])

    def _crossover(self, parent1, parent2):
        """
        Route exchange crossover: for each route position, randomly
        choose which parent to inherit from.

        Returns:
            (child1, child2) tuple
        """
        child1, child2 = [], []

        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1.append(parent1[i].copy())
                child2.append(parent2[i].copy())
            else:
                child1.append(parent2[i].copy())
                child2.append(parent1[i].copy())

        return child1, child2

    def _mutate(self, individual):
        """
        Path regeneration mutation: pick a random route, cut at a
        random point, and regrow the tail via random walk.

        Returns:
            Mutated individual (may need repair)
        """
        route_idx = random.randint(0, len(individual) - 1)
        route = individual[route_idx]

        # Need at least 3 nodes to have a meaningful cut point
        if len(route) < 3:
            return individual

        # Cut and regrow
        cut_point = random.randint(1, len(route) - 2)
        new_route = route[:cut_point + 1]

        while len(new_route) < self.max_route_len:
            neighbors = self._get_valid_neighbors(new_route[-1], exclude=set(new_route))
            if not neighbors:
                break
            new_route.append(random.choice(neighbors))

        individual[route_idx] = new_route
        return individual

    def _repair(self, individual):
        """
        Ensure all routes satisfy feasibility constraints.

        Guarantees on return:
        - Every route has length in [min_route_len, max_route_len]
        - Every route is a valid path (adjacent nodes, no duplicates)

        Raises:
            RuntimeError if unable to build valid route after max attempts
            (indicates network topology issue)
        """
        max_attempts = 10

        for route_idx, route in enumerate(individual):
            # Truncate if too long
            if len(route) > self.max_route_len:
                route = route[:self.max_route_len]

            # Extend or rebuild if too short
            attempts = 0
            while len(route) < self.min_route_len and attempts < max_attempts:
                neighbors = self._get_valid_neighbors(route[-1], exclude=set(route))

                if neighbors:
                    route.append(random.choice(neighbors))
                else:
                    # Dead end - rebuild entire route
                    route = self._build_route(demand_guided=False)
                    attempts += 1

            # Final check
            if len(route) < self.min_route_len:
                raise RuntimeError(
                    f"Cannot build valid route after {max_attempts} attempts. "
                    f"Length {len(route)} < min {self.min_route_len}. "
                    f"Check network connectivity."
                )

            individual[route_idx] = route

        return individual

    # ─────────────────────────────────────────────────────────────────────────
    # Fitness Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_population(self, population):
        """
        Evaluate fitness for all individuals in the population.

        Uses parallel evaluation when num_workers > 1, otherwise sequential.
        Memoization is handled properly: cached individuals are not re-evaluated.

        Args:
            population: List of individuals

        Returns:
            List of fitness scores (same order as population)
        """
        if self.num_workers <= 1:
            # Sequential evaluation
            return [self._evaluate_fitness(ind) for ind in population]

        # Parallel evaluation with memoization and batch deduplication
        fitness_scores = [None] * len(population)
        key_to_indices = {}  # key -> list of population indices with this key

        # Group by key, checking cache first
        for i, individual in enumerate(population):
            key = self._individual_to_key(individual)
            if key in self._fitness_cache:
                fitness_scores[i] = self._fitness_cache[key]
                self._cache_hits += 1
            elif key in key_to_indices:
                # Duplicate within this batch - will reuse result
                key_to_indices[key].append(i)
            else:
                # New key - needs evaluation
                key_to_indices[key] = [i]

        # Collect unique individuals to evaluate
        unique_evals = []  # (key, individual) for unique uncached individuals
        for i, individual in enumerate(population):
            key = self._individual_to_key(individual)
            if key in key_to_indices and key_to_indices[key][0] == i:
                # This is the first occurrence of this key
                unique_evals.append((key, individual))

        # If all cached or duplicates, return early
        if not unique_evals:
            return fitness_scores

        # Prepare args for parallel evaluation (only unique individuals)
        # Use env.config (not self.config) to get the updated seed from env.reset()
        config_dict = dict(self.env.config)
        worker_args = [(ind, config_dict) for _, ind in unique_evals]

        # Run parallel evaluation using spawn context (same as PPO)
        with mp_ctx.Pool(processes=self.num_workers) as pool:
            results = pool.map(_ga_evaluate_individual, worker_args)

        # Update cache and assign fitness to all indices sharing each key
        for (key, _), fitness in zip(unique_evals, results):
            self._fitness_cache[key] = fitness
            self._cache_misses += 1
            for idx in key_to_indices[key]:
                fitness_scores[idx] = fitness

        return fitness_scores

    def _evaluate_fitness(self, individual):
        """
        Evaluate fitness via traffic simulation.

        Uses memoization to avoid redundant simulations.
        Uses same reward function as PPO/MCTS for fair comparison.

        Returns:
            Scalar fitness value
        """
        key = self._individual_to_key(individual)

        if key in self._fitness_cache:
            self._cache_hits += 1
            return self._fitness_cache[key]

        self._cache_misses += 1

        # Setup env state
        self.env.all_routes = [[str(n) for n in route] for route in individual]
        self.env.current_route = []
        self.env.current_route_index = len(individual)
        self.env.is_baseline = True

        # Run simulation
        self.env.world = self.env.build_world(self.config.get("network"))
        self.env._apply_action()
        sim_result = self.env._step_until(self.env.horizon, print_metrics=False)

        # Compute reward (same as RL)
        sim_result['route_completed'] = True
        sim_result['route_forced_end'] = False
        fitness = self.env.compute_reward(sim_result, is_route_end=True, is_forced_end=False)

        self._fitness_cache[key] = fitness
        return fitness

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _save_initial_population(self, population):
        """
        Save initial population to JSON files for inspection.

        Creates: {main_save_dir}/initial_population/individual_01.json, etc.
        """
        pop_dir = os.path.join(self.main_save_dir, "initial_population")
        os.makedirs(pop_dir, exist_ok=True)

        for i, individual in enumerate(population):
            filename = f"individual_{i+1:02d}.json"
            filepath = os.path.join(pop_dir, filename)

            # Convert to readable format
            data = {
                "individual_id": i + 1,
                "num_routes": len(individual),
                "routes": {
                    f"route_{j+1}": {
                        "nodes": [str(n) for n in route],
                        "length": len(route)
                    }
                    for j, route in enumerate(individual)
                }
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        print(f"  Saved {len(population)} individuals to {pop_dir}")

    def _save_best_solution(self, solution, fitness, generation):
        """
        Save current best solution metadata and routes to disk (lightweight).
        Full evaluation runs once at the end via _evaluate_final_best.

        Creates: {main_save_dir}/current_best/
                   - best_metadata.json
                   - routes.json
        """
        best_dir = os.path.join(self.main_save_dir, "current_best")
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        os.makedirs(best_dir)

        # Save metadata
        with open(os.path.join(best_dir, "best_metadata.json"), "w") as f:
            json.dump({
                "generation_found": generation,
                "fitness": fitness,
                "num_routes": len(solution)
            }, f, indent=2)

        # Save routes
        routes = [[str(n) for n in route] for route in solution]
        with open(os.path.join(best_dir, "routes.json"), "w") as f:
            json.dump({
                f"route_{i+1}": route for i, route in enumerate(routes)
            }, f, indent=2)

    def _evaluate_final_best(self, solution):
        """
        Run full evaluation on final best solution with all seeds.
        Called once at the end of evolution.

        Creates: {main_save_dir}/current_best/
                   - eval_results_summary.json
                   - seed_X/designed_routes.json, etc.
        """
        best_dir = os.path.join(self.main_save_dir, "current_best")
        routes = [[str(n) for n in route] for route in solution]
        eval_offset = self.config["eval_seed_offset"]
        results = []

        print(f"\nRunning final evaluation ({self.num_runs} seeds)...")

        for run in range(self.num_runs):
            current_seed = self.base_seed + (run * eval_offset)
            set_global_seeds(current_seed)
            print(f"  Eval run {run+1}/{self.num_runs} (seed={current_seed})")

            seed_dir, img_dir = make_seed_output_dir(best_dir, current_seed)

            self.env.reset(seed=current_seed)
            result = simulate_baseline_routes(
                self.env, self.config, routes, img_dir, seed_dir
            )
            results.append(result)

        aggregated = aggregate_results(results)
        write_results_summary(aggregated, self.num_runs, best_dir, 'eval_results_summary.json')
        print_results(results, aggregated)

    def _individual_to_key(self, individual):
        """Convert individual to hashable key for memoization."""
        return tuple(tuple(str(n) for n in route) for route in individual)

    def _copy_individual(self, individual):
        """Deep copy an individual."""
        return [route.copy() for route in individual]
