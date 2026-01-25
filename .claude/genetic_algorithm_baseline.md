# Genetic Algorithm Baseline for TRNDP

## Paper Writeup

> **Genetic Algorithm**: We implement a transit-network genetic algorithm baseline following prior TRNDP/TNDP metaheuristics that evolve a *route set* (multiple stop sequences) under feasibility constraints using selection, crossover, mutation, and elitism (Fan & Machemehl 2006, Nayeem et al. 2014). Each individual encodes a fixed-size network Π={r₁,...,rₖ} of K bus routes, where each route rₖ is a simple path on the road graph satisfying L_min ≤ |rₖ| ≤ L_max and (in our Bloomington setting) is initialized at the designated transit center to match the deployed system configuration. To evaluate fitness, we deterministically assign each route a service frequency using the same max-load rule used by all methods, then run a full UXsim transit simulation to obtain passenger- and operator-side metrics (coverage, service rate, waiting and in-vehicle times, overlap, fleet size, and utilization). Fitness is computed by the same scalar objective R (Eq. reward) used by PPO and MCTS, ensuring a fair comparison that differs only in the search strategy. We run the GA for G generations with population size P (approximately P×G full simulator evaluations) and report the best-found design under the same evaluation protocol (multiple random seeds) as other baselines.

---

## Design Plan (TRNDP-focused)

### 2.1 Core positioning: why this is a TRNDP GA (not generic)

This GA is not a "bitstring GA." It is a **route-set GA**:
- **Chromosome** = a *set of transit routes* (multiple sequences of stops on a graph).
- **Fitness** = measured through *transit performance evaluation* (coverage/service/waiting/transfers/operator costs).

This matches the TRNDP literature pattern in Fan & Machemehl (GA + network analysis / assignment) and Nayeem et al. (GA + elitism + transit objectives).

### 2.2 Solution representation (explicitly TRNDP-like)

- **Individual**: Π={r₁,...,rₖ} with **fixed route count** K=`num_routes` (default 16).
- **Route**: ordered stop sequence rₖ=(v₁,...,vₗ) forming a **simple path / self-avoiding walk** on the road graph:
  - adjacency constraint: (vᵢ,vᵢ₊₁) ∈ E
  - ℓ ∈ [L_min, L_max] (config: 2–14)
- **Bidirectional service**: evaluated by the simulator's routing / transit graph construction.

**TRNDP specificity**: this is the standard TNDP/TRNDP "route set" encoding used in GA-based transit design papers (route sequences under constraints).

### 2.3 Fitness evaluation (network analysis + simulation = TRNDP-standard)

For each individual Π:
1. **Deterministically set frequencies** using environment rule (`max_load`), which depends on demand + overlaps (conceptually a "frequency-setting step," common in TRNDP formulations).
2. **Run full simulation** in UXsim for horizon T_sim=10,000 seconds.
3. Compute the scalar score using the **same system reward** as RL:
   ```
   fitness(Π) = env.compute_reward(sim_result, is_route_end=True, is_forced_end=False)
   ```

This is TRNDP-consistent because GA papers typically embed a **network evaluation / assignment procedure** inside the evolutionary loop; the simulator + demand allocation plays that role.

### 2.4 Initialization (TRNDP-inspired)

**Population seeding mix:**
- **(A) Demand-guided construction** (TRNDP-style): build routes by repeatedly extending toward nodes with large OD interaction with the partial route.
- **(B) Random feasible routes**: random walks with feasibility checks.
- **(C) "Redesign/warm-start" seed** (very TRNDP): inject *one* individual derived from the **real-world route set** (or top-K subset). This mirrors TRNDP "redesign existing network" settings discussed in Fan & Machemehl.

### 2.5 Genetic operators (route-set operators, TRNDP-style)

**Selection**
- Tournament selection (size k), standard and fine.

**Crossover (route exchange)**
- "Swap routes between parents" is a **route exchange crossover** at the *route level*. This is exactly the kind of encoding TRNDP GAs use (operate on route sets rather than arbitrary strings).

**Mutation (route re-growth / segment replacement)**
- "Cut point then regrow via random walk" is a **path regeneration** mutation (TRNDP-relevant because routes must remain walks on the graph).

**Elitism**
- Preserve top-N; directly aligned with Nayeem et al.'s GA-with-elitism framing.

### 2.6 Constraint handling & repair

TRNDP papers almost always require feasibility checks or repair because crossover/mutation can yield invalid route sets.

**repair() step after crossover/mutation:**
- **Length repair**: if |rₖ| < L_min, extend using nearest feasible neighbors until L_min or restart that route.
- **Connectivity**: ensure each adjacent pair is connected (construction already enforces).
- **No duplicates**: ensure simple path (enforced via "not in route_set").
- **Stop spacing**: enforced via simulation (`STOP_SPACING`).

### 2.7 Budget matching (fairness definition)

"GA simulations ≈ RL environment steps" is *potentially misleading*, because:
- GA fitness eval = **one full simulation** (expensive).
- PPO env steps are **mostly cheap** (partial metrics), with full simulations only at route completion.

A cleaner, TRNDP-relevant budget metric is:
- **simulation budget** = number of full UXsim runs.

For GA:
- sims ≈ `population_size × generations` (plus minor overhead from memoization savings)

**In the paper, state budget matching as:**
> "We match baselines by **number of full simulator evaluations** (or wall-clock), since that dominates cost."

### 2.8 Practical compute sanity check

Example:
- pop = 20, gen = 50 → **1000** sims.

If each sim costs 2–5 seconds:
- 1000 × 2s = 2000s = 33.33 min
- 1000 × 5s = 5000s = 83.33 min

So "~30–60 minutes" is plausible but slightly optimistic unless sims are closer to 2–3.5 seconds.

---

## Implementation Notes

### 3.1 Memoization (implemented)

Elitism + tournament selection can cause repeated individuals. Fitness is cached:
- key = `tuple(tuple(route) for route in individual)`
- value = fitness

This can cut simulator calls significantly.

### 3.2 Determinism / noise control

Within a GA run, the simulator seed is fixed via `world.random_seed=self.config["seed"]`, so selection pressure is meaningful.

### 3.3 Parallel evaluation (optional, not yet implemented)

Fitness evaluation is embarrassingly parallel. If runtime becomes limiting, parallelize `_evaluate_fitness()` across population members (careful: each worker needs its own env/world instance).

---

## Usage

```bash
# Basic run
python main.py --mode=baseline --baseline_type=genetic \
    --ga_population=20 --ga_generations=50 \
    --num_eval_runs=5 --alpha=0.3

# Quick smoke test
python main.py --mode=baseline --baseline_type=genetic \
    --ga_population=5 --ga_generations=3 \
    --num_eval_runs=1
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ga_population` | 20 | Population size |
| `--ga_generations` | 50 | Number of generations |
| `--ga_mutation_rate` | 0.2 | Mutation probability |
| `--ga_crossover_rate` | 0.8 | Crossover probability |
| `--ga_tournament_size` | 3 | Tournament selection size |
| `--ga_elitism` | 2 | Number of elite individuals preserved |

## References

- Fan, W., & Machemehl, R. B. (2006). Optimal transit route network design problem with variable transit demand: genetic algorithm approach. *Journal of Transportation Engineering*, 132(1), 40-51.
- Nayeem, M. A., Rahman, M. K., & Rahman, M. S. (2014). Transit network design by genetic algorithm with elitism. *Transportation Research Part C*, 46, 30-45.
