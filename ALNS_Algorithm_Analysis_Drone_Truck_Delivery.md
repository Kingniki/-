# ALNS Algorithm Analysis for Drone-Truck Collaborative Delivery

## Table of Contents
1. [ALNS Algorithm Framework Overview](#1-alns-algorithm-framework-overview)
2. [Destroy Operators](#2-destroy-operators)
3. [Repair Operators](#3-repair-operators)
4. [Acceptance Criterion](#4-acceptance-criterion)
5. [Weight Update Mechanism](#5-weight-update-mechanism)
6. [ALNS Implementations in Different Literature](#6-alns-implementations-in-different-literature)
7. [Algorithm Parameter Settings](#7-algorithm-parameter-settings)

---

## 1. ALNS Algorithm Framework Overview

### 1.1 Basic Principles

**ALNS (Adaptive Large Neighborhood Search)** is an extension of LNS that progressively improves initial solutions through statistical selection of multiple destroy and repair methods.

**Core Concepts**:
- Progressive improvement: Repeatedly destroying and repairing the current solution
- Adaptive selection: Statistically choosing operators based on search performance
- Diversity exploration: Destroy and repair methods contain randomness for solution space diversification

### 1.2 Algorithm Flow (Sacramento et al., 2019)

```
Input: Initial solution s
Output: Best solution s*

1. Initialization:
   - Generate initial solution s
   - Set initial temperature T_st
   - Initialize operator weights w_i (all weights equal)
   
2. Main loop (until time limit t_max):
   While t_elap < t_max:
      a. Select destroy operator d ∈ Ω⁻ based on roulette wheel
      b. Select repair operator r ∈ Ω⁺ based on roulette wheel
      c. Apply destroy operator: s_partial = d(s)
      d. Apply repair operator: s_new = r(s_partial)
      e. Apply acceptance criterion to decide whether to accept s_new
      f. Update operator weights
      g. Update temperature: T = T_st * (1 - t_elap/t_max)
      h. If no improvement for noImprovMax iterations, restore best solution
      
3. Return best solution s*
```

### 1.3 Destruction Degree Control

The number of customers to remove β is controlled by the following formula:

```
β = min(max(ψ, ρ·|C|), c_lim)
```

Where:
- ψ (psi): Ratio of customers to remove (typically 0.15)
- c_low: Lower bound on number of customers to remove (random 1-3)
- c_lim: Upper bound on number of customers to remove (typically 40)
- |C|: Total number of customers

---

## 2. Destroy Operators

### 2.1 Random Destroy

**Algorithm Description** (Sacramento et al., 2019):
```
Algorithm: Random Destroy
Input: Current solution s, removal count β
Output: Partial solution s_partial

1. Initialize removal counter removed = 0
2. While removed < β:
   a. Randomly select customer c to remove from solution
   b. If c is a launch/recovery point:
      - Also remove associated drone customer(s)
   c. removed += (number of customers removed)
3. Return s_partial
```

**Characteristics**:
- Completely random customer removal
- If a drone launch/recovery point is removed, associated drone customers are also removed
- May remove 1-2 more customers than β specifies (due to drone associations)

### 2.2 Cluster Destroy

**Algorithm Description** (Sacramento et al., 2019):
```
Algorithm: Cluster Destroy
Input: Current solution s, removal count β
Output: Partial solution s_partial

1. Randomly select seed customer c₁ as focal point
2. Remove c₁ from solution
3. removed = 1
4. While removed < β:
   a. Find 2 closest customers to c₁ in current partial solution
   b. Randomly select 1 of these 2 customers to remove
   c. removed += 1
   d. If removed customer is launch/recovery point, remove associated drone customers
5. Return s_partial
```

**Characteristics**:
- Removes customers in a concentrated zone
- Adds noise to avoid obtaining the same partial solution
- Suitable for destroying local solution structures

**Pseudocode Example**:
```python
# Algorithm 3 from Sacramento et al., 2019
def cluster_removal(s, beta):
    c1 = RandomCustomer(s)
    remove c1 from s
    removed = {c1}
    
    while len(removed) < beta:
        c = RandomCloseCustomer(c1, s)
        if c is Launch and/or Recovery Position:
            remove drone customer(s) associated with c from s
        remove c from s
        update removed
    
    return s
```

### 2.3 Two-Stage Destroy (Kitjacharoenchai et al., 2020)

For the 2EVRPD problem, Kitjacharoenchai et al. proposed three sequentially executed destroy operators:

1. **Drone Node Removal**
   - Remove ω·v₂·|C_drone| nodes served by drone only
   - If all nodes in a sub-route are removed, launch/recovery points become available for other drones

2. **Truck Node Removal**
   - Remove ω·v₃·|C_truck| nodes served by truck only
   - Excluding launch/recovery nodes

3. **Sub-drone Route Removal**
   - Remove ω·v_t·|R_sub| complete drone sub-routes
   - If all sub-routes are removed, degenerates to standard VRP

---

## 3. Repair Operators

### 3.1 Greedy Truck-First Sortie-Second Repair

**Algorithm Description** (Sacramento et al., 2019):

```
Algorithm: Greedy Truck-First Sortie-Second Repair
Input: Partial solution s, customer set to insert D
Output: Complete solution s

Phase 1: Truck Route Reconstruction
1. While D ≠ ∅:
   a. Randomly select customer c ∈ D
   b. TruckBestInsertion(c, s)  # Best insertion
   c. D = D \ {c}

Phase 2: Drone Sortie Addition
1. C = AllCustomers(s)
2. While C ≠ ∅:
   a. Randomly select customer c ∈ C
   b. C = C \ {c}
   c. If q_c ≤ Q^D AND Type(c) = Truck:
      d. s' = s  # Save current solution
      e. η = f(s')  # Current cost
      f. Remove c from s
      g. p = FindSortie(c, s, η)  # Find best sortie
      h. If p ≠ ∅:
         i. s = s ∪ {p}  # Add sortie to solution
      i. Else:
         j. s = s'  # Restore original solution
3. Return s
```

**Key Function FindSortie**:
```
Function: FindSortie(c, s, η)
Input: Customer c, partial solution s, threshold cost η
Output: Best sortie p or null

1. best_sortie = null
2. best_saving = 0
3. For each feasible launch point i in s:
   For each feasible recovery point k in s:
      a. Construct sortie = (i, c, k)
      b. If sortie is feasible (satisfies endurance, time constraints):
         c. saving = η - f(s with sortie)
         d. If saving > best_saving:
            e. best_saving = saving
            f. best_sortie = sortie
4. Return best_sortie
```

**Illustration** (see Sacramento Fig. 8):
- Left: Partial solution after customer removal
- Middle: Truck route reconstructed via best insertion
- Right: Final solution with drone sorties added

### 3.2 Nearby-Area Truck-First Repair

**Characteristics**:
- Similar to method 3.1, but different insertion strategy
- Phase 1: Randomly select feasible positions within 5-mile range of customer
- Phase 2: Randomly select sortie with cost increase ≤10% (instead of optimal)
- Increases search diversity, suitable for small instances

### 3.3 Closest Insertion Repair

**Algorithm Description** (Sacramento et al., 2019):

```python
# Algorithm 5 from Sacramento et al., 2019
def closest_insertion_repair(s, D):
    D_N = ∅
    
    while D ≠ ∅:
        c = RandomCustomer(D)
        D = D \ {c}
        c' = NearestCustomer(c, s)
        r = RouteOf(c')
        
        if AttemptBestInsertion(c, r) == false:
            D_N = D_N ∪ {c}
    
    if D_N ≠ ∅:
        s = RepairTruckFirstSortieSecond(D_N, s)
    
    return s
```

**Characteristics**:
- Only attempts to insert customer into route containing its nearest customer
- Considers both truck and drone service modes
- Failed customers are handled using method 3.1

### 3.4 Heavy Insertion Repair

**Algorithm Flow**:
1. Extract all customers with demand > Q^D from D → D_T
2. Randomly select c ∈ D_T, insert via TruckBestInsertion
3. Repeat until D_T = ∅
4. Use method 3.1 for remaining customers

**Advantages**:
- Prioritizes heavy cargo that must be delivered by truck
- Leaves more opportunities for lightweight drone deliveries

### 3.5 Two-Stage Repair (Kitjacharoenchai et al., 2020)

**Three parallel repair operators for 2EVRPD problem**:

1. **Drone Node Insertion**
   ```
   - Insert node into existing drone sub-route
   - Use simplified cheapest insertion heuristic
   - Check capacity and battery constraints
   ```

2. **Truck Node Insertion**
   ```
   - Insert node into truck main route
   - Search all feasible positions, select minimum cost increase
   - Can open new route when capacity is full
   ```

3. **Drone Route Creation**
   ```
   - Insert node j between truck node pair (i,k)
   - Create new drone sub-route: i → j → k
   - Search for node pair with minimum cost increase
   ```

**Repair Process**:
```
For each node to insert:
    solution1 = DroneNodeInsertion(node)
    solution2 = TruckNodeInsertion(node)  
    solution3 = DroneRouteCreation(node)
    current_solution = min(solution1, solution2, solution3)
```

---

## 4. Acceptance Criterion

### 4.1 Simulated Annealing Criterion

**Sacramento et al. (2019) Acceptance Mechanism**:

```
Input: New solution s_t, current solution s, temperature T
Output: Accept or reject

If f(s_t) < f(s):
    Accept s_t  # Always accept better solutions
Else:
    Probabilistic acceptance: P = exp(-(f(s_t) - f(s)) / T)
    Generate random number rand ∈ [0,1]
    If rand < P:
        Accept s_t
    Else:
        Reject s_t
```

### 4.2 Temperature Update Strategy

**Linear Cooling**:
```
T = T_st × (1 - t_elap / t_max)
```

Where:
- T_st: Initial temperature = α × f(s_initial)
- α: Temperature factor (typically 0.004)
- t_elap: Elapsed time
- t_max: Total time limit

**Initial Temperature Calculation**:
```
T_st = 0.004 × f(s_initial)

# Increase by 10% for small instances to avoid too low temperature
If instance_size < threshold:
    T_st = T_st × 1.1
```

### 4.3 Temperature Control Effects

**Graph Analysis** (Sacramento Fig. 9):

**Small Instance (12.10.3)**:
- Optimal found early → flat best solution curve
- Algorithm accepts highly deteriorated solutions (high initial temperature)
- Fluctuations decrease as temperature lowers

**Large Instance (150.10.3)**:
- Best solution curve gradually descends
- More controlled fluctuations
- Only accepts slightly worse solutions in later stages

---

## 5. Weight Update Mechanism

### 5.1 Roulette Wheel Selection

**Operator Selection Probability**:
```
P(select operator i) = w_i / Σ(w_j)  for all j ∈ Ω
```

### 5.2 Weight Update Formula

```
w_{i,j+1} = w_{i,j} × (1 - λ) + λ × σ_i
```

Where:
- w_{i,j}: Weight of operator i at iteration j
- λ: Reaction factor ∈ [0,1] (typically 0.9)
- σ_i: Score of operator i

### 5.3 Scoring Mechanism (Sacramento et al., 2019)

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| σ₁ | New solution is new global best | 33 |
| σ₂ | New solution accepted and better than current | 9 |
| σ₃ | New solution accepted but worse than current | 13 |
| σ₄ | New solution rejected | 0 |

**Update Example**:
```python
# Initialization: all weights equal
w = [1.0, 1.0, 1.0, 1.0]  # 4 repair operators

# Iteration 1: operator 2 finds new global best
w[2] = 1.0 × (1 - 0.9) + 0.9 × 33 = 29.8

# Iteration 2: operator 1 accepted but worse
w[1] = 1.0 × (1 - 0.9) + 0.9 × 13 = 11.8

# Normalize to calculate selection probability
total = sum(w)
prob = [w_i / total for w_i in w]
```

### 5.4 Adaptive Effects

**Advantages**:
- Well-performing operators gain higher weights
- Dynamically adapts to problem characteristics
- Balances exploration and exploitation

**Experimental Observations** (Sacramento Fig. 9 tables):
- Small instances: Repair method 1 more effective (higher acceptance rate)
- Large instances: More balanced contribution across methods
- Algorithm automatically adjusts weight distribution

---

## 6. ALNS Implementations in Different Literature

### 6.1 Sacramento et al. (2019) - VRP-D

**Problem**: Vehicle Routing Problem with Drones
- Multiple trucks + single drone per truck
- Capacity constraints + time limits
- Minimize total cost

**ALNS Configuration**:
- **Destroy operators**: 2 types (random, cluster), equal probability selection
- **Repair operators**: 4 types (greedy truck-first, nearby-area, closest insertion, heavy-first), adaptive selection
- **Acceptance criterion**: Simulated annealing
- **Parameter settings**:
  - λ = 0.9
  - ψ = 0.15
  - T_st = 0.004 × f(s_init)
  - noImprovMax = 1000

**Features**:
- Two-stage initial solution construction (truck routes → drone sorties)
- String relocation local search
- Solution restoration mechanism (after consecutive no-improvement)

### 6.2 Tu et al. (2018) - TSP-mD

**Problem**: Traveling Salesman Problem with Multiple Drones
- Single truck + multiple drones
- Variable drone speeds and battery consumption

**ALNS Features**:
- Specialized destroy/repair operators for multiple drones
- Considers parallel drone operations

### 6.3 Kitjacharoenchai et al. (2020) - 2EVRPD

**Problem**: Two-Echelon Vehicle Routing Problem with Drones
- Two-echelon delivery structure
- Truck main routes + drone sub-routes

**LNS Configuration**:
- **Destroy operators**: 3 types executed sequentially (drone nodes, truck nodes, sub-routes)
- **Repair operators**: 3 types executed in parallel, select best (drone insertion, truck insertion, route creation)
- **Special mechanisms**:
  - Initial solution: DTRC (Drone-Truck Route Construction)
  - Restart strategy: Start from new initial solution after consecutive no-improvement

**Algorithm Flow**:
```
1. Generate initial solution via DTRC
2. While not reaching max iterations:
   a. Sequentially execute 3 destroy operators
   b. Execute 3 repair operators in parallel, select best
   c. Compare new solution with current solution
   d. Update global best
   e. Restart when no improvement reaches threshold
3. Return global best solution
```

### 6.4 Mara et al. (2022) - FSTSP Variant

**Features**:
- Drone can deliver to multiple customers per flight
- ALNS specifically handles multi-drop routes

### 6.5 Young Jeong & Lee (2023) - DRP-T

**Problem**: Drone Routing Problem with Truck
**Method**: Memetic Algorithm with Crossover Heuristic (MACH)
- Not pure ALNS, but incorporates destroy-repair concept
- Destroy: Select best destroy points in drone routes
- Repair: Connect with next truck route

---

## 7. Algorithm Parameter Settings

### 7.1 Sacramento et al. (2019) Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Reaction factor | λ | 0.9 | Weight update rate |
| Destruction ratio | ψ | 0.15 | Remove 15% of customers |
| Minimum removal | c_low | 1-3 | Random |
| Maximum removal | c_lim | 40 | Fixed |
| Temperature factor | T_st | 0.004 | Initial temperature coefficient |
| No-improvement limit | noImprovMax | 1000 | Triggers solution restoration |
| Score σ₁ | - | 33 | New global best |
| Score σ₂ | - | 9 | Accept better solution |
| Score σ₃ | - | 13 | Accept worse solution |
| Score σ₄ | - | 0 | Reject solution |

### 7.2 Kitjacharoenchai et al. (2020) Parameters

| Parameter | Description |
|-----------|-------------|
| v₂ | Drone node removal ratio |
| v₃ | Truck node removal ratio |
| v_t | Sub-route removal ratio |
| q_rs | Restart threshold (consecutive no-improvement count) |
| +pi_max | Maximum iterations |

### 7.3 Parameter Tuning Recommendations

**Destruction Ratio ψ**:
- Too small (< 0.1): Difficult to escape local optima
- Too large (> 0.3): Difficult to reconstruct quality solutions
- Recommended: 0.15-0.20

**Reaction Factor λ**:
- Close to 1 (e.g., 0.9): Rapid response to operator performance
- Smaller (e.g., 0.5): Smoother weight changes
- Recommended: 0.8-0.95

**Temperature Factor**:
- Adjust based on instance size
- Small instances: Increase 10% to avoid premature convergence
- Large instances: Standard settings

**Score Settings**:
- σ₁ >> σ₂ > σ₃: Emphasize finding global optima
- σ₃ > 0: Encourage exploration of worse solutions
- σ₄ = 0: Don't penalize rejected operators

---

## 8. Performance and Results

### 8.1 Computational Efficiency

**Sacramento et al. (2019) Experiments**:

**Small Instances (6-20 customers)**:
- Time limit: 5 minutes
- Average iterations: 20,000-400,000
- Optimality gap: Average < 1%

**Large Instances (50-150 customers)**:
- Time limit: 5 minutes  
- Average iterations: 40,000-500,000
- Savings vs. truck-only: 20-30%

### 8.2 Operator Performance Analysis

**Repair Operator Acceptance Rate** (Instance 12.10.3):
- Method 1 (greedy truck-first): High acceptance rate, finds global best multiple times
- Method 2 (nearby-area): Medium performance
- Method 3 (closest insertion): Lower acceptance rate
- Method 4 (heavy-first): Suitable for heavy cargo instances

**Adaptive Effects**:
- Small instances: Method 1 dominates
- Large instances: Balanced contribution across methods
- Weight automatically adjusts to reflect problem characteristics

### 8.3 Comparison with Other Methods

**Kitjacharoenchai et al. (2020)**:
- LNS vs MILP: Significantly reduced solution time for large instances
- LNS vs Heuristics: Solution quality improved 15-25%

**Young Jeong & Lee (2023)**:
- MACH vs GA/ACO/SA: Objective value 10-25% better on average
- Demonstrates advantages of destroy-repair concepts in MACH

---

## 9. Key Insights and Recommendations

### 9.1 Advantages of ALNS for Drone-Truck Problems

1. **Flexibility**: Easy to adapt to different constraints (battery, weight, time windows)
2. **Adaptability**: Automatically adjusts operator usage to problem characteristics
3. **Robustness**: Insensitive to initial solution quality
4. **Scalability**: Supports small to large-scale instances

### 9.2 Operator Design Principles

**Destroy Operators**:
- Random destroy: Ensures diversity
- Cluster destroy: Concentrated destruction of local structures
- Hierarchical destroy: For two-echelon problems (e.g., 2EVRPD)

**Repair Operators**:
- Greedy strategies: Ensure solution quality
- Diversification strategies: Increase exploration
- Staged repair: Truck first, then drone
- Parallel attempts: Multiple approaches, select best

### 9.3 Parameter Setting Guidelines

1. **Initialization**: Start with equal weights, let algorithm adapt
2. **Destruction degree**: 15-20% is empirically optimal
3. **Temperature control**: Linear cooling is simple and effective
4. **Restoration mechanism**: Avoid wasting time on poor solutions

### 9.4 Future Research Directions

1. **Learning-based ALNS**: Use machine learning to predict operator selection
2. **Parallel ALNS**: Multi-threading to accelerate large-scale problems
3. **Hybrid methods**: ALNS + exact algorithms (e.g., branch-and-bound)
4. **Adaptive destruction degree**: Dynamically adjust β value

---

## 10. Implementation Framework

### 10.1 Python Pseudocode Framework

```python
class ALNS:
    def __init__(self, problem):
        self.problem = problem
        self.destroy_ops = [random_destroy, cluster_destroy]
        self.repair_ops = [greedy_repair, nearby_repair, 
                          closest_repair, heavy_repair]
        self.weights = self.init_weights()
        
    def solve(self, time_limit):
        s_current = self.generate_initial_solution()
        s_best = s_current.copy()
        T = self.calc_initial_temp(s_current)
        
        t_start = time.time()
        iter_no_improve = 0
        
        while time.time() - t_start < time_limit:
            # Select operators
            destroy_op = self.roulette_wheel_select(
                self.destroy_ops, self.weights['destroy'])
            repair_op = self.roulette_wheel_select(
                self.repair_ops, self.weights['repair'])
            
            # Destroy and repair
            s_partial = destroy_op(s_current)
            s_new = repair_op(s_partial)
            
            # Acceptance criterion
            if self.accept(s_new, s_current, T):
                s_current = s_new
                iter_no_improve = 0
                
                if s_new.cost < s_best.cost:
                    s_best = s_new
                    score = SIGMA_1  # New global best
                else:
                    score = SIGMA_2  # Accept better solution
            else:
                score = SIGMA_3 if random() < 0.5 else SIGMA_4
                iter_no_improve += 1
            
            # Update weights
            self.update_weights(repair_op, score)
            
            # Update temperature
            t_elap = time.time() - t_start
            T = self.update_temperature(T, t_elap, time_limit)
            
            # Restoration mechanism
            if iter_no_improve > NO_IMPROVE_MAX:
                s_current = s_best.copy()
                iter_no_improve = 0
        
        return s_best
    
    def accept(self, s_new, s_current, T):
        if s_new.cost < s_current.cost:
            return True
        else:
            prob = exp(-(s_new.cost - s_current.cost) / T)
            return random() < prob
    
    def update_weights(self, op, score):
        idx = self.repair_ops.index(op)
        self.weights['repair'][idx] = \
            self.weights['repair'][idx] * (1 - LAMBDA) + \
            LAMBDA * score
```

### 10.2 Key Data Structures

```python
class Solution:
    def __init__(self):
        self.truck_routes = []  # Truck routes
        self.drone_sorties = []  # Drone sorties
        self.cost = 0.0
        
    def copy(self):
        # Deep copy solution
        pass
    
    def is_feasible(self):
        # Check constraints: capacity, time, drone endurance, etc.
        pass

class TruckRoute:
    def __init__(self):
        self.sequence = []  # Visit sequence
        self.load = 0.0
        self.duration = 0.0

class DroneSortie:
    def __init__(self, launch, customer, recover):
        self.launch = launch      # Launch point
        self.customer = customer  # Delivery customer
        self.recover = recover    # Recovery point
        self.duration = 0.0       # Flight time
```

---

## 11. Summary

ALNS algorithm demonstrates excellent performance in drone-truck collaborative delivery problems, primarily manifested in:

1. **Adaptive capability**: Automatically adapts to problem characteristics through weight update mechanism
2. **Solution quality**: 15-25% improvement over initial heuristic solutions
3. **Computational efficiency**: Solves 100+ customer instances within 5 minutes
4. **Flexibility**: Easy to extend to different problem variants

**Core Success Factors**:
- Diverse combination of destroy/repair operators
- Simulated annealing acceptance criterion balances exploration and exploitation
- Adaptive weight mechanism identifies effective operators
- Solution restoration mechanism avoids getting stuck in poor solutions

**Practical Recommendations**:
- Start with classic configuration (λ=0.9, ψ=0.15)
- Design specialized operators for specific problems
- Thoroughly test different parameter combinations
- Design initial solution construction based on problem characteristics

---

## References

1. Sacramento, D., Pisinger, D., & Ropke, S. (2019). An adaptive large neighborhood search metaheuristic for the vehicle routing problem with drones. Transportation Research Part C, 102, 289-315.

2. Kitjacharoenchai, P., Ventresca, M., Moshref-Javadi, M., Lee, S., Tanchoco, J. M., & Brunese, P. A. (2020). Two echelon vehicle routing problem with drones in last mile delivery. International Journal of Production Economics, 225, 107598.

3. Tu, P. A., Dat, N. T., & Dung, P. Q. (2018). Traveling salesman problem with multiple drones. Proceedings of the Ninth International Symposium on Information and Communication Technology, 46-53.

4. Mara, S. T. W., Rifai, A. P., Noche, B., & Prakosha, R. (2022). An adaptive large neighborhood search heuristic for the flying sidekick traveling salesman problem with multiple drops per drone trip. Expert Systems with Applications, 190, 116125.

5. Young Jeong, H., & Lee, S. (2023). Drone routing problem with truck: Optimization and quantitative analysis. Expert Systems with Applications, 227, 120260.

6. Ropke, S., & Pisinger, D. (2006). An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows. Transportation Science, 40(4), 455-472.

7. Pisinger, D., & Ropke, S. (2010). Large neighborhood search. In Handbook of Metaheuristics (pp. 399-419). Springer.
