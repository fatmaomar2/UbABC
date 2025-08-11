# task3edit.py
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import csv
import os

@dataclass
class PMResources:
    cpu: int
    memory: int
    available_cpu: int
    available_memory: int
    vms: List[int]  # List of VM IDs

class PhysicalMachine:
    def __init__(self, id: int, cpu: int, memory: int):
        self.id = id
        self.resources = PMResources(cpu, memory, cpu, memory, [])
        self.energy_consumption = 0  # For energy efficiency calculation

class VirtualMachine:
    def __init__(self, id: int, cpu_req: int, memory_req: int):
        self.id = id
        self.cpu_req = cpu_req
        self.memory_req = memory_req
        self.assigned_pm = None

class Solution:
    def __init__(self, assignment: List[int], fitness: float = 0.0):
        self.assignment = assignment  # assignment[vm_id] = pm_id
        self.fitness = fitness
        self.trial_counter = 0
        self.resource_usage = None  # Will store PM resource usage
        self.migration_count = 0  # Track migrations for this solution

class UbABC:
    def __init__(self, num_employed_bees: int, num_onlooker_bees: int, 
                 max_iterations: int, limit: int, 
                 w1: float = 0.4, w2: float = 0.4, w3: float = 0.2):
        self.num_employed_bees = num_employed_bees
        self.num_onlooker_bees = num_onlooker_bees
        self.max_iterations = max_iterations
        self.limit = limit
        self.w1 = w1  # Weight for utilization
        self.w2 = w2  # Weight for load balancing
        self.w3 = w3  # Weight for energy efficiency
        
        # For tracking progress
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.active_pms_history = []
        self.best_solution = None  # Store the best solution
        
    def initialize_population(self, pm_list: List[PhysicalMachine], 
                             vm_list: List[VirtualMachine]) -> List[Solution]:
        """Create initial population of solutions"""
        population = []
        num_vms = len(vm_list)
        num_pms = len(pm_list)
        
        for _ in range(self.num_employed_bees):
            assignment = [-1] * num_vms  # -1 means unassigned
            
            # Create a copy of PM resources for this solution
            pm_resources = [copy.deepcopy(pm.resources) for pm in pm_list]
            
            # Assign each VM to a random suitable PM
            for vm in vm_list:
                suitable_pms = [i for i in range(num_pms) 
                               if (pm_resources[i].available_cpu >= vm.cpu_req and 
                                   pm_resources[i].available_memory >= vm.memory_req)]
                
                if suitable_pms:
                    pm_idx = random.choice(suitable_pms)
                    assignment[vm.id] = pm_idx
                    pm_resources[pm_idx].available_cpu -= vm.cpu_req
                    pm_resources[pm_idx].available_memory -= vm.memory_req
                    pm_resources[pm_idx].vms.append(vm.id)
            
            # Calculate fitness (initial migration count is 0)
            fitness = self.calculate_fitness(assignment, pm_resources, pm_list, vm_list, 0)
            solution = Solution(assignment, fitness)
            solution.resource_usage = pm_resources
            population.append(solution)
            
        return population
    
    def calculate_fitness(self, assignment: List[int], pm_resources: List[PMResources],
                          pm_list: List[PhysicalMachine], vm_list: List[VirtualMachine], 
                          migration_count: int) -> float:
        """Calculate multi-objective fitness function"""
        num_pms = len(pm_list)
        
        # Calculate utilization component
        total_cpu_util = 0
        total_mem_util = 0
        active_pms = 0
        
        for i in range(num_pms):
            if pm_resources[i].vms:  # PM has VMs
                active_pms += 1
                cpu_util = (pm_list[i].resources.cpu - pm_resources[i].available_cpu) / pm_list[i].resources.cpu
                mem_util = (pm_list[i].resources.memory - pm_resources[i].available_memory) / pm_list[i].resources.memory
                total_cpu_util += cpu_util
                total_mem_util += mem_util
        
        avg_util = (total_cpu_util + total_mem_util) / (2 * num_pms) if num_pms > 0 else 0
        f_util = avg_util
        
        # Calculate load balancing component (standard deviation of utilization)
        cpu_utils = []
        mem_utils = []
        for i in range(num_pms):
            if pm_resources[i].vms:
                cpu_utils.append((pm_list[i].resources.cpu - pm_resources[i].available_cpu) / pm_list[i].resources.cpu)
                mem_utils.append((pm_list[i].resources.memory - pm_resources[i].available_memory) / pm_list[i].resources.memory)
        
        all_utils = cpu_utils + mem_utils
        f_balance = np.std(all_utils) if all_utils else 0
        
        # Calculate energy efficiency component
        f_energy = active_pms / num_pms if num_pms > 0 else 0  # Fewer active PMs = better energy efficiency
        
        # Calculate migration component
        f_mig = migration_count / len(vm_list) if vm_list else 0
        
        # Combined fitness with weights
        fitness = (self.w1 * f_util + 
                  self.w2 * f_balance + 
                  self.w3 * f_energy + 
                  0.1 * f_mig)  # Small weight for migrations
        
        return fitness
    
    def generate_neighbor_solution(self, solution: Solution, pm_list: List[PhysicalMachine], 
                                 vm_list: List[VirtualMachine]) -> Solution:
        """Generate a neighbor solution using different move strategies"""
        new_assignment = solution.assignment.copy()
        new_pm_resources = [copy.deepcopy(res) for res in solution.resource_usage]
        
        # Initialize migration count for neighbor
        migrations = 0
        
        # Select a random VM to move
        vm_id = random.randint(0, len(vm_list) - 1)
        vm = vm_list[vm_id]
        current_pm = new_assignment[vm_id]
        
        # Choose move strategy based on probability
        move_type = random.random()
        
        if move_type < 0.7:  # 70% chance: single VM migration
            # Find suitable PMs (excluding current one)
            suitable_pms = [i for i in range(len(pm_list)) 
                           if i != current_pm and 
                           new_pm_resources[i].available_cpu >= vm.cpu_req and 
                           new_pm_resources[i].available_memory >= vm.memory_req]
            
            if suitable_pms:
                # Select PM based on probability (prefer less utilized PMs)
                probs = [1.0 / (new_pm_resources[i].available_cpu + new_pm_resources[i].available_memory + 1) 
                        for i in suitable_pms]
                probs = [p/sum(probs) for p in probs]
                new_pm = np.random.choice(suitable_pms, p=probs)
                
                # Move VM
                if current_pm != -1:
                    new_pm_resources[current_pm].available_cpu += vm.cpu_req
                    new_pm_resources[current_pm].available_memory += vm.memory_req
                    new_pm_resources[current_pm].vms.remove(vm_id)
                
                new_assignment[vm_id] = new_pm
                new_pm_resources[new_pm].available_cpu -= vm.cpu_req
                new_pm_resources[new_pm].available_memory -= vm.memory_req
                new_pm_resources[new_pm].vms.append(vm_id)
                
                # Increment migration counter
                migrations += 1
                
        elif move_type < 0.9:  # 20% chance: VM swap
            if current_pm != -1:
                # Find another VM in a different PM to swap with
                other_vm_id = random.choice([i for i in range(len(vm_list)) 
                                           if new_assignment[i] != -1 and 
                                           new_assignment[i] != current_pm])
                other_vm = vm_list[other_vm_id]
                other_pm = new_assignment[other_vm_id]
                
                # Check if swap is feasible
                if (new_pm_resources[current_pm].available_cpu + vm.cpu_req >= other_vm.cpu_req and
                    new_pm_resources[current_pm].available_memory + vm.memory_req >= other_vm.memory_req and
                    new_pm_resources[other_pm].available_cpu + other_vm.cpu_req >= vm.cpu_req and
                    new_pm_resources[other_pm].available_memory + other_vm.memory_req >= vm.memory_req):
                    
                    # Perform swap
                    new_pm_resources[current_pm].available_cpu += vm.cpu_req - other_vm.cpu_req
                    new_pm_resources[current_pm].available_memory += vm.memory_req - other_vm.memory_req
                    new_pm_resources[other_pm].available_cpu += other_vm.cpu_req - vm.cpu_req
                    new_pm_resources[other_pm].available_memory += other_vm.memory_req - vm.memory_req
                    
                    new_pm_resources[current_pm].vms.remove(vm_id)
                    new_pm_resources[current_pm].vms.append(other_vm_id)
                    new_pm_resources[other_pm].vms.remove(other_vm_id)
                    new_pm_resources[other_pm].vms.append(vm_id)
                    
                    new_assignment[vm_id] = other_pm
                    new_assignment[other_vm_id] = current_pm
                    
                    # Increment migration counter (2 migrations for swap)
                    migrations += 2
        
        # 10% chance: no change (exploration)
        
        # Calculate fitness for new solution
        total_migrations = solution.migration_count + migrations
        fitness = self.calculate_fitness(new_assignment, new_pm_resources, pm_list, vm_list, total_migrations)
        neighbor = Solution(new_assignment, fitness)
        neighbor.resource_usage = new_pm_resources
        neighbor.migration_count = total_migrations
        
        return neighbor
    
    def employed_bee_phase(self, population: List[Solution], pm_list: List[PhysicalMachine], 
                           vm_list: List[VirtualMachine]) -> List[Solution]:
        """Employed bee phase: each solution explores its neighborhood"""
        new_population = []
        
        for solution in population:
            neighbor = self.generate_neighbor_solution(solution, pm_list, vm_list)
            
            # Greedy selection
            if neighbor.fitness < solution.fitness:
                neighbor.trial_counter = 0
                new_population.append(neighbor)
            else:
                solution.trial_counter += 1
                new_population.append(solution)
                
        return new_population
    
    def onlooker_bee_phase(self, population: List[Solution], pm_list: List[PhysicalMachine], 
                          vm_list: List[VirtualMachine]) -> List[Solution]:
        """Onlooker bee phase: solutions selected based on fitness probability"""
        # Calculate selection probabilities (lower fitness = higher probability)
        fitness_values = [sol.fitness for sol in population]
        max_fitness = max(fitness_values) if fitness_values else 1
        probabilities = [(max_fitness - f + 1e-6) for f in fitness_values]
        total_prob = sum(probabilities)
        probabilities = [p/total_prob for p in probabilities]
        
        new_population = population.copy()
        
        for _ in range(self.num_onlooker_bees):
            # Select a solution based on probability
            selected_idx = np.random.choice(len(population), p=probabilities)
            solution = population[selected_idx]
            
            # Generate neighbor solution
            neighbor = self.generate_neighbor_solution(solution, pm_list, vm_list)
            
            # Greedy selection
            if neighbor.fitness < solution.fitness:
                neighbor.trial_counter = 0
                new_population[selected_idx] = neighbor
            else:
                new_population[selected_idx].trial_counter += 1
                
        return new_population
    
    def scout_bee_phase(self, population: List[Solution], pm_list: List[PhysicalMachine], 
                       vm_list: List[VirtualMachine]) -> List[Solution]:
        """Scout bee phase: replace abandoned solutions with new random ones"""
        new_population = []
        
        for solution in population:
            if solution.trial_counter > self.limit:
                # Replace with new random solution
                new_solution = self.initialize_population(pm_list, vm_list)[0]
                new_solution.trial_counter = 0
                new_population.append(new_solution)
            else:
                new_population.append(solution)
                
        return new_population
    
    def optimized_vm_placement(self, pm_list: List[PhysicalMachine], 
                             vm_list: List[VirtualMachine]) -> Solution:
        """Main UbABC algorithm for VM placement optimization"""
        # Initialize population
        population = self.initialize_population(pm_list, vm_list)
        
        # Track best solution
        best_solution = min(population, key=lambda x: x.fitness)
        self.best_fitness_history.append(best_solution.fitness)
        self.avg_fitness_history.append(np.mean([sol.fitness for sol in population]))
        
        # Count active PMs in best solution
        active_pms = len(set(best_solution.assignment) - {-1})
        self.active_pms_history.append(active_pms)
        
        # Main loop
        for iteration in range(self.max_iterations):
            # Employed bee phase
            population = self.employed_bee_phase(population, pm_list, vm_list)
            
            # Onlooker bee phase
            population = self.onlooker_bee_phase(population, pm_list, vm_list)
            
            # Scout bee phase
            population = self.scout_bee_phase(population, pm_list, vm_list)
            
            # Update best solution
            current_best = min(population, key=lambda x: x.fitness)
            if current_best.fitness < best_solution.fitness:
                best_solution = current_best
            
            # Track progress
            self.best_fitness_history.append(best_solution.fitness)
            self.avg_fitness_history.append(np.mean([sol.fitness for sol in population]))
            
            active_pms = len(set(best_solution.assignment) - {-1})
            self.active_pms_history.append(active_pms)
            
            # Print progress
            if iteration % 10 == 0 or iteration == self.max_iterations - 1:
                print(f"Iteration {iteration}: Best Fitness = {best_solution.fitness:.4f}, "
                      f"Active PMs = {active_pms}")
        
        # Store the best solution in the instance
        self.best_solution = best_solution
        return best_solution

    def optimized_vm_placement_baseline(self, pm_list: List[PhysicalMachine],
                                        vm_list: List[VirtualMachine]) -> Solution:
        """
        Baseline ABC (classic) run — same operators but **migration contribution ignored**
        so we can compare fairly without the migration penalty/handling.
        """
        # We'll reuse most code from optimized_vm_placement but force migration_count = 0 in fitness calculations.
        # Initialize population
        population = self.initialize_population(pm_list, vm_list)
        
        # For baseline histories, use separate local lists then return them for comparison externally
        best_solution = min(population, key=lambda x: x.fitness)
        baseline_best_history = [best_solution.fitness]
        baseline_avg_history = [np.mean([sol.fitness for sol in population])]
        baseline_active_pms = [len(set(best_solution.assignment) - {-1})]
        
        for iteration in range(self.max_iterations):
            # Employed
            new_population = []
            for sol in population:
                neighbor = self.generate_neighbor_solution(sol, pm_list, vm_list)
                # Recompute fitness ignoring migrations (set migration_count=0)
                neighbor.fitness = self.calculate_fitness(neighbor.assignment, neighbor.resource_usage, pm_list, vm_list, 0)
                sol.fitness = self.calculate_fitness(sol.assignment, sol.resource_usage, pm_list, vm_list, 0)
                
                if neighbor.fitness < sol.fitness:
                    neighbor.trial_counter = 0
                    new_population.append(neighbor)
                else:
                    sol.trial_counter += 1
                    new_population.append(sol)
            population = new_population
            
            # Onlooker
            # compute probabilities using baseline fitnesses (lower is better)
            fitness_values = [sol.fitness for sol in population]
            max_f = max(fitness_values) if fitness_values else 1
            probs = [(max_f - f + 1e-6) for f in fitness_values]
            s = sum(probs)
            probs = [p/s for p in probs]
            for _ in range(self.num_onlooker_bees):
                idx = np.random.choice(len(population), p=probs)
                sol = population[idx]
                neighbor = self.generate_neighbor_solution(sol, pm_list, vm_list)
                neighbor.fitness = self.calculate_fitness(neighbor.assignment, neighbor.resource_usage, pm_list, vm_list, 0)
                sol.fitness = self.calculate_fitness(sol.assignment, sol.resource_usage, pm_list, vm_list, 0)
                if neighbor.fitness < sol.fitness:
                    neighbor.trial_counter = 0
                    population[idx] = neighbor
                else:
                    population[idx].trial_counter += 1
            
            # Scout
            for i, sol in enumerate(population):
                if sol.trial_counter > self.limit:
                    population[i] = self.initialize_population(pm_list, vm_list)[0]
            
            # Update best
            current_best = min(population, key=lambda x: x.fitness)
            if current_best.fitness < best_solution.fitness:
                best_solution = current_best
            
            baseline_best_history.append(best_solution.fitness)
            baseline_avg_history.append(np.mean([sol.fitness for sol in population]))
            baseline_active_pms.append(len(set(best_solution.assignment) - {-1}))
            
            if iteration % 10 == 0 or iteration == self.max_iterations - 1:
                print(f"[Baseline] Iter {iteration}: Best={best_solution.fitness:.4f}")
        
        # attach baseline histories to a simple container (Solution object + extras)
        best_solution._baseline_best_history = baseline_best_history
        best_solution._baseline_avg_history = baseline_avg_history
        best_solution._baseline_active_pms = baseline_active_pms
        
        return best_solution

# Helper run function for baseline ABC
def run_abc(pm_list: List[PhysicalMachine], vm_list: List[VirtualMachine],
            num_employed=10, num_onlooker=10, max_iterations=100, limit=20) -> Tuple[Solution, dict]:
    abc = UbABC(num_employed, num_onlooker, max_iterations, limit)
    best = abc.optimized_vm_placement_baseline(pm_list, vm_list)
    results = {
        "best_solution": best,
        "best_history": getattr(best, "_baseline_best_history", []),
        "avg_history": getattr(best, "_baseline_avg_history", []),
        "active_pms_history": getattr(best, "_baseline_active_pms", []),
    }
    return best, results

# Comparison and saving function
def compare_and_save_results(abc_results: dict, ub_results: dict, out_folder: str = "output/comparison"):
    """
    abc_results, ub_results: dicts with keys 'best_history','avg_history','active_pms_history','best_solution'
    Saves:
      - comparison_fitness.png (both curves)
      - comparison_final_bar.png (final best values)
      - comparison_summary.csv
    """
    os.makedirs(out_folder, exist_ok=True)
    # Ensure histories same length by padding shorter with last value
    def pad(a, n):
        if not a:
            return [0]*n
        if len(a) >= n:
            return a
        return a + [a[-1]] * (n - len(a))
    
    abc_h = abc_results.get("best_history", [])
    ub_h = ub_results.get("best_history", [])
    n = max(len(abc_h), len(ub_h))
    abc_h = pad(abc_h, n)
    ub_h = pad(ub_h, n)
    
    # Plot fitness curves
    plt.figure(figsize=(8,5))
    plt.plot(range(len(abc_h)), abc_h, label="ABC (baseline)")
    plt.plot(range(len(ub_h)), ub_h, label="UbABC (proposed)")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("ABC vs UbABC - Best Fitness over Iterations")
    plt.legend()
    plt.grid(True)
    comp_curve = os.path.join(out_folder, "comparison_fitness.png")
    plt.tight_layout()
    plt.savefig(comp_curve)
    plt.close()
    
    # Final best comparison bar
    abc_final = abc_h[-1] if abc_h else float('nan')
    ub_final = ub_h[-1] if ub_h else float('nan')
    plt.figure(figsize=(6,4))
    plt.bar(["ABC","UbABC"], [abc_final, ub_final], color=['#1f77b4','#ff7f0e'])
    plt.ylabel("Final Best Fitness")
    plt.title("Final Best Fitness Comparison")
    comp_bar = os.path.join(out_folder, "comparison_final_bar.png")
    plt.tight_layout()
    plt.savefig(comp_bar)
    plt.close()
    
    # Save CSV summary
    csv_path = os.path.join(out_folder, "comparison_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm","FinalBestFitness","AvgFinalFitness","FinalActivePMs"])
        # compute avg final fitness from avg_history if present
        abc_avg_final = abc_results.get("avg_history", [np.nan])[-1] if abc_results.get("avg_history") else np.nan
        ub_avg_final = ub_results.get("avg_history", [np.nan])[-1] if ub_results.get("avg_history") else np.nan
        abc_active = abc_results.get("active_pms_history", [np.nan])[-1] if abc_results.get("active_pms_history") else np.nan
        ub_active = ub_results.get("active_pms_history", [np.nan])[-1] if ub_results.get("active_pms_history") else np.nan
        writer.writerow(["ABC", abc_final, abc_avg_final, abc_active])
        writer.writerow(["UbABC", ub_final, ub_avg_final, ub_active])
    
    return {
        "curve_png": comp_curve,
        "bar_png": comp_bar,
        "csv": csv_path
    }

# ---------- Data helpers ----------
def load_data_from_file(file_path: str) -> Tuple[List[PhysicalMachine], List[VirtualMachine]]:
    """Load PM and VM data from a CSV or TXT file (same format expected)"""
    pm_list = []
    vm_list = []
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            # allow whitespace or header differences
            tag = row[0].strip().lower()
            if tag == 'pm':
                pm_id = int(row[1])
                cpu = int(row[2])
                memory = int(row[3])
                pm_list.append(PhysicalMachine(pm_id, cpu, memory))
            elif tag == 'vm':
                vm_id = int(row[1])
                cpu_req = int(row[2])
                memory_req = int(row[3])
                vm_list.append(VirtualMachine(vm_id, cpu_req, memory_req))
    return pm_list, vm_list

def generate_random_data(num_pms: int = 5, num_vms: int = 8) -> Tuple[List[PhysicalMachine], List[VirtualMachine]]:
    """Generate random PM and VM data"""
    pm_list = []
    vm_list = []
    
    # Generate PMs with random resources
    for i in range(num_pms):
        cpu = random.randint(800, 2000)  # CPU between 800 and 2000 MIPS
        memory = random.randint(1024, 8192)  # Memory between 1GB and 8GB
        pm_list.append(PhysicalMachine(i, cpu, memory))
    
    # Generate VMs with random requirements
    for i in range(num_vms):
        cpu_req = random.randint(100, 500)  # CPU requirement between 100 and 500 MIPS
        memory_req = random.randint(256, 2048)  # Memory requirement between 256MB and 2GB
        vm_list.append(VirtualMachine(i, cpu_req, memory_req))
    
    return pm_list, vm_list

def save_data_to_file(pm_list: List[PhysicalMachine], vm_list: List[VirtualMachine], file_path: str):
    """Save PM and VM data to a CSV file"""
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write PM data
        for pm in pm_list:
            writer.writerow(['pm', pm.id, pm.resources.cpu, pm.resources.memory])
        # Write VM data
        for vm in vm_list:
            writer.writerow(['vm', vm.id, vm.cpu_req, vm.memory_req])
