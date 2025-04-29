import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Parse the .dat file
def parse_dat_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip comment lines and metadata
    lines_multi = [lines[1].strip().split("=")[-1]]
    lines = [line for line in lines if not line.startswith("#") and line.strip()]
    
    multi_skilling_level = int(lines_multi[0].strip())
    problem_type = int(lines[0].split("=")[-1].strip())
    num_jobs = int(lines[1].split("=")[-1].strip())
    
    job_times = []
    for i in range(2, 3 + (num_jobs-1)):
        start, end = map(int, lines[i].strip().split())
        job_times.append((start, end))
    
    num_shifts = int(lines[(num_jobs+2)].split("=")[-1].strip())
    shift_qualifications = []
    for i in range((3 + num_jobs), (4 + num_jobs + num_shifts-1)):
        qualified_jobs = list(map(int, lines[i].strip().split()[1:]))
        shift_qualifications.append(qualified_jobs)

    return {
        "multi_skilling_level": multi_skilling_level,
        "problem_type": problem_type,
        "num_jobs": num_jobs,
        "job_times": job_times,
        "num_shifts": num_shifts,
        "shift_qualifications": shift_qualifications
    }

# Define the problem as a multi-objective optimization
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))  # Minimize three objectives
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Define the individual and population
def setup_toolbox(num_jobs, num_shifts, shift_qualifications):
    def generate_valid_shift(job_idx):
        # Return only a valid shift for the job
        valid_shifts = [i for i, qualified in enumerate(shift_qualifications) if job_idx in qualified]
        return random.choice(valid_shifts) if valid_shifts else random.randint(0, num_shifts - 1)

    def individual_generator():
        return creator.Individual([generate_valid_shift(i) for i in range(num_jobs)])
    
    toolbox.register("individual", individual_generator)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the objective functions
def evaluate(individual, job_times, shift_qualifications):
    shifts_used = len(set(individual))
    
    preference_deviation = 0
    for job_idx, shift_idx in enumerate(individual):
        if job_idx not in shift_qualifications[shift_idx]:
            preference_deviation += 1

    shift_counts = {}
    for shift_idx in individual:
        shift_counts[shift_idx] = shift_counts.get(shift_idx, 0) + 1
    
    workload_values = list(shift_counts.values())
    workload_imbalance = np.std(workload_values) if len(workload_values) > 1 else 0
    
    return shifts_used, preference_deviation, workload_imbalance

# Mutation that respects qualification
def mutate_valid_shift(individual, num_shifts, shift_qualifications, indpb):
    for job_idx in range(len(individual)):
        if random.random() < indpb:
            valid_shifts = [s for s in range(num_shifts) if job_idx in shift_qualifications[s]]
            if valid_shifts:
                individual[job_idx] = random.choice(valid_shifts)
    return individual,

# Crossover with repair
def crossover_with_repair(ind1, ind2, shift_qualifications):
    tools.cxTwoPoint(ind1, ind2)
    for ind in (ind1, ind2):
        for job_idx, shift_idx in enumerate(ind):
            if job_idx not in shift_qualifications[shift_idx]:
                valid_shifts = [s for s in range(len(shift_qualifications)) if job_idx in shift_qualifications[s]]
                if valid_shifts:
                    ind[job_idx] = random.choice(valid_shifts)
    return ind1, ind2

# Main function
def main():
    random.seed(21)
    extract_dir = "sdf"
    dat_files = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.dat')]
    
    for dat_file in dat_files:
        print(f"\nProcessing file: {dat_file}")
        data = parse_dat_file(dat_file)
        num_jobs = data["num_jobs"]
        num_shifts = data["num_shifts"]
        job_times = data["job_times"]
        shift_qualifications = data["shift_qualifications"]
        multi_skilling_level = data["multi_skilling_level"]
        
        setup_toolbox(num_jobs, num_shifts, shift_qualifications)
        
        toolbox.register("evaluate", evaluate, 
                         job_times=job_times, 
                         shift_qualifications=shift_qualifications)
        toolbox.register("mate", crossover_with_repair, shift_qualifications=shift_qualifications)
        toolbox.register("mutate", mutate_valid_shift, 
                         num_shifts=num_shifts, 
                         shift_qualifications=shift_qualifications,
                         indpb=0.2)
        toolbox.register("select", tools.selNSGA2)
        
        # Updated parameters
        population = toolbox.population(n=300)  # Best Population Size
        NGEN, CXPB, MUTPB = 50, 0.9, 0.15  # Best Generations, Crossover, and Mutation Probabilities
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)
        stats.register("avg", np.mean, axis=0)
        stats.register("max", np.max, axis=0)
        
        logbook = tools.Logbook()
        
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        for gen in range(NGEN):
            if gen % 10 == 0:
                print(f"Generation {gen}/{NGEN}")
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values
            
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            population[:] = offspring
            record = stats.compile(population)
            logbook.record(gen=gen, **record)
        
        print(f"Final statistics: {logbook.select('min')[-1]}")
        
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        pareto_fitness = np.array([ind.fitness.values for ind in pareto_front])
        
        # Plot Pareto Front in 3D
        if len(pareto_front) > 0:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            pareto_fitness_sorted = pareto_fitness[np.argsort(pareto_fitness[:, 0])]
            ax.plot(pareto_fitness_sorted[:, 0], pareto_fitness_sorted[:, 1], pareto_fitness_sorted[:, 2],
                    color='blue', marker='o', linestyle='-', linewidth=2)
            ax.set_xlabel('Total Shifts Used')
            ax.set_ylabel('Preference Deviation')
            ax.set_zlabel('Workload Imbalance')
            ax.set_title(f'Pareto Front for {os.path.basename(dat_file)}')
            plt.tight_layout()
            plt.show()

            # Plot 2D projections of Pareto Front
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.scatter(pareto_fitness[:, 0], pareto_fitness[:, 1], c='blue')
            plt.xlabel('Total Shifts Used')
            plt.ylabel('Preference Deviation')
            plt.title('Total Shifts Used vs Preference Deviation')
            plt.grid(True)

            plt.subplot(132)
            plt.scatter(pareto_fitness[:, 0], pareto_fitness[:, 2], c='red')
            plt.xlabel('Total Shifts Used')
            plt.ylabel('Workload Imbalance')
            plt.title('Total Shifts Used vs Workload Imbalance')
            plt.grid(True)

            plt.subplot(133)
            plt.scatter(pareto_fitness[:, 1], pareto_fitness[:, 2], c='green')
            plt.xlabel('Preference Deviation')
            plt.ylabel('Workload Imbalance')
            plt.title('Preference Deviation vs Workload Imbalance')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        for i, ind in enumerate(pareto_front[:3]):
            plt.figure(figsize=(12, 6))
            solution_matrix = np.zeros((num_jobs, num_shifts))
            for job_idx, shift_idx in enumerate(ind):
                solution_matrix[job_idx, shift_idx] = 1
            sns.heatmap(solution_matrix, cmap="Blues", cbar=False,
                        xticklabels=[f"Shift {i}" for i in range(num_shifts)],
                        yticklabels=[f"Job {i}" for i in range(num_jobs)])
            plt.title(f'Solution {i+1} - Fitness: {ind.fitness.values}')
            plt.ylabel('Jobs')
            plt.xlabel('Shifts')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()
