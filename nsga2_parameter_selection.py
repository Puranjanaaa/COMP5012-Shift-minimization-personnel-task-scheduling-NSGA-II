import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import csv
import pandas as pd
from sklearn.manifold import MDS
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.ticker as ticker

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

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

# Function to perform parameter tuning
def parameter_tuning(extract_dir, num_trials=100):
    random.seed(21)
    dat_files = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.dat')]
    
    # Define parameter ranges for experimentation
    # Define expanded parameter ranges for experimentation
    population_sizes = [50, 100, 150, 200, 250, 300,350, 400]
    generations = [50, 100, 150, 200, 250,300,350, 400]
    crossover_probs = [0.4,0.5, 0.6, 0.7, 0.8, 0.9]
    mutation_probs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    
    best_params = None
    best_pareto_size = 0
    trial_results = []
    
    # For storing all pareto fronts and solutions for later visualization
    all_pareto_fronts = []

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        # Randomly select parameters for this trial
        pop_size = random.choice(population_sizes)
        ngen = random.choice(generations)
        cxpb = random.choice(crossover_probs)
        mutpb = random.choice(mutation_probs)
        
        print(f"Parameters: Population={pop_size}, Generations={ngen}, Crossover={cxpb}, Mutation={mutpb}")
        
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
            
            population = toolbox.population(n=pop_size)
            
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("min", np.min, axis=0)
            stats.register("avg", np.mean, axis=0)
            stats.register("max", np.max, axis=0)
            
            logbook = tools.Logbook()
            
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # Store generation-by-generation data for convergence plotting
            gen_data = []
            
            for gen in range(ngen):
                if gen % 10 == 0:
                    print(f"Generation {gen}/{ngen}")
                offspring = toolbox.select(population, len(population))
                offspring = list(map(toolbox.clone, offspring))
                
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < cxpb:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values, child2.fitness.values
                
                for mutant in offspring:
                    if random.random() < mutpb:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                population[:] = offspring
                record = stats.compile(population)
                logbook.record(gen=gen, **record)
                
                # Extract and store the current Pareto front for convergence analysis
                current_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
                current_fitness = [ind.fitness.values for ind in current_front]
                gen_data.append({
                    'generation': gen,
                    'front_size': len(current_front),
                    'front_fitness': current_fitness
                })
            
            pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
            pareto_size = len(pareto_front)
            
            # Store this pareto front for later visualization
            all_pareto_fronts.append({
                'trial': trial + 1,
                'population': pop_size,
                'generations': ngen,
                'crossover': cxpb,
                'mutation': mutpb,
                'pareto_front': pareto_front,
                'pareto_fitness': [ind.fitness.values for ind in pareto_front],
                'gen_data': gen_data
            })
            
            # Save trial results
            trial_results.append({
                "Trial": trial + 1,
                "Population": pop_size,
                "Generations": ngen,
                "Crossover": cxpb,
                "Mutation": mutpb,
                "Pareto Size": pareto_size
            })
            
            # Update the best parameters if this trial is better
            if pareto_size > best_pareto_size:
                best_pareto_size = pareto_size
                best_params = {
                    "Population": pop_size,
                    "Generations": ngen,
                    "Crossover": cxpb,
                    "Mutation": mutpb,
                    "Pareto Size": pareto_size
                }
    
    # Save all trial results to a CSV file
    csv_file = os.path.join("results", "trial_results.csv")
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Trial", "Population", "Generations", "Crossover", "Mutation", "Pareto Size"])
        writer.writeheader()
        writer.writerows(trial_results)
    
    print(f"\nTrial results saved to {csv_file}")
    print("\nBest Parameters:")
    print(best_params)
    
    return best_params, all_pareto_fronts

# Function to visualize parameter tuning results
def visualize_parameter_results(best_params, trial_results):
    # Convert trial results to a DataFrame
    df = pd.DataFrame(trial_results)
    
    # Plot Pareto Size vs Trials
    plt.figure(figsize=(10, 6))
    plt.plot(df["Trial"], df["Pareto Size"], marker="o", linestyle="-", label="Pareto Size")
    plt.xlabel("Trial")
    plt.ylabel("Pareto Size")
    plt.title("Pareto Size Across Trials")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join("results", "pareto_size_trials.png"))
    plt.close()
    
    # Plot distribution of Pareto Size by parameter value
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Population size vs Pareto Size
    sns.boxplot(x="Population", y="Pareto Size", data=df, ax=axes[0, 0])
    axes[0, 0].set_title("Population Size vs Pareto Size")
    axes[0, 0].grid(True)
    
    # Generations vs Pareto Size
    sns.boxplot(x="Generations", y="Pareto Size", data=df, ax=axes[0, 1])
    axes[0, 1].set_title("Generations vs Pareto Size")
    axes[0, 1].grid(True)
    
    # Crossover probability vs Pareto Size
    sns.boxplot(x="Crossover", y="Pareto Size", data=df, ax=axes[1, 0])
    axes[1, 0].set_title("Crossover Probability vs Pareto Size")
    axes[1, 0].grid(True)
    
    # Mutation probability vs Pareto Size
    sns.boxplot(x="Mutation", y="Pareto Size", data=df, ax=axes[1, 1])
    axes[1, 1].set_title("Mutation Probability vs Pareto Size")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join("results", "parameter_distributions.png"))
    plt.close()
    
    # Highlight the best parameters
    print("\nBest Parameters Visualization:")
    print(best_params)

# Function to generate decision making plots for MOGA NSGA-II
def generate_decision_making_plots(trial_results, all_pareto_fronts, best_params):
    print("\nGenerating decision making plots...")
    
    # 1. Parameter Space Visualization
    parameter_space_visualization(trial_results)
    
    # 2. Parameter Influence Analysis
    parameter_influence_analysis(trial_results)
    
    # 3. Convergence Analysis
    convergence_analysis(all_pareto_fronts)
    
    # 4. Pareto Front Visualization (Static and Interactive)
    best_fronts = [f for f in all_pareto_fronts if 
                   f['population'] == best_params['Population'] and 
                   f['generations'] == best_params['Generations'] and 
                   f['crossover'] == best_params['Crossover'] and 
                   f['mutation'] == best_params['Mutation']]
    
    if best_fronts:
        visualize_best_pareto_fronts(best_fronts)
    
    # 5. Trade-off Analysis
    if all_pareto_fronts:
        largest_front = max(all_pareto_fronts, key=lambda x: len(x['pareto_front']))
        tradeoff_analysis(largest_front['pareto_fitness'])
    
    # 6. Correlation Analysis
    correlation_analysis(trial_results)
    
    # 7. Decision Support Visualization
    if all_pareto_fronts:
        decision_support_visualization(all_pareto_fronts)
    
    print("Decision making plots generated and saved to the 'results' folder.")

# 1. Parameter Space Visualization
def parameter_space_visualization(trial_results):
    df = pd.DataFrame(trial_results)
    
    # Create a 3D scatter plot of the parameter space
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(df['Population'], df['Generations'], df['Crossover'],
               c=df['Pareto Size'], cmap='viridis', s=70, alpha=0.7)
    
    ax.set_xlabel('Population Size')
    ax.set_ylabel('Number of Generations')
    ax.set_zlabel('Crossover Rate')
    ax.set_title('Parameter Space Visualization (color = Pareto Size)')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Pareto Front Size')
    
    plt.tight_layout()
    plt.savefig(os.path.join("results", "parameter_space_3d.png"))
    plt.close()
    
    # Create a parallel coordinates plot for parameter relationships
    plt.figure(figsize=(12, 7))
    pd.plotting.parallel_coordinates(df, 'Population', colormap='viridis')
    plt.title('Parallel Coordinates Plot of Parameters')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "parameter_parallel_coordinates.png"))
    plt.close()

# 2. Parameter Influence Analysis
def parameter_influence_analysis(trial_results):
    df = pd.DataFrame(trial_results)
    
    # Create a correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Parameter Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "parameter_correlation_heatmap.png"))
    plt.close()
    
    # Create regression plots for each parameter vs Pareto Size
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    params = ['Population', 'Generations', 'Crossover', 'Mutation']
    for i, param in enumerate(params):
        sns.regplot(x=param, y='Pareto Size', data=df, ax=axes[i], 
                  scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
        axes[i].set_title(f'Pareto Size vs {param}')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join("results", "parameter_influence_regression.png"))
    plt.close()
    
    # Calculate Spearman rank correlation
    spearman_corr = df[params + ['Pareto Size']].corr(method='spearman')
    
    # Create a bar plot for parameter importance
    plt.figure(figsize=(10, 6))
    param_importance = abs(spearman_corr['Pareto Size'][:-1])
    param_importance.sort_values(ascending=False).plot(kind='bar', color='teal')
    plt.title('Parameter Importance for Pareto Size')
    plt.ylabel('|Spearman Correlation|')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "parameter_importance.png"))
    plt.close()

# 3. Convergence Analysis
def convergence_analysis(all_pareto_fronts):
    # Plot convergence of Pareto front size across generations
    plt.figure(figsize=(12, 7))
    
    # Select a few representative trials to avoid overcrowding
    if len(all_pareto_fronts) > 5:
        selected_fronts = random.sample(all_pareto_fronts, 5)
    else:
        selected_fronts = all_pareto_fronts
    
    for i, front_data in enumerate(selected_fronts):
        # Extract generation data
        generations = [d['generation'] for d in front_data['gen_data']]
        front_sizes = [d['front_size'] for d in front_data['gen_data']]
        
        # Plot convergence curve
        plt.plot(generations, front_sizes, 
                label=f"Pop={front_data['population']}, Gen={front_data['generations']}, CR={front_data['crossover']:.1f}, MR={front_data['mutation']:.1f}")
    
    plt.xlabel('Generation')
    plt.ylabel('Pareto Front Size')
    plt.title('Convergence of Pareto Front Size')
    plt.legend(title='Parameters', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "convergence_analysis.png"))
    plt.close()
    
    # Plot hypervolume convergence if there are objective values (simulated)
    plt.figure(figsize=(12, 7))
    
    for i, front_data in enumerate(selected_fronts):
        # Simulate hypervolume trend - in a real case, use actual hypervolume calculations
        generations = [d['generation'] for d in front_data['gen_data']]
        max_gen = max(generations)
        # Simulate hypervolume data with logistic growth curve
        hv_values = [1.0 - (1.0 / (1 + np.exp(-10 * (g/max_gen - 0.5)))) for g in generations]
        
        # Add some noise unique to this trial
        hv_values = [hv + 0.05 * np.sin(g * (i+1) * 0.1) for g, hv in zip(generations, hv_values)]
        
        plt.plot(generations, hv_values, 
                label=f"Pop={front_data['population']}, Gen={front_data['generations']}, CR={front_data['crossover']:.1f}, MR={front_data['mutation']:.1f}")
    
    plt.xlabel('Generation')
    plt.ylabel('Normalized Hypervolume (simulated)')
    plt.title('Convergence of Hypervolume (Simulated)')
    plt.legend(title='Parameters', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "hypervolume_convergence.png"))
    plt.close()

# 4. Pareto Front Visualization
def visualize_best_pareto_fronts(best_fronts):
    # Select the front with the largest number of solutions
    best_front = max(best_fronts, key=lambda x: len(x['pareto_front']))
    pareto_fitness = np.array(best_front['pareto_fitness'])
    
    if len(pareto_fitness) > 0 and pareto_fitness.shape[1] >= 3:
        # Static 3D Pareto front visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            pareto_fitness[:, 0], pareto_fitness[:, 1], pareto_fitness[:, 2], 
            c=pareto_fitness[:, 0], cmap='viridis', s=100, alpha=0.7
        )
        
        ax.set_xlabel('Shifts Used')
        ax.set_ylabel('Preference Deviation')
        ax.set_zlabel('Workload Imbalance')
        ax.set_title('3D Pareto Front for Best Parameters')
        
        # Add a color bar to explain the color mapping
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Shifts Used', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join("results", "best_pareto_front_3d.png"))
        plt.close()
        
        # Create 2D projections with convex hull
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        objective_pairs = [(0, 1), (0, 2), (1, 2)]
        titles = ['Shifts vs Preference', 'Shifts vs Workload', 'Preference vs Workload']
        
        for i, (obj1, obj2) in enumerate(objective_pairs):
            # Plot points
            axes[i].scatter(pareto_fitness[:, obj1], pareto_fitness[:, obj2], c='blue', s=50, alpha=0.7)
            
            # Create convex hull if enough points
            if len(pareto_fitness) >= 3:
                try:
                    points = pareto_fitness[:, [obj1, obj2]]
                    hull = ConvexHull(points)
                    hull_vertices = np.append(hull.vertices, hull.vertices[0])  # Close the polygon
                    axes[i].plot(points[hull_vertices, 0], points[hull_vertices, 1], 'r--', lw=2)
                except:
                    # If convex hull fails, just skip it
                    pass
            
            axes[i].set_xlabel(f'Objective {obj1+1}')
            axes[i].set_ylabel(f'Objective {obj2+1}')
            axes[i].set_title(titles[i])
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("results", "best_pareto_projections.png"))
        plt.close()
        
        # Create interactive 3D visualization with Plotly
        try:
            fig = go.Figure(data=[go.Scatter3d(
                x=pareto_fitness[:, 0],
                y=pareto_fitness[:, 1],
                z=pareto_fitness[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=pareto_fitness[:, 0],  # Color by shift count
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=[f"Solution {i+1}<br>Shifts: {f[0]:.2f}<br>Preference: {f[1]:.2f}<br>Workload: {f[2]:.2f}" 
                      for i, f in enumerate(pareto_fitness)]
            )])
            
            fig.update_layout(
                title="Interactive 3D Pareto Front",
                scene=dict(
                    xaxis_title='Shifts Used',
                    yaxis_title='Preference Deviation',
                    zaxis_title='Workload Imbalance'
                ),
                width=900,
                height=700
            )
            
            fig.write_html(os.path.join("results", "interactive_pareto_front.html"))
        except:
            print("Warning: Failed to create interactive Plotly visualization")

# 5. Trade-off Analysis
def tradeoff_analysis(pareto_fitness):
    pareto_fitness = np.array(pareto_fitness)
    
    if len(pareto_fitness) > 0 and pareto_fitness.shape[1] >= 3:
        # Calculate pairewise trade-offs between objectives
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        objective_pairs = [(0, 1), (0, 2), (1, 2)]
        titles = ['Shifts vs Preference', 'Shifts vs Workload', 'Preference vs Workload']
        
        for i, (obj1, obj2) in enumerate(objective_pairs):
            x = pareto_fitness[:, obj1]
            y = pareto_fitness[:, obj2]
            
            # Calculate trade-off ratio (slope of connecting lines)
            trade_offs = []
            for j in range(len(x)):
                for k in range(j+1, len(x)):
                    if abs(x[j] - x[k]) > 1e-6:  # Avoid division by zero
                        trade_off = (y[j] - y[k]) / (x[j] - x[k])
                        trade_offs.append((trade_off, (x[j], y[j]), (x[k], y[k])))
            
            # Plot points
            axes[i].scatter(x, y, c='blue', s=50, alpha=0.7)
            
            # Plot trade-off lines for a selection of extremes
            if trade_offs:
                # Sort by absolute trade-off value
                trade_offs.sort(key=lambda t: abs(t[0]))
                
                # Select some extreme trade-offs to highlight
                highlights = [trade_offs[0]]  # Smallest absolute trade-off
                if len(trade_offs) > 1:
                    highlights.append(trade_offs[-1])  # Largest absolute trade-off
                
                # Add some middle values if available
                if len(trade_offs) >= 5:
                    highlights.append(trade_offs[len(trade_offs)//2])  # Median trade-off
                
                # Plot the highlighted trade-offs
                for to, p1, p2 in highlights:
                    axes[i].plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', alpha=0.7)
                    
                    # Add trade-off value text at midpoint
                    mid_x = (p1[0] + p2[0]) / 2
                    mid_y = (p1[1] + p2[1]) / 2
                    axes[i].annotate(f"{to:.2f}", (mid_x, mid_y), 
                                   fontsize=8, color='black',
                                   bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
            
            axes[i].set_xlabel(f'Objective {obj1+1}')
            axes[i].set_ylabel(f'Objective {obj2+1}')
            axes[i].set_title(titles[i])
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join("results", "trade_off_analysis.png"))
        plt.close()
        
        # Create a radar chart to compare selected solutions
        if len(pareto_fitness) >= 3:
            # Select a few representative solutions (e.g., best in each objective)
            best_obj1 = pareto_fitness[np.argmin(pareto_fitness[:, 0])]
            best_obj2 = pareto_fitness[np.argmin(pareto_fitness[:, 1])]
            best_obj3 = pareto_fitness[np.argmin(pareto_fitness[:, 2])]
            
            # Normalize objectives to [0,1] for radar chart
            # For minimization, smaller is better, so invert the normalization
            min_vals = np.min(pareto_fitness, axis=0)
            max_vals = np.max(pareto_fitness, axis=0)
            range_vals = max_vals - min_vals
            
            # Prevent division by zero
            range_vals[range_vals == 0] = 1.0
            
            # Normalize and invert (1 - norm) so higher is better on radar chart
            norm_best_obj1 = 1 - (best_obj1 - min_vals) / range_vals
            norm_best_obj2 = 1 - (best_obj2 - min_vals) / range_vals
            norm_best_obj3 = 1 - (best_obj3 - min_vals) / range_vals

            # Combine normalized values for radar chart
            radar_data = np.vstack([norm_best_obj1, norm_best_obj2, norm_best_obj3])
            labels = ['Objective 1 (Shifts Used)', 'Objective 2 (Preference Deviation)', 'Objective 3 (Workload Imbalance)']

            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            radar_data = np.concatenate((radar_data, radar_data[:, :1]), axis=1)  # Close the radar chart
            angles += angles[:1]  # Close the radar chart

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            for i, data in enumerate(radar_data):
                ax.plot(angles, data, label=f'Solution {i+1}', linewidth=2)
                ax.fill(angles, data, alpha=0.25)

            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_title('Radar Chart of Selected Pareto Solutions', size=16, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.tight_layout()
            plt.savefig(os.path.join("results", "radar_chart.png"))
            plt.close()

# 6. Correlation Analysis
def correlation_analysis(trial_results):
    # Convert trial results to a DataFrame
    df = pd.DataFrame(trial_results)
    
    # Create a correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Parameter Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "parameter_correlation_heatmap.png"))
    plt.close()
    
    print("Correlation analysis completed. Heatmap saved to 'results/parameter_correlation_heatmap.png'.")

# 7. Decision Support Visualization
def decision_support_visualization(all_pareto_fronts):
    print("\nGenerating decision support visualization...")

    # Select the largest Pareto front for visualization
    largest_front = max(all_pareto_fronts, key=lambda x: len(x['pareto_front']))
    pareto_fitness = np.array(largest_front['pareto_fitness'])

    if len(pareto_fitness) > 0 and pareto_fitness.shape[1] >= 3:
        # Normalize the objectives for visualization
        min_vals = np.min(pareto_fitness, axis=0)
        max_vals = np.max(pareto_fitness, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0  # Prevent division by zero
        normalized_fitness = (pareto_fitness - min_vals) / range_vals

        # Add a dummy class column for parallel coordinates
        df = pd.DataFrame(normalized_fitness, columns=['Shifts Used', 'Preference Deviation', 'Workload Imbalance'])
        df['Class'] = 'Pareto Solution'  # Add a dummy class column

        # Create a parallel coordinates plot
        plt.figure(figsize=(12, 8))
        pd.plotting.parallel_coordinates(df, class_column='Class', colormap='viridis')
        plt.title('Parallel Coordinates Plot of Pareto Solutions')
        plt.xlabel('Objectives')
        plt.ylabel('Normalized Values')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join("results", "decision_support_parallel_coordinates.png"))
        plt.close()

        print("Decision support visualization completed. Plot saved to 'results/decision_support_parallel_coordinates.png'.")

# Function to perform parameter tuning
def parameter_tuning(extract_dir, num_trials=100):
    random.seed(21)
    dat_files = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.dat')]
    
    # Define parameter ranges for experimentation
    # Define expanded parameter ranges for experimentation
    population_sizes = [50, 100, 150, 200, 250, 300]
    generations = [50, 100, 150, 200, 250]
    crossover_probs = [0.5, 0.6, 0.7, 0.8, 0.9]
    mutation_probs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    best_params = None
    best_pareto_size = 0
    trial_results = []  # Store all trial results here
    
    # For storing all pareto fronts and solutions for later visualization
    all_pareto_fronts = []

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        # Randomly select parameters for this trial
        pop_size = random.choice(population_sizes)
        ngen = random.choice(generations)
        cxpb = random.choice(crossover_probs)
        mutpb = random.choice(mutation_probs)
        
        print(f"Parameters: Population={pop_size}, Generations={ngen}, Crossover={cxpb}, Mutation={mutpb}")
        
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
            
            population = toolbox.population(n=pop_size)
            
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("min", np.min, axis=0)
            stats.register("avg", np.mean, axis=0)
            stats.register("max", np.max, axis=0)
            
            logbook = tools.Logbook()
            
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # Store generation-by-generation data for convergence plotting
            gen_data = []
            
            for gen in range(ngen):
                if gen % 10 == 0:
                    print(f"Generation {gen}/{ngen}")
                offspring = toolbox.select(population, len(population))
                offspring = list(map(toolbox.clone, offspring))
                
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < cxpb:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values, child2.fitness.values
                
                for mutant in offspring:
                    if random.random() < mutpb:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                population[:] = offspring
                record = stats.compile(population)
                logbook.record(gen=gen, **record)
                
                # Extract and store the current Pareto front for convergence analysis
                current_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
                current_fitness = [ind.fitness.values for ind in current_front]
                gen_data.append({
                    'generation': gen,
                    'front_size': len(current_front),
                    'front_fitness': current_fitness
                })
            
            pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
            pareto_size = len(pareto_front)
            
            # Store this pareto front for later visualization
            all_pareto_fronts.append({
                'trial': trial + 1,
                'population': pop_size,
                'generations': ngen,
                'crossover': cxpb,
                'mutation': mutpb,
                'pareto_front': pareto_front,
                'pareto_fitness': [ind.fitness.values for ind in pareto_front],
                'gen_data': gen_data
            })
            
            # Save trial results
            trial_results.append({
                "Trial": trial + 1,
                "Population": pop_size,
                "Generations": ngen,
                "Crossover": cxpb,
                "Mutation": mutpb,
                "Pareto Size": pareto_size
            })
            
            # Update the best parameters if this trial is better
            if pareto_size > best_pareto_size:
                best_pareto_size = pareto_size
                best_params = {
                    "Population": pop_size,
                    "Generations": ngen,
                    "Crossover": cxpb,
                    "Mutation": mutpb,
                    "Pareto Size": pareto_size
                }
    
    # Save all trial results to a CSV file
    csv_file = os.path.join("results", "trial_results.csv")
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Trial", "Population", "Generations", "Crossover", "Mutation", "Pareto Size"])
        writer.writeheader()
        writer.writerows(trial_results)
    
    print(f"\nTrial results saved to {csv_file}")
    print("\nBest Parameters:")
    print(best_params)
    
    return best_params, trial_results, all_pareto_fronts  # Return trial_results as well


def run_moga_with_best_params(extract_dir, best_params):
    random.seed(21)
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
        
        # Use best parameters from tuning
        population = toolbox.population(n=best_params["Population"])
        NGEN, CXPB, MUTPB = best_params["Generations"], best_params["Crossover"], best_params["Mutation"]
        
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
        
        if len(pareto_front) > 0:
            # Save the Pareto front solutions
            output_file = f"pareto_solutions_{os.path.basename(dat_file)}.csv"
            with open(output_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Solution", "Shifts Used", "Preference Deviation", "Workload Imbalance", "Assignment"])
                for i, ind in enumerate(pareto_front):
                    writer.writerow([i+1, ind.fitness.values[0], ind.fitness.values[1], ind.fitness.values[2], ind])
            
            print(f"Pareto solutions saved to {output_file}")
            
    
    return population, logbook

# Function to visualize the evolution of solutions using MDS
def visualize_solution_evolution(logbook, population, num_generations, output_file="solution_evolution.png"):
    print("\nVisualizing solution evolution using MDS...")
    
    # Collect all solutions across generations
    all_solutions = []
    for gen in range(num_generations):
        gen_solutions = [ind.fitness.values for ind in population]
        all_solutions.extend(gen_solutions)
    
    # Use landmark MDS to embed a smaller subset of solutions
    num_landmarks = min(500, len(all_solutions))  # Use up to 500 landmarks
    random_indices = random.sample(range(len(all_solutions)), num_landmarks)
    landmark_solutions = np.array([all_solutions[i] for i in random_indices])
    
    # Perform MDS embedding
    mds = MDS(n_components=2, random_state=42, dissimilarity="euclidean")
    embedded_landmarks = mds.fit_transform(landmark_solutions)
    
    # Map the rest of the solutions to the embedded space
    embedded_solutions = np.zeros((len(all_solutions), 2))
    for i, solution in enumerate(all_solutions):
        if i in random_indices:
            embedded_solutions[i] = embedded_landmarks[random_indices.index(i)]
        else:
            # Use the nearest landmark to approximate the embedding
            distances = np.linalg.norm(landmark_solutions - solution, axis=1)
            nearest_landmark = np.argmin(distances)
            embedded_solutions[i] = embedded_landmarks[nearest_landmark]
    
    # Plot the evolution of solutions
    plt.figure(figsize=(12, 8))
    for gen in range(num_generations):
        gen_solutions = embedded_solutions[gen * len(population):(gen + 1) * len(population)]
        plt.scatter(gen_solutions[:, 0], gen_solutions[:, 1], label=f"Generation {gen}", alpha=0.5, s=10)
    
    plt.title("Evolution of Solutions Over Generations (MDS Embedding)")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Solution evolution plot saved to {output_file}")


def main():
    # Set random seed for reproducibility
    random.seed(21)

    # Define the directory containing .dat files
    extract_dir = "data"
    os.makedirs("results", exist_ok=True)  # Ensure the results directory exists

    print("Starting parameter tuning...")
    num_trials = 100  # Number of trials for parameter tuning
    best_params, trial_results, all_pareto_fronts = parameter_tuning(extract_dir, num_trials)

    print("\nVisualizing parameter tuning results...")
    visualize_parameter_results(best_params, trial_results)

    print("\nGenerating decision-making plots...")
    generate_decision_making_plots(trial_results, all_pareto_fronts, best_params)

    print("\nProcess complete! Results saved in the 'results' directory.")

if __name__ == "__main__":
    main()