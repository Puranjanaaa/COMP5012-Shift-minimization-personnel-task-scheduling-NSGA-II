# 📊 COMP5012 – Shift Minimization & Personnel Task Scheduling using NSGA-II

This project applies a Multi-Objective Genetic Algorithm (NSGA-II) to solve the **multi-skilled shift scheduling problem**, targeting three competing objectives:  
1. Minimizing the total number of shifts  
2. Minimizing workload imbalance  
3. Minimizing deviation from worker preferences

---

## 🚀 How It Works

### 1. **Parameter Selection with NSGA-II**
Run the script `nsga2_parameter_selection.py` to perform **100 trials** of NSGA-II using different combinations of:
- Population sizes
- Number of generations
- Crossover probabilities
- Mutation rates

🔧 The script automatically evaluates these combinations to find the most effective parameter set.  
📈 **Results**: The best-performing parameters are printed to the terminal, and summary plots are saved in the `results/` folder for reference.

---

### 2. **Running the Final Optimization**
Use the script `final_pareto_solutions.py` to execute the MOGA using the best parameters obtained from the previous step.

🎯 This script generates:
- A 3D Pareto front showing the trade-offs between objectives  
- 2D plots for deeper analysis and visualization  

All outputs are saved in the `results/` folder.

---

## 📂 Input Data

All input data files should be placed in the `data/` directory.

- The code is designed to iterate over multiple `.dat` files if present.
- Make sure your problem instances are correctly formatted and stored in this folder before running the scripts.

---

## 📌 Summary

- ⚙️ Parameter tuning: `nsga2_parameter_selection.py`  
- 🧠 Final optimization: `final_pareto_solutions.py`  
- 📁 Data files go in: `data/`  
- 📊 Output and plots go to: `results/`

---
