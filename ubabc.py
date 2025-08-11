import os
import datetime
import matplotlib.pyplot as plt
from typing import List, Tuple
import task3edit


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def make_dir(out_folder):
    os.makedirs(out_folder, exist_ok=True)


def plot_histories(best_history, avg_history, active_history, title_prefix="UbABC"):
    figs = []
    # 1) Best & Avg fitness curves
    fig1, ax1 = plt.subplots(figsize=(7,4))
    if best_history:
        ax1.plot(best_history, label="Best")
    if avg_history:
        ax1.plot(avg_history, label="Average", linestyle="--")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fitness")
    ax1.set_title(f"{title_prefix} - Fitness over Iterations")
    ax1.grid(True)
    ax1.legend()
    figs.append(fig1)

    # 2) Active PMs over iterations
    fig2, ax2 = plt.subplots(figsize=(6,3))
    if active_history:
        ax2.plot(active_history, marker='o')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Active PMs")
    ax2.set_title(f"{title_prefix} - Active PMs over Iterations")
    ax2.grid(True)
    figs.append(fig2)

    return figs


def make_vm_distribution_figure(best_solution, pm_list, vm_list, title_prefix="UbABC"):
    pm_to_vms = {pm.id: [] for pm in pm_list}
    for vm_id, pm_id in enumerate(best_solution.assignment):
        if pm_id != -1:
            pm_to_vms.setdefault(pm_id, []).append(vm_id)

    table_data = []
    pm_ids = sorted(pm_to_vms.keys())
    for pm_id in pm_ids:
        vms = pm_to_vms[pm_id]
        vm_str = ", ".join(str(x) for x in vms) if vms else "-"
        table_data.append([f"PM{pm_id}", len(vms), vm_str])

    fig, ax = plt.subplots(figsize=(8, max(2, 0.5*len(table_data)+1)))
    ax.axis('off')
    table = ax.table(cellText=table_data,
                    colLabels=["PM", "VM count", "VM IDs"],
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    ax.set_title(f"{title_prefix} - VM distribution (final)")
    return fig


def calculate_resource_utilization(pm_list, vm_list, best_solution):
    pm_cpu_used = {pm.id: 0.0 for pm in pm_list}
    pm_mem_used = {pm.id: 0.0 for pm in pm_list}

    for vm_id, pm_id in enumerate(best_solution.assignment):
        if pm_id != -1:
            pm_cpu_used[pm_id] += vm_list[vm_id].cpu_req
            pm_mem_used[pm_id] += vm_list[vm_id].memory_req

    pm_cpu_util = []
    pm_mem_util = []
    for pm in pm_list:
        cpu_cap = pm.resources.cpu
        mem_cap = pm.resources.memory
        cpu_util = pm_cpu_used[pm.id] / cpu_cap if cpu_cap else 0
        mem_util = pm_mem_used[pm.id] / mem_cap if mem_cap else 0
        pm_cpu_util.append(cpu_util)
        pm_mem_util.append(mem_util)

    return pm_cpu_util, pm_mem_util



def plot_resource_utilization(cpu_util, mem_util, title_prefix="UbABC"):
    fig, ax = plt.subplots(figsize=(8,4))
    indices = range(len(cpu_util))
    width = 0.35
    ax.bar(indices, cpu_util, width, label="CPU Utilization")
    ax.bar([i + width for i in indices], mem_util, width, label="Memory Utilization")
    ax.set_xlabel("Physical Machine")
    ax.set_ylabel("Utilization")
    ax.set_title(f"{title_prefix} - Resource Utilization per PM")
    ax.legend()
    ax.grid(True)
    return fig


def run_ubabc(pm_list: List[task3edit.PhysicalMachine],
              vm_list: List[task3edit.VirtualMachine],
              num_employed=10, num_onlooker=10, max_iterations=100, limit=20,
              out_folder: str = "output/UbABC") -> Tuple[task3edit.Solution, dict, List[str]]:
    """
    Runs UbABC (wrapper around task3edit.UbABC) and returns:
      - best_solution
      - results dict (best_history, avg_history, active_pms_history)
      - list of saved figure paths
    """
    make_dir(out_folder)
    ub = task3edit.UbABC(num_employed_bees=num_employed,
                         num_onlooker_bees=num_onlooker,
                         max_iterations=max_iterations,
                         limit=limit)
    best = ub.optimized_vm_placement(pm_list, vm_list)
    results = {
        "best_solution": best,
        "best_history": ub.best_fitness_history,
        "avg_history": ub.avg_fitness_history,
        "active_pms_history": ub.active_pms_history
    }

    figs = plot_histories(results["best_history"], results["avg_history"], results["active_pms_history"], title_prefix="UbABC (proposed)")
    figs.append(make_vm_distribution_figure(best, pm_list, vm_list, title_prefix="UbABC (proposed)"))

    # Calculate and plot resource utilization
    pm_cpu_util, pm_mem_util = calculate_resource_utilization(pm_list, vm_list, best)
    fig_util = plot_resource_utilization(pm_cpu_util, pm_mem_util, title_prefix="UbABC (proposed)")
    figs.append(fig_util)

    # Save figs
    ts = timestamp()
    saved = []
    for i, fig in enumerate(figs, start=1):
        fname = os.path.join(out_folder, f"UbABC_{ts}_plot{i}.png")
        fig.tight_layout()
        fig.savefig(fname)
        saved.append(fname)
        plt.close(fig)
    results['saved_pngs'] = saved
    return best, results, saved
