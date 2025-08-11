# abc_algorithm.py
import os
import datetime
import matplotlib.pyplot as plt
from typing import List, Tuple
import task3edit

def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure(out_folder):
    os.makedirs(out_folder, exist_ok=True)

def _make_vm_table_fig(solution: task3edit.Solution, pm_list: List[task3edit.PhysicalMachine], vm_list: List[task3edit.VirtualMachine], title="ABC - VM distribution"):
    pm_to_vms = {pm.id: [] for pm in pm_list}
    for vm_id, pm_id in enumerate(solution.assignment):
        if pm_id != -1:
            pm_to_vms.setdefault(pm_id, []).append(vm_id)
    rows = []
    for pm_id in sorted(pm_to_vms.keys()):
        vms = pm_to_vms[pm_id]
        rows.append([f"PM{pm_id}", str(len(vms)), ", ".join(map(str,vms)) if vms else "-"])
    fig, ax = plt.subplots(figsize=(8, max(2, 0.4*len(rows)+1)))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=["PM","VM count","VM IDs"], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1,1.2)
    ax.set_title(title)
    return fig

def _make_resource_bar_fig(solution: task3edit.Solution, pm_list: List[task3edit.PhysicalMachine], title="Resource Utilization"):
    cpu_utils = []
    mem_utils = []
    labels = []
    for i, res in enumerate(solution.resource_usage):
        if res.vms:
            total_cpu_used = res.cpu - res.available_cpu
            total_mem_used = res.memory - res.available_memory
            cpu_utils.append(total_cpu_used / res.cpu)
            mem_utils.append(total_mem_used / res.memory)
            labels.append(f"PM{i}")
    fig, ax = plt.subplots(figsize=(8,4))
    if cpu_utils:
        idx = range(len(cpu_utils))
        width = 0.35
        ax.bar(idx, cpu_utils, width, label="CPU util")
        ax.bar([i+width for i in idx], mem_utils, width, label="Mem util")
        ax.set_xticks([i+width/2 for i in idx])
        ax.set_xticklabels(labels)
        ax.set_ylim(0,1.05)
        ax.set_ylabel("Utilization (fraction)")
        ax.legend()
    else:
        ax.text(0.5,0.5,"No active PMs", ha='center', va='center')
        ax.axis('off')
    ax.set_title(title)
    return fig

def run_abc(pm_list: List[task3edit.PhysicalMachine], vm_list: List[task3edit.VirtualMachine],
            num_employed=10, num_onlooker=10, max_iterations=100, limit=20,
            out_folder: str = "output/ABC") -> Tuple[task3edit.Solution, dict, list]:
    """
    wrapper for baseline ABC using task3edit.optimized_vm_placement_baseline (run_abc already in task3edit).
    returns: best_solution, results_dict, list_of_saved_png_paths
    """
    _ensure(out_folder)
    # Use run_abc helper already defined in task3edit if present (it returns best + results)
    # If not, we call UbABC.optimized_vm_placement_baseline via a UbABC instance
    try:
        # task3edit.run_abc exists in your file — use it
        best, results = task3edit.run_abc(pm_list, vm_list, num_employed, num_onlooker, max_iterations, limit)
        # note: task3edit.run_abc returns best and results dict
    except Exception:
        # fallback: construct UbABC and call method
        abc_inst = task3edit.UbABC(num_employed, num_onlooker, max_iterations, limit)
        best = abc_inst.optimized_vm_placement_baseline(pm_list, vm_list)
        results = {
            "best_history": getattr(best, "_baseline_best_history", []),
            "avg_history": getattr(best, "_baseline_avg_history", []),
            "active_pms_history": getattr(best, "_baseline_active_pms", [])
        }

    # Create figures
    figs = []
    # fitness curve
    fig1, ax1 = plt.subplots(figsize=(7,4))
    if results.get("best_history"):
        ax1.plot(results["best_history"], label="best")
    if results.get("avg_history"):
        ax1.plot(results["avg_history"], label="avg", linestyle="--")
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Fitness"); ax1.set_title("ABC - Fitness over Iterations")
    ax1.grid(True); ax1.legend()
    figs.append(fig1)

    # active pm fig
    fig2, ax2 = plt.subplots(figsize=(6,3))
    if results.get("active_pms_history"):
        ax2.plot(results["active_pms_history"], marker='o')
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Active PMs"); ax2.set_title("ABC - Active PMs")
    ax2.grid(True)
    figs.append(fig2)

    # vm distribution table + resource bar (use best from results)
    if isinstance(best, task3edit.Solution):
        fig3 = _make_vm_table_fig(best, pm_list, vm_list, title="ABC - VM distribution")
        figs.append(fig3)
        if getattr(best, "resource_usage", None):
            fig4 = _make_resource_bar_fig(best, pm_list, title="ABC - Final Resource Utilization")
            figs.append(fig4)

    # save figs
    ts = _timestamp()
    saved = []
    for i, fig in enumerate(figs, start=1):
        fname = os.path.join(out_folder, f"ABC_{ts}_plot{i}.png")
        fig.tight_layout(); fig.savefig(fname); plt.close(fig)
        saved.append(fname)
    results['saved_pngs'] = saved
    return best, results, saved
