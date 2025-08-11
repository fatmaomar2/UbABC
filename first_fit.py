# first_fit.py
import os
import datetime
import copy
import matplotlib.pyplot as plt
from typing import List, Tuple
import task3edit

def _ts(): return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def _ensure(folder): os.makedirs(folder, exist_ok=True)

def _compute_pm_usage(pm_resources, pm_list):
    """Return dict pm_id -> usage info"""
    usage = {}
    for i, res in enumerate(pm_resources):
        vm_count = len(res.vms)
        cpu_total = res.cpu
        mem_total = res.memory
        cpu_used = cpu_total - res.available_cpu
        mem_used = mem_total - res.available_memory
        cpu_util = (cpu_used / cpu_total) if cpu_total>0 else 0
        mem_util = (mem_used / mem_total) if mem_total>0 else 0
        usage[i] = {
            "vm_count": vm_count,
            "cpu_used": cpu_used,
            "mem_used": mem_used,
            "cpu_total": cpu_total,
            "mem_total": mem_total,
            "cpu_util": cpu_util,
            "mem_util": mem_util
        }
    return usage

def run_first_fit(pm_list: List[task3edit.PhysicalMachine],
                  vm_list: List[task3edit.VirtualMachine],
                  out_folder: str = "output/FirstFit") -> Tuple[task3edit.Solution, dict, list]:
    _ensure(out_folder)
    num_pms = len(pm_list)
    pm_resources = [copy.deepcopy(pm.resources) for pm in pm_list]
    assignment = [-1]*len(vm_list)

    # First-Fit: iterate VMs in given order, put in first PM with capacity
    for vm in vm_list:
        placed = False
        for i in range(num_pms):
            r = pm_resources[i]
            if r.available_cpu >= vm.cpu_req and r.available_memory >= vm.memory_req:
                assignment[vm.id] = i
                r.available_cpu -= vm.cpu_req
                r.available_memory -= vm.memory_req
                r.vms.append(vm.id)
                placed = True
                break
        if not placed:
            # leave -1 (unplaced) — you can also implement dropping or new-PM logic
            pass

    sol = task3edit.Solution(assignment, 0.0)
    sol.resource_usage = pm_resources
    sol.migration_count = 0

    # compute fitness via UbABC helper (weights default)
    ub = task3edit.UbABC(10,10,1,1)
    sol.fitness = ub.calculate_fitness(assignment, pm_resources, pm_list, vm_list, 0)

    pm_usage = _compute_pm_usage(pm_resources, pm_list)

    results = {
        "best_solution": sol,
        "best_history": [sol.fitness],   # single point (no iterations)
        "avg_history": [sol.fitness],
        "active_pms_history": [len([r for r in pm_resources if r.vms])],
        "pm_usage": pm_usage
    }

    # Make visuals: distribution table + resource bar
    figs = []
    # 1) VM distribution table
    rows = []
    for pid in range(len(pm_list)):
        vms = pm_resources[pid].vms
        rows.append([f"PM{pid}", str(len(vms)), ", ".join(map(str,vms)) if vms else "-"])
    fig, ax = plt.subplots(figsize=(8, max(2, 0.4*len(rows)+1)))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=["PM","VM count","VM IDs"], loc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,1.2)
    ax.set_title("FirstFit - VM distribution")
    figs.append(fig)

    # 2) resource utilization bar
    labels = []
    cpu_utils = []
    mem_utils = []
    for pid in sorted(pm_usage.keys()):
        u = pm_usage[pid]
        labels.append(f"PM{pid}")
        cpu_utils.append(u["cpu_util"])
        mem_utils.append(u["mem_util"])
    fig2, ax2 = plt.subplots(figsize=(8,4))
    if labels:
        idx = range(len(labels))
        width = 0.35
        ax2.bar(idx, cpu_utils, width, label="CPU util")
        ax2.bar([i+width for i in idx], mem_utils, width, label="Mem util")
        ax2.set_xticks([i+width/2 for i in idx]); ax2.set_xticklabels(labels)
        ax2.set_ylim(0,1.05); ax2.legend()
    else:
        ax2.text(0.5,0.5,"No PM data", ha='center', va='center'); ax2.axis('off')
    ax2.set_title("FirstFit - Resource Utilization")
    figs.append(fig2)

    # save figs
    ts = _ts()
    saved = []
    for i, fig in enumerate(figs, start=1):
        p = os.path.join(out_folder, f"FirstFit_{ts}_plot{i}.png")
        fig.tight_layout(); fig.savefig(p); plt.close(fig)
        saved.append(p)
    results['saved_pngs'] = saved
    return sol, results, saved
