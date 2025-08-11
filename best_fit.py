# best_fit.py
import os
import datetime
import copy
import matplotlib.pyplot as plt
from typing import List, Tuple
import task3edit

def _ts(): return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def _ensure(folder): os.makedirs(folder, exist_ok=True)

def _compute_pm_usage(pm_resources, pm_list):
    usage = {}
    for i, res in enumerate(pm_resources):
        vm_count = len(res.vms)
        cpu_total = res.cpu
        mem_total = res.memory
        cpu_used = cpu_total - res.available_cpu
        mem_used = mem_total - res.available_memory
        cpu_util = (cpu_used / cpu_total) if cpu_total>0 else 0
        mem_util = (mem_used / mem_total) if mem_total>0 else 0
        usage[i] = {"vm_count":vm_count,"cpu_used":cpu_used,"mem_used":mem_used,
                    "cpu_total":cpu_total,"mem_total":mem_total,"cpu_util":cpu_util,"mem_util":mem_util}
    return usage

def run_best_fit(pm_list: List[task3edit.PhysicalMachine],
                 vm_list: List[task3edit.VirtualMachine],
                 out_folder: str = "output/BestFit") -> Tuple[task3edit.Solution, dict, list]:
    _ensure(out_folder)
    pm_resources = [copy.deepcopy(pm.resources) for pm in pm_list]
    assignment = [-1]*len(vm_list)

    # Best-Fit: for each VM choose PM that leaves minimal leftover (cpu_left+mem_left)
    for vm in vm_list:
        best_pm = None
        best_score = None
        for i, r in enumerate(pm_resources):
            if r.available_cpu >= vm.cpu_req and r.available_memory >= vm.memory_req:
                cpu_left = (r.available_cpu - vm.cpu_req) / r.cpu if r.cpu>0 else 0
                mem_left = (r.available_memory - vm.memory_req) / r.memory if r.memory>0 else 0
                score = cpu_left + mem_left
                if best_score is None or score < best_score:
                    best_score = score
                    best_pm = i
        if best_pm is not None:
            assignment[vm.id] = best_pm
            pm_resources[best_pm].available_cpu -= vm.cpu_req
            pm_resources[best_pm].available_memory -= vm.memory_req
            pm_resources[best_pm].vms.append(vm.id)

    sol = task3edit.Solution(assignment, 0.0)
    sol.resource_usage = pm_resources
    sol.migration_count = 0

    ub = task3edit.UbABC(10,10,1,1)
    sol.fitness = ub.calculate_fitness(assignment, pm_resources, pm_list, vm_list, 0)

    pm_usage = _compute_pm_usage(pm_resources, pm_list)
    results = {
        "best_solution": sol,
        "best_history":[sol.fitness],
        "avg_history":[sol.fitness],
        "active_pms_history":[len([r for r in pm_resources if r.vms])],
        "pm_usage": pm_usage
    }

    # visuals: table + resource bar
    figs = []
    rows = []
    for pid in range(len(pm_list)):
        vms = pm_resources[pid].vms
        rows.append([f"PM{pid}", str(len(vms)), ", ".join(map(str,vms)) if vms else "-"])
    fig, ax = plt.subplots(figsize=(8, max(2, 0.4*len(rows)+1)))
    ax.axis('off'); table = ax.table(cellText=rows, colLabels=["PM","VM count","VM IDs"], loc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,1.2)
    ax.set_title("BestFit - VM distribution")
    figs.append(fig)

    labels = []; cpu_utils=[]; mem_utils=[]
    for pid in sorted(pm_usage.keys()):
        u = pm_usage[pid]
        labels.append(f"PM{pid}"); cpu_utils.append(u["cpu_util"]); mem_utils.append(u["mem_util"])
    fig2, ax2 = plt.subplots(figsize=(8,4))
    if labels:
        idx = range(len(labels)); width=0.35
        ax2.bar(idx, cpu_utils, width, label="CPU util")
        ax2.bar([i+width for i in idx], mem_utils, width, label="Mem util")
        ax2.set_xticks([i+width/2 for i in idx]); ax2.set_xticklabels(labels); ax2.set_ylim(0,1.05); ax2.legend()
    else:
        ax2.text(0.5,0.5,"No PM data", ha='center', va='center'); ax2.axis('off')
    ax2.set_title("BestFit - Resource Utilization")
    figs.append(fig2)

    ts = _ts(); saved=[]
    for i, fig in enumerate(figs, start=1):
        p = os.path.join(out_folder, f"BestFit_{ts}_plot{i}.png")
        fig.tight_layout(); fig.savefig(p); plt.close(fig)
        saved.append(p)
    results['saved_pngs'] = saved
    results['pm_usage'] = pm_usage
    return sol, results, saved
