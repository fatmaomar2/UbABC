# gui.py
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime
import task3edit

# import algorithm wrappers
from abc_algorithm import run_abc
from ubabc import run_ubabc
from first_fit import run_first_fit
from best_fit import run_best_fit

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

class VMPlacementApp:
    def __init__(self, root):
        self.root = root
        root.title("Multi-Algorithm VM Placement")
        root.geometry("1000x720")
        self.file_path = None

        # top controls
        ctrl = ttk.Frame(root, padding=8)
        ctrl.pack(fill="x")

        # data source
        self.source_var = tk.StringVar(value="random")
        ttk.Radiobutton(ctrl, text="Import file", variable=self.source_var, value="file").grid(row=0,column=0,sticky="w")
        ttk.Radiobutton(ctrl, text="Generate random", variable=self.source_var, value="random").grid(row=0,column=1,sticky="w")
        ttk.Button(ctrl, text="Choose file...", command=self.choose_file).grid(row=0,column=2,padx=6)

        ttk.Label(ctrl, text="#PMs:").grid(row=1,column=0,sticky="e")
        self.pms_entry = ttk.Entry(ctrl, width=6); self.pms_entry.insert(0,"5"); self.pms_entry.grid(row=1,column=1,sticky="w")
        ttk.Label(ctrl, text="#VMs:").grid(row=1,column=2,sticky="e")
        self.vms_entry = ttk.Entry(ctrl, width=6); self.vms_entry.insert(0,"8"); self.vms_entry.grid(row=1,column=3,sticky="w")

        # algorithm selection (checkboxes)
        self.alg_vars = {
            "ABC": tk.BooleanVar(value=False),
            "UbABC": tk.BooleanVar(value=True),
            "FirstFit": tk.BooleanVar(value=False),
            "BestFit": tk.BooleanVar(value=False)
        }
        alg_frame = ttk.LabelFrame(ctrl, text="Algorithms", padding=6)
        alg_frame.grid(row=2, column=0, columnspan=4, sticky="w", pady=6)
        ttk.Checkbutton(alg_frame, text="ABC (baseline)", variable=self.alg_vars["ABC"]).pack(side="left", padx=6)
        ttk.Checkbutton(alg_frame, text="UbABC (proposed)", variable=self.alg_vars["UbABC"]).pack(side="left", padx=6)
        ttk.Checkbutton(alg_frame, text="First Fit", variable=self.alg_vars["FirstFit"]).pack(side="left", padx=6)
        ttk.Checkbutton(alg_frame, text="Best Fit", variable=self.alg_vars["BestFit"]).pack(side="left", padx=6)

        # algorithm params
        param_frame = ttk.Frame(ctrl)
        param_frame.grid(row=3,column=0, columnspan=4, sticky="w")
        ttk.Label(param_frame, text="Employed:").grid(row=0,column=0); self.employed_entry = ttk.Entry(param_frame,width=6); self.employed_entry.insert(0,"10"); self.employed_entry.grid(row=0,column=1)
        ttk.Label(param_frame, text="Onlooker:").grid(row=0,column=2); self.onlooker_entry = ttk.Entry(param_frame,width=6); self.onlooker_entry.insert(0,"10"); self.onlooker_entry.grid(row=0,column=3)
        ttk.Label(param_frame, text="Iterations:").grid(row=1,column=0); self.iters_entry = ttk.Entry(param_frame,width=6); self.iters_entry.insert(0,"100"); self.iters_entry.grid(row=1,column=1)
        ttk.Label(param_frame, text="Limit:").grid(row=1,column=2); self.limit_entry = ttk.Entry(param_frame,width=6); self.limit_entry.insert(0,"20"); self.limit_entry.grid(row=1,column=3)

        # output folder
        ttk.Label(ctrl, text="Output folder:").grid(row=4,column=0,sticky="e")
        self.output_entry = ttk.Entry(ctrl, width=60); self.output_entry.insert(0, os.path.abspath("output")); self.output_entry.grid(row=4,column=1,columnspan=2, sticky="w")
        ttk.Button(ctrl, text="Browse...", command=self.choose_output).grid(row=4,column=3)

        # run buttons
        btn_frame = ttk.Frame(ctrl); btn_frame.grid(row=5,column=0,columnspan=4,pady=8)
        ttk.Button(btn_frame, text="Run Selected", command=self.run_selected).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Run All & Compare", command=self.run_all_and_compare).pack(side="left", padx=6)

        # notebook for outputs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.tab_summary = ttk.Frame(self.notebook); self.notebook.add(self.tab_summary, text="Summary")
        self.tab_plots = ttk.Frame(self.notebook); self.notebook.add(self.tab_plots, text="Plots")
        self.tab_details = ttk.Frame(self.notebook); self.notebook.add(self.tab_details, text="Details")

        self.summary_text = scrolledtext.ScrolledText(self.tab_summary, width=110, height=30)
        self.summary_text.pack(fill="both", expand=True)
        self.plots_holder = ttk.Frame(self.tab_plots); self.plots_holder.pack(fill="both", expand=True)
        self.details_text = scrolledtext.ScrolledText(self.tab_details, width=110, height=30)
        self.details_text.pack(fill="both", expand=True)

    def choose_file(self):
        p = filedialog.askopenfilename(filetypes=[("Text files","*.txt"),("CSV files","*.csv"),("All files","*.*")])
        if p:
            self.file_path = p
            messagebox.showinfo("File selected", f"{p}")

    def choose_output(self):
        p = filedialog.askdirectory()
        if p:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, p)

    def _load_data(self):
        if self.source_var.get() == "file":
            if not self.file_path:
                messagebox.showerror("Error","No file selected.")
                return None, None
            return task3edit.load_data_from_file(self.file_path)
        else:
            return task3edit.generate_random_data(int(self.pms_entry.get()), int(self.vms_entry.get()))

    def _params(self):
        return dict(
            num_employed=int(self.employed_entry.get()),
            num_onlooker=int(self.onlooker_entry.get()),
            max_iterations=int(self.iters_entry.get()),
            limit=int(self.limit_entry.get()),
            out_folder=self.output_entry.get()
        )

    def _clear_plots(self):
        for w in self.plots_holder.winfo_children():
            w.destroy()

    def _embed_images(self, paths):
        # embed thumbnails vertically
        try:
            from PIL import Image, ImageTk
        except Exception:
            Image = None
            ImageTk = None
        self._clear_plots()
        for p in paths:
            if Image:
                img = Image.open(p)
                img.thumbnail((900,450))
                photo = ImageTk.PhotoImage(img)
                lbl = ttk.Label(self.plots_holder, image=photo)
                lbl.image = photo
                lbl.pack(padx=4, pady=4)
            else:
                ttk.Label(self.plots_holder, text=p).pack()

    def run_selected(self):
        chosen = [k for k,v in self.alg_vars.items() if v.get()]
        if not chosen:
            messagebox.showerror("Error","Select at least one algorithm")
            return
        pm_list, vm_list = self._load_data()
        if pm_list is None:
            return
        params = self._params()
        all_saved = []
        summary_lines = []
        # run each chosen
        for alg in chosen:
            alg_out = os.path.join(params['out_folder'], alg)
            ensure_dir(alg_out)
            if alg == "ABC":
                best, results, saved = run_abc(pm_list, vm_list,
                                              params['num_employed'], params['num_onlooker'],
                                              params['max_iterations'], params['limit'],
                                              out_folder=alg_out)
            elif alg == "UbABC":
                best, results, saved = run_ubabc(pm_list, vm_list,
                                                 params['num_employed'], params['num_onlooker'],
                                                 params['max_iterations'], params['limit'],
                                                 out_folder=alg_out)
            elif alg == "FirstFit":
                best, results, saved = run_first_fit(pm_list, vm_list, out_folder=alg_out)
            elif alg == "BestFit":
                best, results, saved = run_best_fit(pm_list, vm_list, out_folder=alg_out)
            else:
                continue
            all_saved += saved
            summary_lines.append(f"{alg}: final best = {results.get('best_history',[-1])[-1]:.4f}")
            # show details (allocation)
            self.details_text.insert(tk.END, f"=== {alg} allocation ===\n")
            pm_alloc = {}
            for vm_id, pm_id in enumerate(best.assignment):
                if pm_id != -1:
                    pm_alloc.setdefault(pm_id, []).append(vm_id)
            for pm_id,vms in sorted(pm_alloc.items()):
                self.details_text.insert(tk.END, f" {alg} - PM{pm_id}: VMs={vms}\n")
        # update GUI
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, "Run Summary\n"+"="*50+"\n")
        for l in summary_lines: self.summary_text.insert(tk.END, l+"\n")
        self._embed_images(all_saved)
        messagebox.showinfo("Done", "Selected algorithms finished. Images saved to output folder.")

    def run_all_and_compare(self):
        # force run all selected algorithms and create comparison chart if histories available
        pm_list, vm_list = self._load_data()
        if pm_list is None:
            return
        params = self._params()
        ensure_dir(params['out_folder'])
        # run each algorithm (even if not selected) for comparison
        a_best, a_res, a_saved = run_abc(pm_list, vm_list, params['num_employed'], params['num_onlooker'], params['max_iterations'], params['limit'], out_folder=os.path.join(params['out_folder'],"ABC"))
        u_best, u_res, u_saved = run_ubabc(pm_list, vm_list, params['num_employed'], params['num_onlooker'], params['max_iterations'], params['limit'], out_folder=os.path.join(params['out_folder'],"UbABC"))
        f_best, f_res, f_saved = run_first_fit(pm_list, vm_list, out_folder=os.path.join(params['out_folder'],"FirstFit"))
        b_best, b_res, b_saved = run_best_fit(pm_list, vm_list, out_folder=os.path.join(params['out_folder'],"BestFit"))
        # create comparison plot of best histories
        def pad(a,n):
            if not a: return [0]*n
            if len(a)>=n: return a
            return a + [a[-1]]*(n-len(a))
        # get histories
        A = a_res.get('best_history',[])
        U = u_res.get('best_history',[])
        F = f_res.get('best_history',[])
        B = b_res.get('best_history',[])
        n = max(len(A),len(U),len(F),len(B))
        A = pad(A,n); U = pad(U,n); F = pad(F,n); B = pad(B,n)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9,5))
        ax.plot(A, label="ABC")
        ax.plot(U, label="UbABC")
        ax.plot(F, label="FirstFit")
        ax.plot(B, label="BestFit")
        ax.set_xlabel("Iteration"); ax.set_ylabel("Best fitness"); ax.set_title("Algorithms comparison (best fitness)")
        ax.grid(True); ax.legend()
        comp_path = os.path.join(params['out_folder'], f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.tight_layout(); fig.savefig(comp_path); plt.close(fig)
        all_images = a_res.get('saved_pngs',[])+u_res.get('saved_pngs',[])+f_res.get('saved_pngs',[])+b_res.get('saved_pngs',[])+[comp_path]
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, "Comparison Results\n"+"="*60+"\n")
        self.summary_text.insert(tk.END, f"ABC final: {A[-1]:.4f}\nUbABC final: {U[-1]:.4f}\nFirstFit final: {F[-1]:.4f}\nBestFit final: {B[-1]:.4f}\n")
        self._embed_images(all_images)
        messagebox.showinfo("Done", f"All algorithms finished. Comparison saved to {comp_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VMPlacementApp(root)
    root.mainloop()
