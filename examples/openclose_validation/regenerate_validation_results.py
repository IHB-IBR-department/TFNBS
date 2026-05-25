"""
Master regeneration script for the Open-Close Validation Suite.
This script runs the entire pipeline: Inference -> Audits -> Plots.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    # 1. Inference & Harmonization
    ("Inference: Paired TFNBS", ["python", "run_paired_tfnbs.py"]),
    ("Harmonization: ComBat Pooled", ["python", "harmonize_pooled_cohorts.py"]),
    ("Inference: Pooled Methods", ["python", "run_pooled_methods.py"]),
    ("Inference: Retest/Null Methods", ["python", "run_retest_methods.py"]),
    ("Inference: EH Sensitivity Grid", ["python", "run_eh_sensitivity.py"]),
    ("Inference: ML Feature Selection", ["python", "ml/run_ml_feature_selection.py"]),
    
    # 2. Audits
    ("Audit: Agreement Metrics", ["python", "audit_openclose_agreement.py"]),
    ("Audit: ML Transfer Metrics", ["python", "audit_ml_feature_selection.py"]),
    
    # 3. Plotting
    ("Plot 1: ComBat Impact", ["python", "plot1_combat_impact.py"]),
    ("Plot 2: Block Mass Convergence", ["python", "plot2_block_mass_convergence.py"]),
    ("Plot 3: Method Sensitivity", ["python", "plot3_method_sensitivity.py"]),
    ("Plot 5: Retest Specificity", ["python", "plot5_retest_specificity.py"]),
    ("Plot 6: Hierarchy of Constraint", ["python", "plot6_network_informed_hierarchy.py"]),
    ("Plot 8: Retest/Null Hierarchy", ["python", "plot6_retest_hierarchy.py"]),
    ("Plot 7: ML Transfer AUC", ["python", "plot7_ml_transfer_auc.py"]),
]

def main():
    print("=" * 60)
    print("REGENERATING OPEN-CLOSE VALIDATION SUITE")
    print("=" * 60)
    
    start_all = time.time()
    for i, (name, cmd) in enumerate(STEPS):
        print(f"\n[{i+1}/{len(STEPS)}] Running {name}...")
        t0 = time.time()
        try:
            # Run command, capture output but only show if it fails
            result = subprocess.run(cmd, cwd=HERE, check=True, capture_output=True, text=True)
            dt = time.time() - t0
            print(f"  Done in {dt:.1f}s")
            # Print last line of output if available (often contains save path)
            last_line = result.stdout.strip().split('\n')[-1]
            if last_line:
                print(f"  {last_line}")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR running {name}:")
            print(e.stderr)
            sys.exit(1)
            
    print("\n" + "=" * 60)
    print(f"VALIDATION COMPLETE in {time.time() - start_all:.1f}s")
    print("All artifacts and plots are ready in the results/ directory.")
    print("=" * 60)

if __name__ == "__main__":
    main()
