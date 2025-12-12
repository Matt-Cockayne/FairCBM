#!/usr/bin/env python3
"""
Verification script for fairness-aware CBM implementation.
Checks all components, dependencies, and file structure.

Usage:
    cd FairCBM && python verify_setup.py
"""

import sys
import os
from pathlib import Path
import importlib.util

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(msg):
    print(f"{GREEN}✓{RESET} {msg}")

def print_error(msg):
    print(f"{RED}✗{RESET} {msg}")

def print_warning(msg):
    print(f"{YELLOW}⚠{RESET} {msg}")

def print_section(msg):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{msg}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print_success(f"{description}: {filepath}")
        return True
    else:
        print_error(f"{description} NOT FOUND: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    if Path(dirpath).is_dir():
        print_success(f"{description}: {dirpath}")
        return True
    else:
        print_error(f"{description} NOT FOUND: {dirpath}")
        return False

def check_import(module_name, description):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print_success(f"{description}: {module_name}")
        return True
    except ImportError as e:
        print_error(f"{description} IMPORT FAILED: {module_name}")
        print(f"  Error: {e}")
        return False

def check_python_file(filepath, description):
    """Check if a Python file exists and can be parsed."""
    if not check_file_exists(filepath, description):
        return False
    
    try:
        spec = importlib.util.spec_from_file_location("temp_module", filepath)
        if spec is None:
            print_error(f"  Cannot load spec for {filepath}")
            return False
        module = importlib.util.module_from_spec(spec)
        # Don't execute, just check syntax
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        return True
    except SyntaxError as e:
        print_error(f"  Syntax error in {filepath}: {e}")
        return False
    except Exception as e:
        print_error(f"  Error checking {filepath}: {e}")
        return False

def main():
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}FAIRNESS-AWARE CBM SETUP VERIFICATION{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    # Ensure we're in the FairCBM directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    all_checks_passed = True
    
    # 1. Check directory structure
    print_section("1. Directory Structure")
    dirs = [
        ('src', 'Source code directory'),
        ('src/models', 'Models directory'),
        ('src/utils', 'Utils directory'),
        ('src/data', 'Data directory'),
        ('scripts', 'Scripts directory'),
        ('slurm', 'SLURM scripts directory'),
        ('results', 'Results directory'),
        ('logs', 'Logs directory'),
        ('docs', 'Documentation directory'),
    ]
    
    for dirpath, desc in dirs:
        if not check_directory_exists(dirpath, desc):
            all_checks_passed = False
    
    # 2. Check core Python components
    print_section("2. Core Python Components")
    components = [
        ('src/models/fairness_aware_cbm.py', 'FairnessAwareCBM model'),
        ('src/models/minimal_curriculum_cbm.py', 'MinimalCurriculumCBM model'),
        ('src/models/adversarial_discriminator.py', 'Adversarial discriminator'),
        ('src/utils/fairness_metrics.py', 'Fairness metrics'),
        ('src/utils/adversarial_debiasing.py', 'Adversarial debiasing'),
        ('src/data/dataloader.py', 'SkinCap dataloader'),
    ]
    
    for filepath, desc in components:
        if not check_python_file(filepath, desc):
            all_checks_passed = False
    
    # 3. Check training infrastructure
    print_section("3. Training Infrastructure")
    scripts = [
        ('scripts/train_all_models.py', 'Training script'),
        ('scripts/evaluate_fairness_comparison.py', 'Evaluation script'),
        ('scripts/analyze_multi_run_results.py', 'Analysis script'),
    ]
    
    for filepath, desc in scripts:
        if not check_python_file(filepath, desc):
            all_checks_passed = False
    
    # 4. Check SLURM scripts
    print_section("4. SLURM Scripts")
    slurm_scripts = [
        ('slurm/run_single_experiment.slurm', 'Single experiment SLURM'),
        ('slurm/run_multi_experiments.slurm', 'Multi-run SLURM'),
    ]
    
    for filepath, desc in slurm_scripts:
        if not check_file_exists(filepath, desc):
            all_checks_passed = False
    
    # 5. Check documentation
    print_section("5. Documentation")
    docs = [
        ('README.md', 'README'),
        ('INTEGRATION_PLAN.md', 'Integration plan'),
        ('REVISED_PLAN.md', 'Revised plan'),
        ('PROJECT_STATUS.md', 'Project status'),
        ('QUICKSTART.md', 'Quickstart guide'),
        ('USAGE_GUIDE.md', 'Usage guide'),
        ('IMPLEMENTATION_COMPLETE.md', 'Implementation summary'),
        ('requirements.txt', 'Requirements file'),
    ]
    
    for filepath, desc in docs:
        if not check_file_exists(filepath, desc):
            all_checks_passed = False
    
    # 6. Check existing SynergyCBM dependencies
    print_section("6. Existing SynergyCBM Components")
    synergy_components = [
        ('src/data/dataloader.py', 'SkinCap dataloader'),
        ('src/models/minimal_curriculum_cbm.py', 'MinimalCurriculumCBM'),
        ('src/models/direct_classifier.py', 'DirectClassifier'),
        ('src/models/standard_cbm.py', 'StandardCBM'),
        ('src/utils/metrics.py', 'Metrics utilities'),
    ]
    
    for filepath, desc in synergy_components:
        if not check_file_exists(filepath, desc):
            print_warning(f"{desc} not found (may need to be implemented)")
    
    # 7. Check Python dependencies
    print_section("7. Python Dependencies")
    dependencies = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('scipy', 'SciPy'),
        ('tqdm', 'tqdm'),
    ]
    
    for module, desc in dependencies:
        if not check_import(module, desc):
            all_checks_passed = False
    
    # Optional dependencies
    optional_deps = [
        ('aequitas', 'Aequitas (optional fairness auditing)'),
        ('fairlearn', 'Fairlearn (optional fairness toolkit)'),
    ]
    
    print("\nOptional dependencies:")
    for module, desc in optional_deps:
        check_import(module, desc)
    
    # 8. Check data files
    print_section("8. Data Files")
    data_files = [
        ('data/skincap_concepts.txt', 'SkinCap concepts'),
        ('data/concepts.txt', 'CelebA concepts (fallback)'),
    ]
    
    for filepath, desc in data_files:
        if Path(filepath).exists():
            print_success(f"{desc}: {filepath}")
        else:
            print_warning(f"{desc} not found: {filepath} (may need to be created)")
    
    # 9. Test imports
    print_section("9. Test Imports")
    
    # Add paths to sys.path
    sys.path.insert(0, str(Path.cwd()))
    sys.path.insert(0, str(Path.cwd() / 'fairness'))
    
    test_imports = [
        ('fairness.src.models.fairness_aware_cbm', 'FairnessAwareCBM import'),
        ('fairness.src.models.adversarial_discriminator', 'AdversarialDiscriminator import'),
        ('fairness.src.utils.fairness_metrics', 'Fairness metrics import'),
        ('fairness.src.utils.adversarial_debiasing', 'Adversarial debiasing import'),
    ]
    
    for module, desc in test_imports:
        if not check_import(module, desc):
            all_checks_passed = False
    
    # 10. Summary
    print_section("10. Summary")
    
    if all_checks_passed:
        print_success("ALL CRITICAL CHECKS PASSED!")
        print(f"\n{GREEN}The fairness-aware CBM system is properly set up and ready to use.{RESET}")
        print("\nNext steps:")
        print("  1. Run quick test: ./fairness/quick_test.sh fair_curriculum_cbm 20")
        print("  2. Submit single job: sbatch fairness/slurm/run_single_experiment.slurm")
        print("  3. Launch full experiments: sbatch fairness/slurm/run_multi_experiments.slurm")
    else:
        print_error("SOME CHECKS FAILED!")
        print(f"\n{RED}Please fix the errors above before running experiments.{RESET}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
