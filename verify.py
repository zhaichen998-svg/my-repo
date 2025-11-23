#!/usr/bin/env python3
"""
Comprehensive verification script for FSS Network implementation
Checks all components, dependencies, and functionality
"""

import sys
import os
import importlib.util


def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False


def check_dependency(package_name):
    """Check if a Python package is installed"""
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        print(f"✓ {package_name} installed")
        return True
    else:
        print(f"✗ {package_name} NOT installed")
        return False


def verify_project_structure():
    """Verify the project structure"""
    print("\n" + "=" * 70)
    print("VERIFYING PROJECT STRUCTURE")
    print("=" * 70)
    
    files = [
        ("README.md", "Main README"),
        ("TRAINING_GUIDE.md", "Training Guide"),
        ("IMPLEMENTATION_SUMMARY.md", "Implementation Summary"),
        ("requirements.txt", "Requirements file"),
        ("config.yaml", "Configuration file"),
        (".gitignore", "Git ignore file"),
        ("train.py", "Training script"),
        ("predict.py", "Prediction script"),
        ("demo.py", "Demo script"),
        ("example.py", "Example script"),
        ("test.py", "Test suite"),
        ("models/__init__.py", "Models package init"),
        ("models/fss_network.py", "FSS Network module"),
        ("models/modules.py", "Network modules"),
        ("data/__init__.py", "Data package init"),
        ("data/dataset.py", "Dataset module"),
        ("utils/__init__.py", "Utils package init"),
        ("utils/metrics.py", "Metrics module"),
        ("utils/visualization.py", "Visualization module"),
    ]
    
    all_exist = True
    for filepath, description in files:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist


def verify_dependencies():
    """Verify required dependencies are installed"""
    print("\n" + "=" * 70)
    print("VERIFYING DEPENDENCIES")
    print("=" * 70)
    
    dependencies = [
        "torch",
        "torchvision",
        "numpy",
        "PIL",
        "cv2",
        "matplotlib",
        "tqdm",
        "tensorboard"
    ]
    
    all_installed = True
    for dep in dependencies:
        if not check_dependency(dep):
            all_installed = False
    
    return all_installed


def verify_imports():
    """Verify all modules can be imported"""
    print("\n" + "=" * 70)
    print("VERIFYING MODULE IMPORTS")
    print("=" * 70)
    
    imports = [
        ("models", "FSSNetwork"),
        ("models.modules", "FeatureExtractor"),
        ("models.modules", "SFEM"),
        ("models.modules", "FeatureMatching"),
        ("models.modules", "HPKIM"),
        ("data", "SegPPDDataset"),
        ("data", "MetaLearningDataLoader"),
        ("utils", "compute_iou"),
        ("utils", "compute_dice"),
        ("utils", "compute_pixel_accuracy"),
        ("utils", "visualize_prediction"),
    ]
    
    all_imported = True
    for module_name, obj_name in imports:
        try:
            module = __import__(module_name, fromlist=[obj_name])
            obj = getattr(module, obj_name)
            print(f"✓ Successfully imported {module_name}.{obj_name}")
        except Exception as e:
            print(f"✗ Failed to import {module_name}.{obj_name}: {str(e)}")
            all_imported = False
    
    return all_imported


def verify_model_creation():
    """Verify model can be created"""
    print("\n" + "=" * 70)
    print("VERIFYING MODEL CREATION")
    print("=" * 70)
    
    try:
        import torch
        from models import FSSNetwork
        
        print("Creating FSS Network...")
        model = FSSNetwork(pretrained=False)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created successfully")
        print(f"✓ Total parameters: {num_params:,}")
        
        # Test forward pass
        print("Testing forward pass...")
        query = torch.randn(1, 3, 400, 400)
        support = torch.randn(1, 3, 400, 400)
        mask = torch.randint(0, 2, (1, 400, 400)).float()
        
        with torch.no_grad():
            output = model(query, support, mask)
        
        print(f"✓ Forward pass successful")
        print(f"✓ Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {str(e)}")
        return False


def verify_tests():
    """Run the test suite"""
    print("\n" + "=" * 70)
    print("RUNNING TEST SUITE")
    print("=" * 70)
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "test.py"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print("✓ All tests passed")
            return True
        else:
            print("✗ Some tests failed")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"✗ Test execution failed: {str(e)}")
        return False


def verify_documentation():
    """Verify documentation completeness"""
    print("\n" + "=" * 70)
    print("VERIFYING DOCUMENTATION")
    print("=" * 70)
    
    doc_checks = []
    
    # Check README
    with open("README.md", "r") as f:
        readme_content = f.read()
        doc_checks.append(("Installation" in readme_content or "installation" in readme_content, "Installation instructions in README"))
        doc_checks.append(("Training" in readme_content, "Training section in README"))
        doc_checks.append(("Inference" in readme_content, "Inference section in README"))
        doc_checks.append(("Architecture" in readme_content or "architecture" in readme_content, "Architecture description in README"))
    
    # Check TRAINING_GUIDE
    with open("TRAINING_GUIDE.md", "r") as f:
        guide_content = f.read()
        doc_checks.append(("Quick Start" in guide_content, "Quick Start in TRAINING_GUIDE"))
        doc_checks.append(("Parameters" in guide_content or "parameters" in guide_content, "Parameters documentation"))
        doc_checks.append(("TensorBoard" in guide_content, "TensorBoard instructions"))
    
    all_documented = True
    for check, description in doc_checks:
        if check:
            print(f"✓ {description}")
        else:
            print(f"✗ Missing: {description}")
            all_documented = False
    
    return all_documented


def main():
    """Run all verification checks"""
    print("=" * 70)
    print("FSS NETWORK - COMPREHENSIVE VERIFICATION")
    print("=" * 70)
    
    checks = [
        ("Project Structure", verify_project_structure),
        ("Dependencies", verify_dependencies),
        ("Module Imports", verify_imports),
        ("Model Creation", verify_model_creation),
        ("Test Suite", verify_tests),
        ("Documentation", verify_documentation),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ {name} check failed with exception: {str(e)}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:.<50} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓✓✓ ALL VERIFICATION CHECKS PASSED ✓✓✓")
        print("The FSS Network implementation is complete and functional!")
    else:
        print("✗✗✗ SOME VERIFICATION CHECKS FAILED ✗✗✗")
        print("Please review the failed checks above.")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
