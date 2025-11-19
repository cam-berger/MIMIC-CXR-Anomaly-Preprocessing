#!/usr/bin/env python3
"""
Test script to verify Step 2 setup and dependencies.
Run this before executing the main preprocessing pipeline.
"""
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Check Python version"""
    logger.info("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    logger.info(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check required Python packages"""
    logger.info("\nChecking dependencies...")
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'transformers': 'Transformers',
        'sentence_transformers': 'Sentence Transformers',
        'spacy': 'spaCy',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm'
    }

    optional = {
        'langchain': 'LangChain',
        'langchain_anthropic': 'LangChain Anthropic',
        'anthropic': 'Anthropic'
    }

    all_ok = True

    # Check required
    for module, name in required.items():
        try:
            __import__(module)
            logger.info(f"  ✓ {name}")
        except ImportError:
            logger.error(f"  ✗ {name} - REQUIRED")
            all_ok = False

    # Check optional
    for module, name in optional.items():
        try:
            __import__(module)
            logger.info(f"  ✓ {name} (optional)")
        except ImportError:
            logger.warning(f"  ⚠ {name} (optional - needed for Claude summarization)")

    return all_ok


def check_spacy_model():
    """Check if scispacy model is installed"""
    logger.info("\nChecking spaCy model...")
    try:
        import spacy
        nlp = spacy.load('en_core_sci_md')
        logger.info("  ✓ en_core_sci_md (scispacy)")
        return True
    except OSError:
        logger.error("  ✗ en_core_sci_md not found")
        logger.error("    Install with: python -m spacy download en_core_sci_md")
        return False


def check_data_paths():
    """Check if data paths exist"""
    logger.info("\nChecking data paths...")

    try:
        from config.paths import Step2Paths
        paths = Step2Paths()

        checks = [
            ("Step 1 train cohort", paths.step1_train),
            ("Step 1 val cohort", paths.step1_val),
            ("MIMIC-CXR base", paths.mimic_cxr_base),
            ("MIMIC-IV base", paths.mimic_iv_base),
            ("MIMIC-ED base", paths.mimic_ed_base),
        ]

        all_ok = True
        for name, path in checks:
            if path.exists():
                logger.info(f"  ✓ {name}: {path}")
            else:
                logger.error(f"  ✗ {name}: {path} - NOT FOUND")
                all_ok = False

        return all_ok

    except Exception as e:
        logger.error(f"  Error loading paths: {e}")
        return False


def check_config():
    """Check configuration file"""
    logger.info("\nChecking configuration...")

    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    if not config_path.exists():
        logger.error(f"  ✗ Config file not found: {config_path}")
        return False

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"  ✓ Config file: {config_path}")
        logger.info(f"    - Image normalization: {config['image']['normalize_method']}")
        logger.info(f"    - Full resolution: {config['image']['preserve_full_resolution']}")
        logger.info(f"    - Missing token: {config['structured']['missing_token']}")
        logger.info(f"    - Claude model: {config['text']['summarization']['model']}")

        return True

    except Exception as e:
        logger.error(f"  ✗ Error reading config: {e}")
        return False


def check_gpu():
    """Check GPU availability"""
    logger.info("\nChecking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"    - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("  ⚠ CUDA not available - will use CPU (slower)")
        return True
    except Exception as e:
        logger.error(f"  Error checking GPU: {e}")
        return False


def estimate_memory():
    """Estimate memory requirements"""
    logger.info("\nMemory requirements:")
    logger.info("  - Full-resolution image: ~30 MB each")
    logger.info("  - Batch size 4: ~120 MB images + overhead")
    logger.info("  - Recommended RAM: 16+ GB")
    logger.info("  - Recommended GPU: 8+ GB VRAM (optional)")


def main():
    logger.info("="*60)
    logger.info("STEP 2 PREPROCESSING - SETUP VERIFICATION")
    logger.info("="*60 + "\n")

    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("spaCy model", check_spacy_model),
        ("Data paths", check_data_paths),
        ("Configuration", check_config),
        ("GPU", check_gpu),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            logger.error(f"\nError during {name} check: {e}")
            results[name] = False

    estimate_memory()

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    required_checks = ["Python version", "Dependencies", "spaCy model", "Data paths", "Configuration"]
    required_ok = all(results.get(check, False) for check in required_checks)

    if required_ok:
        logger.info("✓ All required checks passed!")
        logger.info("\nYou can now run the preprocessing pipeline:")
        logger.info("  python main.py --max-samples 10  # Test on 10 samples")
        logger.info("  python main.py  # Process full dataset")
        return 0
    else:
        logger.error("✗ Some required checks failed")
        logger.error("\nPlease fix the errors above before running the pipeline")
        return 1


if __name__ == '__main__':
    sys.exit(main())
