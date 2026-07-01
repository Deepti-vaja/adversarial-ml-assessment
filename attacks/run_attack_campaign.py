"""
Orchestrator script for Task B Adversarial Evaluation Campaign.
Executes FGSM, PGD, and C&W attacks under structured MLflow nested tracking runs,
analyzes per-class vulnerability across super-classes, and outputs reports/attack_evaluation.md.
"""

import os
import argparse
import sys
from typing import Dict, Any, Tuple
import numpy as np
import torch
import mlflow

from data.cifar10_loader import get_dataloaders, load_config
from data.mapping import SUPERCLASS_NAMES
from attacks.art_wrapper import load_wrapped_model
from attacks.fgsm_sweep import evaluate_fgsm_sweep, plot_fgsm_sweep
from attacks.pgd_attack import evaluate_pgd_attacks
from attacks.cw_attack import sample_stratified_subset, evaluate_cw_attack
from utils.seed import set_seed
from utils.logging import get_logger
from utils.mlflow_helpers import setup_experiment, log_params, log_artifact

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Adversarial Evaluation Campaign")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/attacks.yaml",
        help="Path to attacks configuration YAML"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of test samples evaluated (useful for rapid testing)"
    )
    return parser.parse_args()

def extract_test_arrays(test_loader: torch.utils.data.DataLoader, max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts numpy arrays (x, y) from test DataLoader."""
    x_list, y_list = [], []
    total = 0
    for images, targets in test_loader:
        x_list.append(images.numpy())
        y_list.append(targets.numpy())
        total += len(targets)
        if max_samples and total >= max_samples:
            break
    x_arr = np.concatenate(x_list, axis=0)
    y_arr = np.concatenate(y_list, axis=0)
    if max_samples:
        return x_arr[:max_samples], y_arr[:max_samples]
    return x_arr, y_arr

def analyze_per_class_vulnerability(
    clean_y: np.ndarray,
    clean_preds: np.ndarray,
    adv_preds: np.ndarray
) -> Dict[str, float]:
    """Calculates Attack Success Rate (ASR) per superclass."""
    vulnerabilities = {}
    for c_idx, c_name in enumerate(SUPERCLASS_NAMES):
        mask = (clean_y == c_idx) & (clean_preds == c_idx)
        if np.any(mask):
            asr = float(np.mean(adv_preds[mask] != c_idx))
        else:
            asr = 0.0
        vulnerabilities[c_name] = asr
    return vulnerabilities

def generate_evaluation_report(
    fgsm_res: Dict[str, Any],
    pgd_res: Dict[str, Any],
    cw_res: Dict[str, Any],
    per_class_vuln: Dict[str, Dict[str, float]],
    output_path: str = "./reports/attack_evaluation.md"
) -> str:
    """Generates markdown evaluation summary report."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Determine most vulnerable class across attacks
    avg_vuln = {}
    for c_name in SUPERCLASS_NAMES:
        vals = [per_class_vuln.get(atk, {}).get(c_name, 0.0) for atk in per_class_vuln]
        avg_vuln[c_name] = float(np.mean(vals)) if vals else 0.0
    most_vulnerable = max(avg_vuln.items(), key=lambda kv: kv[1])[0]

    lines = [
        "# Task B: Adversarial Evaluation Report",
        "",
        "## Summary of Robustness Evaluation",
        "This report evaluates the baseline `FraudCNN` model against white-box evasion attacks across varying perturbation norms ($L_\\infty$ and $L_2$).",
        "",
        f"**Clean Test Set Baseline Accuracy**: `{fgsm_res['clean_accuracy']*100:.2f}%`",
        f"**Most Vulnerable Superclass Overall**: `{most_vulnerable}` (Average ASR: `{avg_vuln[most_vulnerable]*100:.2f}%`)",
        "",
        "---",
        "",
        "## 1. Fast Gradient Sign Method (FGSM) Sweep",
        "Single-step $L_\\infty$ perturbation sweep evaluating baseline degradation.",
        "",
        "| Epsilon (L-inf Bound) | Adversarial Accuracy (%) | Attack Success Rate (%) | Mean L-inf Norm | Mean L2 Norm |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]

    for eps in fgsm_res["epsilons"]:
        acc = fgsm_res["adversarial_accuracy"][eps] * 100
        asr = fgsm_res["attack_success_rate"][eps] * 100
        linf = fgsm_res["mean_linf_norm"][eps]
        l2 = fgsm_res["mean_l2_norm"][eps]
        lines.append(f"| `{eps:.2f}` | `{acc:.2f}%` | `{asr:.2f}%` | `{linf:.4f}` | `{l2:.4f}` |")

    lines.extend([
        "",
        "![FGSM Epsilon Sweep Curve](./fgsm_epsilon_sweep.png)",
        "",
        "---",
        "",
        "## 2. Projected Gradient Descent (PGD) Evaluation",
        f"Iterative multi-step attack comparison at $\\epsilon = {pgd_res['eps']}$, step size $\\alpha = {pgd_res['eps_step']}$.",
        "",
        "| Step Budget | Adversarial Accuracy (%) | Attack Success Rate (%) | Mean L-inf Norm | Mean L2 Norm |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ])

    for steps in pgd_res["steps_evaluated"]:
        acc = pgd_res["adversarial_accuracy"][steps] * 100
        asr = pgd_res["attack_success_rate"][steps] * 100
        linf = pgd_res["mean_linf_norm"][steps]
        l2 = pgd_res["mean_l2_norm"][steps]
        lines.append(f"| `{steps:d}` | `{acc:.2f}%` | `{asr:.2f}%` | `{linf:.4f}` | `{l2:.4f}` |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Carlini & Wagner (C&W) L2 Targeted/Untargeted Evaluation",
        f"Optimization-based minimum distortion evaluation on `{cw_res['samples_evaluated']}` stratified test samples.",
        "",
        f"- **Adversarial Accuracy**: `{cw_res['adversarial_accuracy']*100:.2f}%`",
        f"- **Attack Success Rate (ASR)**: `{cw_res['attack_success_rate']*100:.2f}%`",
        f"- **Mean L2 Distortion**: `{cw_res['mean_l2_norm']:.4f}`",
        f"- **Mean L-inf Norm**: `{cw_res['mean_linf_norm']:.4f}`",
        "",
        "---",
        "",
        "## 4. Per-Class Vulnerability Analysis",
        "Breakdown of Attack Success Rate (ASR %) across proxy document superclasses.",
        "",
        "| Attack Regime | Genuine (%) | Tampered (%) | Forged (%) |",
        "| :--- | :--- | :--- | :--- |"
    ])

    for atk, vulns in per_class_vuln.items():
        lines.append(f"| `{atk}` | `{vulns.get('Genuine', 0.0)*100:.2f}%` | `{vulns.get('Tampered', 0.0)*100:.2f}%` | `{vulns.get('Forged', 0.0)*100:.2f}%` |")

    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path

def run_campaign(args: argparse.Namespace) -> Dict[str, Any]:
    logger = get_logger("AttackCampaign", log_file="./reports/attack_campaign.log")
    logger.info(f"Starting Adversarial Evaluation Campaign using config: {args.config}")

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    max_samples = args.max_samples or config.get("evaluation", {}).get("max_samples", None)
    batch_size = config.get("evaluation", {}).get("batch_size", 64)

    # Load data
    logger.info("Extracting CIFAR-10 test set arrays...")
    _, _, test_loader = get_dataloaders(load_config(config.get("model", {}).get("config_path", "./configs/baseline.yaml")))
    x_test, y_test = extract_test_arrays(test_loader, max_samples=max_samples)
    logger.info(f"Loaded evaluation dataset shape: {x_test.shape}")

    # Load wrapped model
    logger.info("Loading baseline FraudCNN checkpoint and wrapping in ART PyTorchClassifier...")
    model_cfg_path = config.get("model", {}).get("config_path", "./configs/baseline.yaml")
    ckpt_path = config.get("model", {}).get("checkpoint_path", "./models/checkpoints/fraud_cnn_baseline.pth")
    _, classifier = load_wrapped_model(model_cfg_path, ckpt_path)

    # Pre-compute clean prediction labels for per-class analysis
    clean_preds = np.argmax(classifier.predict(x_test, batch_size=batch_size), axis=1)

    setup_experiment(experiment_name="AML-EVAL-CAMPAIGN")

    per_class_vuln = {}

    with mlflow.start_run(run_name="adversarial_evaluation_campaign") as parent_run:
        logger.info(f"Parent Campaign Run ID started: {parent_run.info.run_id}")
        log_params({"seed": seed, "max_samples": max_samples or len(x_test), "batch_size": batch_size})

        # 1. FGSM Child Run
        with mlflow.start_run(run_name="FGSM_Sweep", nested=True):
            logger.info("Executing FGSM Epsilon Sweep...")
            fgsm_cfg = config.get("fgsm", {})
            epsilons = fgsm_cfg.get("epsilons", [0.01, 0.05, 0.1, 0.2])
            fgsm_res = evaluate_fgsm_sweep(classifier, x_test, y_test, epsilons, batch_size=batch_size)

            # Log metrics
            mlflow.log_metric("clean_accuracy", fgsm_res["clean_accuracy"])
            for e in epsilons:
                mlflow.log_metric(f"adv_acc_eps_{e}", fgsm_res["adversarial_accuracy"][e])
                mlflow.log_metric(f"asr_eps_{e}", fgsm_res["attack_success_rate"][e])

            # Generate and log plots
            plot_path = fgsm_cfg.get("output_plot_path", "./reports/fgsm_epsilon_sweep.png")
            png_path, csv_path = plot_fgsm_sweep(fgsm_res, plot_path)
            log_artifact(png_path)
            log_artifact(csv_path)

            # Per-class vulnerability at strongest epsilon (max in sweep)
            from art.attacks.evasion import FastGradientMethod
            strongest_eps = max(epsilons)
            atk_strong = FastGradientMethod(estimator=classifier, eps=float(strongest_eps), eps_step=float(strongest_eps), batch_size=batch_size)
            adv_preds_fgsm = np.argmax(classifier.predict(atk_strong.generate(x_test), batch_size=batch_size), axis=1)
            per_class_vuln[f"FGSM (eps={strongest_eps})"] = analyze_per_class_vulnerability(y_test, clean_preds, adv_preds_fgsm)

        # 2. PGD Child Run
        with mlflow.start_run(run_name="PGD_Evaluation", nested=True):
            logger.info("Executing PGD Evaluation...")
            pgd_cfg = config.get("pgd", {})
            eps = pgd_cfg.get("eps", 0.05)
            eps_step = pgd_cfg.get("eps_step", 0.01)
            steps_list = pgd_cfg.get("steps_list", [20, 40])
            pgd_res = evaluate_pgd_attacks(classifier, x_test, y_test, eps, eps_step, steps_list, batch_size=batch_size)

            for s in steps_list:
                mlflow.log_metric(f"adv_acc_steps_{s}", pgd_res["adversarial_accuracy"][s])
                mlflow.log_metric(f"asr_steps_{s}", pgd_res["attack_success_rate"][s])

            # Per-class vulnerability at max steps
            from art.attacks.evasion import ProjectedGradientDescentPyTorch
            max_steps = max(steps_list)
            atk_pgd = ProjectedGradientDescentPyTorch(estimator=classifier, norm=np.inf, eps=float(eps), eps_step=float(eps_step), max_iter=int(max_steps), batch_size=batch_size)
            adv_preds_pgd = np.argmax(classifier.predict(atk_pgd.generate(x_test), batch_size=batch_size), axis=1)
            per_class_vuln[f"PGD ({max_steps}-step)"] = analyze_per_class_vulnerability(y_test, clean_preds, adv_preds_pgd)

        # 3. C&W Child Run
        with mlflow.start_run(run_name="CW_L2_Evaluation", nested=True):
            logger.info("Executing Carlini & Wagner L2 Evaluation...")
            cw_cfg = config.get("cw", {})
            spc = cw_cfg.get("samples_per_class", 200)
            if max_samples:
                spc = min(spc, max(1, max_samples // 3))
            x_cw, y_cw = sample_stratified_subset(x_test, y_test, samples_per_class=spc, seed=seed)
            logger.info(f"C&W Evaluation subset shape: {x_cw.shape}")

            cw_res = evaluate_cw_attack(
                classifier, x_cw, y_cw,
                confidence=cw_cfg.get("confidence", 0.0),
                learning_rate=cw_cfg.get("learning_rate", 0.01),
                max_iter=cw_cfg.get("max_iter", 50),
                binary_search_steps=cw_cfg.get("binary_search_steps", 3),
                batch_size=min(batch_size, 32)
            )

            mlflow.log_metric("cw_adv_accuracy", cw_res["adversarial_accuracy"])
            mlflow.log_metric("cw_asr", cw_res["attack_success_rate"])
            mlflow.log_metric("cw_mean_l2_norm", cw_res["mean_l2_norm"])

            # Per-class vulnerability on C&W subset
            clean_preds_cw = np.argmax(classifier.predict(x_cw, batch_size=min(batch_size, 32)), axis=1)
            from art.attacks.evasion import CarliniL2Method
            atk_cw = CarliniL2Method(classifier=classifier, confidence=float(cw_cfg.get("confidence", 0.0)), learning_rate=float(cw_cfg.get("learning_rate", 0.01)), max_iter=int(cw_cfg.get("max_iter", 50)), binary_search_steps=int(cw_cfg.get("binary_search_steps", 3)), batch_size=min(batch_size, 32))
            adv_preds_cw = np.argmax(classifier.predict(atk_cw.generate(x_cw), batch_size=min(batch_size, 32)), axis=1)
            per_class_vuln["C&W (L2)"] = analyze_per_class_vulnerability(y_cw, clean_preds_cw, adv_preds_cw)

        # Generate markdown evaluation report
        logger.info("Generating reports/attack_evaluation.md...")
        report_path = generate_evaluation_report(fgsm_res, pgd_res, cw_res, per_class_vuln, "./reports/attack_evaluation.md")
        log_artifact(report_path)
        logger.info("Adversarial Evaluation Campaign successfully completed.")

    return {
        "fgsm": fgsm_res,
        "pgd": pgd_res,
        "cw": cw_res,
        "per_class_vulnerability": per_class_vuln
    }

if __name__ == "__main__":
    args = parse_args()
    try:
        run_campaign(args)
    except Exception as e:
        sys.stderr.write(f"Fatal error in campaign run: {str(e)}\n")
        sys.exit(1)
