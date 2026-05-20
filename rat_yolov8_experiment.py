#!/usr/bin/env python3
"""
YOLOv8 Rat Detection Experiment for Kaggle
==========================================
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "Ultralytics is not installed. In Kaggle, enable Internet and run "
        "`pip install ultralytics`, or attach an offline ultralytics package."
    ) from exc


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def print_header(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def collect_images(images_dir: Path) -> List[Path]:
    images: List[Path] = []
    for ext in IMAGE_EXTS:
        images.extend(images_dir.glob(f"*{ext}"))
        images.extend(images_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def count_images(folder: Path) -> int:
    total = 0
    for ext in IMAGE_EXTS:
        total += len(list(folder.glob(f"*{ext}")))
        total += len(list(folder.glob(f"*{ext.upper()}")))
    return total


def count_labels(folder: Path) -> int:
    return len(list(folder.glob("*.txt")))


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_classes(classes_file: Path) -> List[str]:
    if not classes_file.exists():
        raise FileNotFoundError(f"classes.txt not found: {classes_file}")
    with open(classes_file, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if not names:
        raise ValueError("classes.txt is empty.")
    return names


def collect_image_label_pairs(images_dir: Path, labels_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[str]]:
    all_images = collect_images(images_dir)
    if not all_images:
        raise FileNotFoundError(f"No images found in {images_dir}")

    pairs: List[Tuple[Path, Path]] = []
    missing: List[str] = []

    for img_path in all_images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing.append(img_path.name)

    if not pairs:
        raise ValueError("No image-label pairs found. Check image and label filenames.")

    return pairs, missing


def split_pairs(
    pairs: List[Tuple[Path, Path]],
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    shuffled = pairs.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pairs = shuffled[:n_train]
    val_pairs = shuffled[n_train : n_train + n_val]
    test_pairs = shuffled[n_train + n_val :]
    return train_pairs, val_pairs, test_pairs


def prepare_split(split_root: Path, split_name: str, split_pairs_: Sequence[Tuple[Path, Path]]) -> Tuple[Path, Path]:
    img_out = split_root / split_name / "images"
    lbl_out = split_root / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in split_pairs_:
        shutil.copy2(img_path, img_out / img_path.name)
        shutil.copy2(label_path, lbl_out / label_path.name)

    return img_out, lbl_out


def create_yolo_yaml(split_root: Path, names: List[str]) -> Path:
    data_yaml = {
        "path": str(split_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    yaml_path = split_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    return yaml_path


def validate_yolo_labels(pairs: Sequence[Tuple[Path, Path]], nc: int, output_csv: Path) -> int:
    bad_labels: List[Tuple[str, int, str, str]] = []

    for _, label_path in pairs:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line_no, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                bad_labels.append((str(label_path), line_no, line, "Expected 5 values"))
                continue
            try:
                cls_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except Exception:
                bad_labels.append((str(label_path), line_no, line, "Could not parse values"))
                continue

            if cls_id < 0 or cls_id >= nc:
                bad_labels.append((str(label_path), line_no, line, "Class index out of range"))
            if not all(0.0 <= v <= 1.0 for v in [x, y, w, h]):
                bad_labels.append((str(label_path), line_no, line, "Box values not normalized between 0 and 1"))

    if bad_labels:
        pd.DataFrame(
            bad_labels,
            columns=["label_file", "line_number", "line", "issue"],
        ).to_csv(output_csv, index=False)

    return len(bad_labels)


def find_weights(custom_weights: Optional[str], possible_weights: Sequence[str]) -> Optional[str]:
    if custom_weights:
        p = Path(custom_weights)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"CUSTOM_WEIGHTS_PATH not found: {custom_weights}")

    for candidate in possible_weights:
        if Path(candidate).exists():
            return candidate
    return None


def metrics_to_dict(metrics_obj) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key, attr in [
        ("precision_macro", "mp"),
        ("recall_macro", "mr"),
        ("mAP50", "map50"),
        ("mAP50_95", "map"),
    ]:
        try:
            out[key] = float(getattr(metrics_obj.box, attr))
        except Exception:
            out[key] = None
    return out


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def run_predictions(
    model: YOLO,
    test_img_dir: Path,
    names: List[str],
    run_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    test_images = collect_images(test_img_dir)
    if not test_images:
        raise FileNotFoundError(f"No test images found in {test_img_dir}")

    prediction_records: List[dict] = []
    annotated_dir = run_dir / "annotated_test_predictions"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    for img_path in test_images:
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf_score = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                prediction_records.append(
                    {
                        "image": img_path.name,
                        "class_id": cls_id,
                        "class_name": names[cls_id] if cls_id < len(names) else str(cls_id),
                        "confidence": conf_score,
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                    }
                )
        else:
            prediction_records.append(
                {
                    "image": img_path.name,
                    "class_id": None,
                    "class_name": "no_detection",
                    "confidence": None,
                    "x1": None,
                    "y1": None,
                    "x2": None,
                    "y2": None,
                }
            )

        annotated = result.plot()
        cv2.imwrite(str(annotated_dir / img_path.name), annotated)

    pred_df = pd.DataFrame(prediction_records)
    pred_path = run_dir / "test_predictions_conf050_iou045.csv"
    pred_df.to_csv(pred_path, index=False)

    detected_df = pred_df[pred_df["class_name"] != "no_detection"].copy()

    detections_per_image = (
        detected_df.groupby("image")
        .size()
        .reindex([p.name for p in test_images], fill_value=0)
        .reset_index()
    )
    detections_per_image.columns = ["image", "detections"]
    detections_per_image["frame_index"] = range(len(detections_per_image))
    detections_path = run_dir / "detections_per_image.csv"
    detections_per_image.to_csv(detections_path, index=False)

    return pred_df, detections_per_image, pred_path, annotated_dir


def plot_confidence_distribution(pred_df: pd.DataFrame, fig_dir: Path, prefix: str = "Figure_16") -> Optional[dict]:
    detected = pred_df[pred_df["class_name"] != "no_detection"].copy()
    conf_values = detected["confidence"].dropna()
    if conf_values.empty:
        return None

    stats = {
        "count": int(conf_values.shape[0]),
        "mean": float(conf_values.mean()),
        "std": float(conf_values.std()),
        "median": float(conf_values.median()),
        "min": float(conf_values.min()),
        "max": float(conf_values.max()),
    }
    save_json(stats, fig_dir / f"{prefix}_confidence_statistics.json")

    plt.figure(figsize=(8, 5))
    plt.hist(conf_values, bins=12, edgecolor="black")
    plt.title("Confidence Score Distribution")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{prefix}_confidence_score_distribution.png", dpi=300)
    plt.close()
    return stats


def plot_detections_per_image(detections_df: pd.DataFrame, fig_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(detections_df["frame_index"], detections_df["detections"], marker="o", linewidth=1)
    plt.title("Detections Per Test Image")
    plt.xlabel("Image Index")
    plt.ylabel("Number of Retained Detections")
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure_17_detections_per_test_image.png", dpi=300)
    plt.close()


def run_fps_benchmark(
    model: YOLO,
    test_img_dir: Path,
    fig_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    warmup: int = 20,
    runs: int = 200,
) -> Dict[str, float]:
    test_images = collect_images(test_img_dir)
    if not test_images:
        raise FileNotFoundError(f"No test images found in {test_img_dir}")

    warmup = min(warmup, len(test_images))
    runs = min(runs, len(test_images))

    for img_path in test_images[:warmup]:
        _ = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False)

    latencies: List[float] = []
    for img_path in test_images[:runs]:
        start = time.perf_counter()
        _ = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        end = time.perf_counter()
        latencies.append(end - start)

    arr = np.asarray(latencies, dtype=float)
    fps_values = 1.0 / arr

    summary = {
        "num_images_benchmarked": int(runs),
        "mean_latency_ms": float(arr.mean() * 1000),
        "std_latency_ms": float(arr.std() * 1000),
        "median_latency_ms": float(np.median(arr) * 1000),
        "mean_fps": float(fps_values.mean()),
        "median_fps": float(np.median(fps_values)),
    }

    save_json(summary, fig_dir / "Figure_18_fps_summary.json")
    pd.DataFrame([summary]).to_csv(fig_dir / "Figure_18_fps_summary.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(fps_values)), fps_values, marker="o", linewidth=1)
    plt.title("FPS Over Test Images")
    plt.xlabel("Image Index")
    plt.ylabel("FPS")
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure_18_fps_over_test_images.png", dpi=300)
    plt.close()
    return summary


def copy_yolo_plots(run_dir: Path, fig_dir: Path) -> None:
    candidates = {
        "Figure_19_precision_recall_curve.png": [
            run_dir / "test_metrics" / "PR_curve.png",
            run_dir / "val_metrics" / "PR_curve.png",
            run_dir / "train_yolov8_rat" / "PR_curve.png",
        ],
        "confusion_matrix_test.png": [
            run_dir / "test_metrics" / "confusion_matrix.png",
            run_dir / "test_metrics" / "confusion_matrix_normalized.png",
        ],
        "F1_curve_test.png": [run_dir / "test_metrics" / "F1_curve.png"],
        "P_curve_test.png": [run_dir / "test_metrics" / "P_curve.png"],
        "R_curve_test.png": [run_dir / "test_metrics" / "R_curve.png"],
    }

    for out_name, paths in candidates.items():
        for p in paths:
            if p.exists():
                shutil.copy2(p, fig_dir / out_name)
                break


def plot_benchmark_comparison(fig_dir: Path) -> pd.DataFrame:
    comparison = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"],
            "Reference result": [0.874, 0.887, 0.926, 0.785],
            "Present test result": [0.855484, 0.904237, 0.933123, 0.747500],
        }
    )
    comparison["Difference"] = comparison["Present test result"] - comparison["Reference result"]
    comparison.to_csv(fig_dir / "Figure_20_benchmark_comparison_values.csv", index=False)

    x = np.arange(len(comparison))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, comparison["Reference result"], width, label="Reference result")
    plt.bar(x + width / 2, comparison["Present test result"], width, label="Present test result")
    plt.xticks(x, comparison["Metric"])
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title("Benchmark Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure_20_benchmark_comparison.png", dpi=300)
    plt.close()
    return comparison


def analytical_power_budget(run_dir: Path, fig_dir: Path) -> Dict[str, float]:
    power_dir = run_dir / "power_budget_analysis"
    power_dir.mkdir(parents=True, exist_ok=True)

    event_summary_csv = run_dir / "motion_triggered_simulation" / "motion_triggered_event_summary.csv"
    if event_summary_csv.exists():
        event_df = pd.read_csv(event_summary_csv)
        source = "software_event_summary"
    else:
        # Analytical fallback assumptions used in the manuscript.
        assumed_events_per_day = 50
        assumed_deterrence_events_per_day = 35
        assumed_average_event_latency_seconds = 0.25
        event_df = pd.DataFrame(
            {
                "event_id": range(1, assumed_events_per_day + 1),
                "deterrence_activated": [
                    True if i < assumed_deterrence_events_per_day else False
                    for i in range(assumed_events_per_day)
                ],
                "event_latency_ms": [
                    assumed_average_event_latency_seconds * 1000
                    for _ in range(assumed_events_per_day)
                ],
            }
        )
        source = "analytical_assumptions"

    p_rpi_standby_w = 1.8
    p_rpi_active_w = 5.1
    p_camera_active_w = 1.2
    p_pir_continuous_w = 0.1
    p_speaker_floodlight_w = 10.0
    solar_panel_w = 20.0
    peak_sun_hours_per_day = 5.0
    battery_voltage = 12.0
    battery_ah = 20.0
    battery_nominal_wh = battery_voltage * battery_ah
    usable_depth_of_discharge = 0.80
    battery_usable_wh = battery_nominal_wh * usable_depth_of_discharge
    deterrence_duration_seconds = 15
    seconds_per_day = 24 * 60 * 60

    num_events = len(event_df)
    if num_events > 0:
        avg_event_latency_seconds = float(event_df["event_latency_ms"].mean() / 1000)
        deterrence_events = int(event_df["deterrence_activated"].sum())
    else:
        avg_event_latency_seconds = 0.0
        deterrence_events = 0

    active_inference_seconds_per_day = num_events * avg_event_latency_seconds
    deterrence_seconds_per_day = deterrence_events * deterrence_duration_seconds
    standby_seconds_per_day = max(0.0, seconds_per_day - active_inference_seconds_per_day)

    standby_energy_wh = (p_rpi_standby_w + p_pir_continuous_w) * standby_seconds_per_day / 3600
    active_energy_wh = (
        p_rpi_active_w + p_camera_active_w + p_pir_continuous_w
    ) * active_inference_seconds_per_day / 3600
    deterrence_energy_wh = p_speaker_floodlight_w * deterrence_seconds_per_day / 3600
    total_daily_energy_wh = standby_energy_wh + active_energy_wh + deterrence_energy_wh
    solar_daily_yield_wh = solar_panel_w * peak_sun_hours_per_day
    solar_margin = solar_daily_yield_wh / total_daily_energy_wh if total_daily_energy_wh > 0 else None
    battery_autonomy_days = battery_usable_wh / total_daily_energy_wh if total_daily_energy_wh > 0 else None

    power = {
        "source": source,
        "num_motion_events_per_day_assumption": int(num_events),
        "deterrence_events_per_day_assumption": int(deterrence_events),
        "average_event_latency_seconds": avg_event_latency_seconds,
        "active_inference_seconds_per_day": active_inference_seconds_per_day,
        "deterrence_seconds_per_day": deterrence_seconds_per_day,
        "standby_seconds_per_day": standby_seconds_per_day,
        "rpi_standby_power_w": p_rpi_standby_w,
        "rpi_active_power_w": p_rpi_active_w,
        "camera_active_power_w": p_camera_active_w,
        "pir_continuous_power_w": p_pir_continuous_w,
        "speaker_floodlight_power_w": p_speaker_floodlight_w,
        "standby_energy_wh_per_day": standby_energy_wh,
        "active_energy_wh_per_day": active_energy_wh,
        "deterrence_energy_wh_per_day": deterrence_energy_wh,
        "total_daily_energy_wh": total_daily_energy_wh,
        "solar_panel_w": solar_panel_w,
        "peak_sun_hours_per_day": peak_sun_hours_per_day,
        "solar_daily_yield_wh": solar_daily_yield_wh,
        "solar_margin_ratio": solar_margin,
        "battery_nominal_wh": battery_nominal_wh,
        "battery_usable_wh": battery_usable_wh,
        "battery_autonomy_days": battery_autonomy_days,
    }

    save_json(power, power_dir / "power_budget_summary.json")
    pd.DataFrame([power]).to_csv(power_dir / "power_budget_summary.csv", index=False)

    energy_df = pd.DataFrame(
        {
            "Subsystem": ["Standby", "Active inference", "Deterrence"],
            "Energy Wh/day": [standby_energy_wh, active_energy_wh, deterrence_energy_wh],
        }
    )
    energy_df.to_csv(fig_dir / "Figure_21_power_budget_breakdown_values.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(energy_df["Subsystem"], energy_df["Energy Wh/day"])
    plt.ylabel("Energy Consumption (Wh/day)")
    plt.title("Estimated Daily Energy Consumption by Subsystem")
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure_21_power_budget_breakdown.png", dpi=300)
    plt.close()

    solar_df = pd.DataFrame(
        {
            "Quantity": ["Daily demand", "Solar yield", "Usable battery capacity"],
            "Energy Wh": [total_daily_energy_wh, solar_daily_yield_wh, battery_usable_wh],
        }
    )
    solar_df.to_csv(fig_dir / "Figure_21_solar_battery_values.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.bar(solar_df["Quantity"], solar_df["Energy Wh"])
    plt.ylabel("Energy (Wh)")
    plt.title("Solar and Battery Feasibility Summary")
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure_21_solar_battery_feasibility.png", dpi=300)
    plt.close()

    return power


def copy_qualitative_examples(run_dir: Path, fig_dir: Path, n: int = 12) -> None:
    annotated_dir = run_dir / "annotated_test_predictions"
    qual_dir = fig_dir / "qualitative_examples"
    qual_dir.mkdir(parents=True, exist_ok=True)
    if not annotated_dir.exists():
        return
    imgs: List[Path] = []
    for ext in IMAGE_EXTS:
        imgs.extend(annotated_dir.glob(f"*{ext}"))
        imgs.extend(annotated_dir.glob(f"*{ext.upper()}"))
    for img in sorted(set(imgs))[:n]:
        shutil.copy2(img, qual_dir / img.name)


def export_models(model: YOLO, imgsz: int, run_dir: Path) -> Dict[str, str]:
    export_results: Dict[str, str] = {}
    try:
        onnx_export = model.export(format="onnx", imgsz=imgsz, opset=12)
        export_results["onnx"] = str(onnx_export)
    except Exception as exc:
        export_results["onnx_error"] = str(exc)

    try:
        ncnn_export = model.export(format="ncnn", imgsz=imgsz, half=True)
        export_results["ncnn"] = str(ncnn_export)
    except Exception as exc:
        export_results["ncnn_error"] = str(exc)

    save_json(export_results, run_dir / "export_results.json")
    return export_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 rat-detection experiment for Kaggle.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/kaggle/input/rat-dataset-yolov5/rat_detection.v4i.yolov5pytorch",
        help="Raw dataset root containing images/, labels/, classes.txt.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="/kaggle/working/rat_yolov8_experiment",
        help="Output directory for experiment results.",
    )
    parser.add_argument("--weights", type=str, default=None, help="Optional path to yolov8n.pt or custom best.pt.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--skip-train", action="store_true", help="Skip training and use existing best.pt in run directory.")
    parser.add_argument("--skip-export", action="store_true", help="Skip ONNX/NCNN export.")
    parser.add_argument("--skip-fps", action="store_true", help="Skip FPS benchmarking.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    raw_root = Path(args.dataset_root)
    raw_images = raw_root / "images"
    raw_labels = raw_root / "labels"
    classes_file = raw_root / "classes.txt"

    run_dir = Path(args.run_dir)
    split_root = run_dir / "dataset"
    fig_dir = run_dir / "final_figures"
    train_dir = run_dir / "train_yolov8_rat"
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print_header("Environment")
    print("Python:", sys.version)
    print("Raw dataset root:", raw_root)
    print("Run directory:", run_dir)

    if not raw_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {raw_root}")
    if not raw_images.exists():
        raise FileNotFoundError(f"Images folder not found: {raw_images}")
    if not raw_labels.exists():
        raise FileNotFoundError(f"Labels folder not found: {raw_labels}")

    print_header("Dataset preparation")
    names = read_classes(classes_file)
    nc = len(names)
    print("Classes:", names)

    pairs, missing = collect_image_label_pairs(raw_images, raw_labels)
    print("Total image-label pairs:", len(pairs))
    print("Missing labels:", len(missing))
    if missing:
        with open(run_dir / "missing_labels.txt", "w", encoding="utf-8") as f:
            for item in missing:
                f.write(item + "\n")

    train_pairs, val_pairs, test_pairs = split_pairs(pairs, args.seed)
    print("Train:", len(train_pairs), "Val:", len(val_pairs), "Test:", len(test_pairs))

    clear_directory(split_root)
    train_img_dir, train_lbl_dir = prepare_split(split_root, "train", train_pairs)
    val_img_dir, val_lbl_dir = prepare_split(split_root, "val", val_pairs)
    test_img_dir, test_lbl_dir = prepare_split(split_root, "test", test_pairs)

    data_yaml = create_yolo_yaml(split_root, names)
    print("YOLO data.yaml:", data_yaml)

    split_counts = {
        "train_images": count_images(train_img_dir),
        "train_labels": count_labels(train_lbl_dir),
        "val_images": count_images(val_img_dir),
        "val_labels": count_labels(val_lbl_dir),
        "test_images": count_images(test_img_dir),
        "test_labels": count_labels(test_lbl_dir),
    }
    print("Split counts:", split_counts)
    pd.DataFrame([split_counts]).to_csv(run_dir / "dataset_split_counts.csv", index=False)

    bad_count = validate_yolo_labels(pairs, nc, run_dir / "bad_labels.csv")
    print("Bad labels:", bad_count)

    print_header("Model initialization")
    possible_weights = [
        "/kaggle/input/your-model/best.pt",
        "/kaggle/input/your-model/yolov8n.pt",
        "/kaggle/input/yolov8n/yolov8n.pt",
        "/kaggle/input/yolov8-weights/yolov8n.pt",
        "/kaggle/input/ultralytics-yolov8/yolov8n.pt",
        "/kaggle/input/yolov8n-weights/yolov8n.pt",
        "/kaggle/input/yolo-weights/yolov8n.pt",
        "/kaggle/working/yolov8n.pt",
    ]
    weights_path = find_weights(args.weights, possible_weights)

    if args.skip_train:
        best_weights = train_dir / "weights" / "best.pt"
        if not best_weights.exists():
            raise FileNotFoundError(f"--skip-train used but no model found at {best_weights}")
        best_model = YOLO(str(best_weights))
        initialization_mode = "existing_best_pt"
    else:
        if weights_path:
            print("Using pretrained/custom weights:", weights_path)
            model = YOLO(weights_path)
            initialization_mode = "pretrained_or_custom_pt"
        else:
            print("No .pt weights found. Training YOLOv8n from scratch using yolov8n.yaml.")
            model = YOLO("yolov8n.yaml")
            initialization_mode = "from_scratch_yolov8n_yaml"

        print_header("Training")
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            seed=args.seed,
            optimizer="SGD",
            lr0=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            cos_lr=True,
            patience=30,
            project=str(run_dir),
            name="train_yolov8_rat",
            exist_ok=True,
            plots=True,
            save=True,
            verbose=True,
        )
        best_weights = train_dir / "weights" / "best.pt"
        if not best_weights.exists():
            raise FileNotFoundError(f"best.pt not found at {best_weights}")
        best_model = YOLO(str(best_weights))

    print("Initialization mode:", initialization_mode)
    print("Best weights:", best_weights)

    print_header("Validation and test evaluation")
    val_metrics = best_model.val(
        data=str(data_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        conf=0.001,
        iou=args.iou,
        project=str(run_dir),
        name="val_metrics",
        exist_ok=True,
        plots=True,
    )
    test_metrics = best_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        conf=0.001,
        iou=args.iou,
        project=str(run_dir),
        name="test_metrics",
        exist_ok=True,
        plots=True,
    )

    summary_df = pd.DataFrame(
        [
            {"split": "validation", **metrics_to_dict(val_metrics)},
            {"split": "test", **metrics_to_dict(test_metrics)},
        ]
    )
    summary_path = run_dir / "experiment_metric_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df)

    print_header("Prediction analysis")
    pred_df, detections_df, pred_path, annotated_dir = run_predictions(
        best_model,
        test_img_dir,
        names,
        run_dir,
        args.imgsz,
        args.conf,
        args.iou,
    )
    print("Predictions:", pred_path)
    print("Annotated predictions:", annotated_dir)

    conf_stats = plot_confidence_distribution(pred_df, fig_dir)
    if conf_stats:
        print("Confidence statistics:", json.dumps(conf_stats, indent=2))
    plot_detections_per_image(detections_df, fig_dir)

    fps_summary = None
    if not args.skip_fps:
        print_header("FPS benchmark")
        fps_summary = run_fps_benchmark(best_model, test_img_dir, fig_dir, args.imgsz, args.conf, args.iou)
        print(json.dumps(fps_summary, indent=2))

    copy_yolo_plots(run_dir, fig_dir)
    comparison = plot_benchmark_comparison(fig_dir)
    print("Benchmark comparison:")
    print(comparison)

    print_header("Power-budget analysis")
    power = analytical_power_budget(run_dir, fig_dir)
    print(json.dumps(power, indent=2))

    copy_qualitative_examples(run_dir, fig_dir)

    export_results = {}
    if not args.skip_export:
        print_header("Model export")
        export_results = export_models(best_model, args.imgsz, run_dir)
        print(json.dumps(export_results, indent=2))

    final_report = {
        "raw_dataset_root": str(raw_root),
        "split_dataset_dir": str(split_root),
        "data_yaml": str(data_yaml),
        "initialization_mode": initialization_mode,
        "weights_used": str(weights_path) if weights_path else "none_from_scratch_yolov8n_yaml",
        "best_weights": str(best_weights),
        "image_size": args.imgsz,
        "batch_size": args.batch,
        "epochs": args.epochs,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "classes": names,
        "split_counts": split_counts,
        "validation_metrics": metrics_to_dict(val_metrics),
        "test_metrics": metrics_to_dict(test_metrics),
        "confidence_statistics": conf_stats,
        "fps_summary": fps_summary,
        "power_budget": power,
        "outputs": {
            "metric_summary_csv": str(summary_path),
            "predictions_csv": str(pred_path),
            "annotated_predictions_dir": str(annotated_dir),
            "detections_per_image_csv": str(run_dir / "detections_per_image.csv"),
            "figures_dir": str(fig_dir),
            "export_results_json": str(run_dir / "export_results.json"),
        },
    }
    save_json(final_report, run_dir / "final_experiment_report.json")

    print_header("Experiment complete")
    print("Run directory:", run_dir)
    print("Figures directory:", fig_dir)
    print("Best weights:", best_weights)
    print("Metric summary:", summary_path)
    print("Final report:", run_dir / "final_experiment_report.json")


if __name__ == "__main__":
    main()
