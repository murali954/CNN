import torch
import json
import numpy as np
from models.resnet_yolo import ResNetYOLO
from models.utils import non_max_suppression
from utils.coco_dataset import CocoYoloDataset
from torch.utils.data import DataLoader
from collections import defaultdict
import time


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_ap(precisions, recalls):
    """Calculate Average Precision (AP) using the precision-recall curve."""
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    precisions = np.array(precisions)[sorted_indices]
    recalls = np.array(recalls)[sorted_indices]
    
    # Calculate AP using trapezoidal rule
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i-1]) * precisions[i]
    
    return ap


def evaluate_detections(predictions, ground_truths, iou_threshold=0.5):
    """Evaluate predictions against ground truth annotations."""
    all_precisions = []
    all_recalls = []
    class_aps = defaultdict(list)
    
    for pred, gt in zip(predictions, ground_truths):
        if len(pred) == 0 and len(gt) == 0:
            continue
            
        if len(gt) == 0:
            # False positives only
            precision = 0.0
            recall = 0.0
        elif len(pred) == 0:
            # False negatives only
            precision = 0.0
            recall = 0.0
        else:
            # Calculate matches
            matched_gt = set()
            tp = 0
            
            for pred_box in pred:
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = calculate_iou(pred_box[:4], gt_box[:4])
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx != -1:
                    matched_gt.add(best_gt_idx)
                    tp += 1
            
            fp = len(pred) - tp
            fn = len(gt) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        all_precisions.append(precision)
        all_recalls.append(recall)
    
    return all_precisions, all_recalls


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = ResNetYOLO(num_classes=80)
    try:
        model.load_state_dict(torch.load("models/model_weights.pth", map_location=device))
        print("Model weights loaded successfully")
    except FileNotFoundError:
        print("Warning: Model weights not found, using random initialization")
    
    model.to(device).eval()
    
    # Load validation dataset
    val_dataset = CocoYoloDataset("dataset/valid/images", "dataset/valid/labels")
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=val_dataset.collate_fn)
    
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Evaluation metrics
    all_predictions = []
    all_ground_truths = []
    total_inference_time = 0.0
    num_batches = 0
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(val_loader):
            imgs = imgs.to(device)
            
            # Measure inference time
            start_time = time.time()
            preds = model(imgs)
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Apply NMS to predictions
            batch_predictions = []
            for i in range(preds.shape[0]):
                # Assuming preds format: [batch, num_predictions, 6] where 6 = [x1, y1, x2, y2, conf, class]
                pred_boxes = non_max_suppression(preds[i:i+1], conf_thres=0.5, iou_thres=0.45)
                if len(pred_boxes) > 0:
                    batch_predictions.append(pred_boxes[0].cpu().numpy())
                else:
                    batch_predictions.append(np.array([]))
            
            # Process ground truth
            batch_ground_truths = []
            for target in targets:
                if target is not None and len(target) > 0:
                    # Convert YOLO format to [x1, y1, x2, y2, class] if needed
                    gt_boxes = target.cpu().numpy()
                    batch_ground_truths.append(gt_boxes)
                else:
                    batch_ground_truths.append(np.array([]))
            
            all_predictions.extend(batch_predictions)
            all_ground_truths.extend(batch_ground_truths)
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(val_loader)}")
    
    print("Calculating metrics...")
    
    # Calculate evaluation metrics
    precisions, recalls = evaluate_detections(all_predictions, all_ground_truths)
    
    # Calculate summary statistics
    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0.0
    
    # Calculate mAP (simplified version)
    if precisions and recalls:
        mAP = calculate_ap(precisions, recalls)
    else:
        mAP = 0.0
    
    # Calculate average inference time
    avg_inference_time = total_inference_time / num_batches if num_batches > 0 else 0.0
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
    
    # Count statistics
    total_predictions = sum(len(pred) for pred in all_predictions)
    total_ground_truths = sum(len(gt) for gt in all_ground_truths)
    
    # Compile metrics
    metrics = {
        "evaluation_summary": {
            "dataset_size": len(val_dataset),
            "total_batches": num_batches,
            "total_predictions": int(total_predictions),
            "total_ground_truths": int(total_ground_truths)
        },
        "performance_metrics": {
            "mean_precision": float(mean_precision),
            "mean_recall": float(mean_recall),
            "f1_score": float(f1_score),
            "mAP": float(mAP)
        },
        "inference_metrics": {
            "average_inference_time_seconds": float(avg_inference_time),
            "fps": float(fps),
            "total_inference_time_seconds": float(total_inference_time)
        },
        "model_info": {
            "num_classes": 80,
            "device": str(device),
            "model_type": "ResNetYOLO"
        }
    }
    
    # Save metrics to JSON file
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("\nEvaluation Results:")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"mAP: {mAP:.4f}")
    print(f"Average Inference Time: {avg_inference_time:.4f}s")
    print(f"FPS: {fps:.2f}")
    print(f"\nMetrics saved to metrics.json")
    
    return metrics


if __name__ == "__main__":
    evaluate_model()