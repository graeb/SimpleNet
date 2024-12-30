import cv2
import numpy as np
from sklearn import metrics
from skimage import measure


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )

    precision, recall, _ = metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auc_pr = metrics.auc(recall, precision)

    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


def compute_pro(masks, amaps, num_th=200):
    """
    Compute the PRO (Per-Region Overlap) metric and the AUC (Area Under Curve) for a given set of masks and anomaly maps.

    Args:
        masks (np.ndarray): Ground truth binary masks, shape (N, H, W).
        amaps (np.ndarray): Anomaly maps, shape (N, H, W).
        num_th (int): Number of thresholds to evaluate.

    Returns:
        float: The PRO-AUC score.
    """
    # DataFrame for results
    df = {"pro": [], "fpr": [], "threshold": []}

    # Threshold range
    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    # Iterate over thresholds
    for th in np.arange(min_th, max_th, delta):
        # Threshold anomaly maps
        binary_amaps = np.zeros_like(amaps, dtype=bool)
        binary_amaps[amaps > th] = 1

        # Calculate PRO
        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        # Calculate FPR
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        # Append results
        df["pro"].append(np.mean(pros))
        df["fpr"].append(fpr)
        df["threshold"].append(th)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df["fpr"] = np.clip(df["fpr"], 0, 0.3)
    df["fpr"] = df["fpr"] / max(df["fpr"])

    # Sort by FPR for AUC computation
    sorted_indices = np.argsort(df["fpr"])
    fpr_sorted = np.array(df["fpr"])[sorted_indices]
    pro_sorted = np.array(df["pro"])[sorted_indices]

    # Compute AUC
    pro_auc = metrics.auc(fpr_sorted, pro_sorted)
    return pro_auc
