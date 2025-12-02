def evaluate_cnn_vit_model(model_path, test_loader, device, model_name="CNN-ViT Hybrid"):
    """
    Evaluate CNN-ViT hybrid model and save each metric plot separately.
    """
    # Load model
    model = CNNViTHybrid(num_classes=2, heads=12, depth=12, embed_dim=768)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Collect predictions
    all_predictions, all_probabilities, all_labels = [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(target.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)[:, 1]

    # Compute metrics
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)

    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal','Pneumonia'],
                yticklabels=['Normal','Pneumonia'])
    plt.title(f'{model_name} – Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=150)
    plt.close()

    # 2. ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
    plt.title(f'{model_name} – ROC Curve')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right'); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model_name}_roc_curve.png', dpi=150)
    plt.close()

    # 3. Precision–Recall Curve
    plt.figure(figsize=(6, 5))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2,
             label=f'AUC = {pr_auc:.4f}')
    plt.title(f'{model_name} – Precision–Recall Curve')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model_name}_pr_curve.png', dpi=150)
    plt.close()

    # 4. F1 vs Threshold
    thresholds = np.arange(0.1, 1.0, 0.05)
    f1_scores = [(y_prob >= t).astype(int) for t in thresholds]
    f1_scores = [f1_score(y_true, preds) for preds in f1_scores]
    best_idx = np.argmax(f1_scores)
    best_thr, best_f1 = thresholds[best_idx], f1_scores[best_idx]
    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, f1_scores, 'g-', lw=2)
    plt.axvline(best_thr, color='red', linestyle='--',
                label=f'Best thr = {best_thr:.2f}')
    plt.title(f'{model_name} – F1 vs Threshold')
    plt.xlabel('Threshold'); plt.ylabel('F1 Score')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model_name}_f1_threshold.png', dpi=150)
    plt.close()

    # 5. Class Distribution
    unique, counts = np.unique(y_true, return_counts=True)
    plt.figure(figsize=(6, 5))
    bars = plt.bar(['Normal','Pneumonia'], counts, color=['skyblue','salmon'], alpha=0.7)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x()+bar.get_width()/2, count+1, str(count),
                 ha='center', va='bottom')
    plt.title(f'{model_name} – Class Distribution')
    plt.xlabel('Class'); plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{model_name}_class_distribution.png', dpi=150)
    plt.close()

    # 6. Metrics Bar Chart
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1-Score': f1_score(y_true, y_pred, average='weighted'),
        'ROC AUC': roc_auc
    }
    plt.figure(figsize=(6, 5))
    names = list(metrics.keys()); values = list(metrics.values())
    bars = plt.bar(names, values, color=['gold','lightgreen','lightblue','violet','orange'], alpha=0.7)
    plt.ylim(0,1); plt.title(f'{model_name} – Summary Metrics')
    plt.xticks(rotation=45)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x()+bar.get_width()/2, val+0.02,
                 f'{val:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f'{model_name}_summary_metrics.png', dpi=150)
    plt.close()

    # Print metrics summary
    print(f"\n{model_name} evaluation complete. Plots saved separately.")
    print(f"ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f} | Best F1: {best_f1:.4f} at {best_thr:.2f}")

    return metrics



def compare_models(model_paths, model_names, test_loader, device):
    results = {}

    for path, name in zip(model_paths, model_names):
        print(f"\n{'='*60}")
        print(f"EVALUATING: {name}")
        print(f"{'='*60}")
        results[name] = evaluate_cnn_vit_model(path, test_loader, device, name)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        name: [
            results[name]['accuracy'],
            results[name]['precision'], 
            results[name]['recall'],
            results[name]['f1_score'],
            results[name]['roc_auc']
        ] for name in model_names
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'])

    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(comparison_df.round(4))

    # Determine best model
    best_model = comparison_df.loc['F1-Score'].idxmax()
    print(f"\nRECOMMENDED MODEL: {best_model}")
    print(f"Best F1-Score: {comparison_df.loc['F1-Score', best_model]:.4f}")

    return results, comparison_df
