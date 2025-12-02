img_size = 64
batch_size = 128
lr = 0.0005
num_cls = 2
pytorch_state_dict_path = r"C:\classification.pth"
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device used is {device}")

    train_transform, validation_transform, test_transform = make_transforms(img_size)

    train_path = r"C:\Chestxrayvision\chest_xray\train"
    test_path = r"C:\Chestxrayvision\chest_xray\test"

    train_loader, validation_loader, test_loader = make_dataloaders(
        train_path, test_path, train_transform, validation_transform, test_transform, batch_size
    )

    # Global ViT parameters
    attn_heads = 12
    depth = 12
    embed_dim = 768

    print(f"Configuration | batch:{batch_size} | attn_heads:{attn_heads} | depth:{depth} | embed_dim:{embed_dim}")

    # Short run: Optimal discovery LR
    lr_short = 0.002

    # Long run: Fine-tuning LR with adaptive reduction
    lr_long = 0.0015  # + ReduceLROnPlateau scheduler

    # Short training run with fixed higher learning rate
    epochs_short = 10
    model_name_short = "ai_model_state_dict.pth"
    model = CNNViTHybrid(num_classes=num_cls, heads=attn_heads, depth=depth, embed_dim=embed_dim)
    print(f"\n🚀 Starting short training: {epochs_short} epochs, LR={lr_short}")
    results_short = run_training(model, model_name_short, train_loader, validation_loader, device, epochs_short, lr_short)
    plot_metrics(results_short, title_prefix=f"Short_Run_LR_{lr_short}")

    # Long training run with adaptive learning rate scheduling
    epochs_long = 20
    model_name_long = "vit_model_test_state_dict.pth"
    model_test = CNNViTHybrid(num_classes=num_cls, heads=attn_heads, depth=depth, embed_dim=embed_dim)
    print(f"\n⏳ Starting long training: {epochs_long} epochs, LR={lr_long} + ReduceLROnPlateau")
    results_long = run_training_with_scheduler(model_test, model_name_long, train_loader, validation_loader, device, epochs_long, lr_long)
    plot_metrics(results_long, title_prefix=f"Long_Run_LR_{lr_long}_Scheduled")

    # Compare results
    print("\n" + "="*60)
    print("TRAINING COMPARISON")
    print("="*60)
    print(f"Short run (LR={lr_short}): Best val acc = {max(results_short['te_acc']):.4f}")
    print(f"Long run (LR={lr_long} + scheduler): Best val acc = {max(results_long['te_acc']):.4f}")
    
    improvement = max(results_long['te_acc']) - max(results_short['te_acc'])
    print(f"Improvement with longer training: {improvement:.4f} ({improvement*100:.2f}%)")

    print("\nAll training complete.")

    # Evaluation section
    print("\n" + "="*60)
    print("STARTING MODEL EVALUATION")
    print("="*60)
    
    short_results = evaluate_cnn_vit_model(
        model_name_short, test_loader, device, f"Short_Training_LR_{lr_short}"
    )
    
    long_results = evaluate_cnn_vit_model(
        model_name_long, test_loader, device, f"Long_Training_LR_{lr_long}_Scheduled"
    )
if __name__ == "__main__":
    multiprocessing.freeze_support()  
    set_seed(42)                      
    main()
