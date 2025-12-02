def train(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, correct = 0, 0
    for batch_idx, (x, y) in enumerate(tqdm(loader, desc="Training  ")):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)
def evaluate(model, loader, criterion, device):
    with torch.no_grad():
        model.eval()
        loss_sum, correct = 0, 0
        for batch_idx, (x, y) in enumerate(tqdm(loader, desc="Validation")):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            correct  += (out.argmax(1) == y).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)
def run_training(model, model_name, train_loader, validation_loader, device, epochs, lr):
    model.to(device)
    try:
        model.cnn.load_state_dict(torch.load(pytorch_state_dict_path), strict=False)
    except Exception as e:
        print(f"Warning: could not load pretrained cnn weights: {e}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    tr_loss_all = []
    te_loss_all = []
    tr_acc_all = []
    te_acc_all = []
    training_time = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        print(f"\nEpoch {epoch:02d}/{epochs:02d} started at {present_time()} (UTC)")
        tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, validation_loader, criterion, device)
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {te_loss:.4f} acc {te_acc:.4f} |"
            f" in  {elapsed:.02f}s"
        )

        tr_loss_all.append(tr_loss)
        te_loss_all.append(te_loss)
        tr_acc_all.append(tr_acc)
        te_acc_all.append(te_acc)
        training_time.append(elapsed)

        if te_loss < best_loss:
            print(f"Current loss ({te_loss:.04f}) lower than previous best ({best_loss:.04f}), saving {model_name}")
            best_loss = te_loss
            torch.save(model.state_dict(), model_name)

    return {
        "tr_loss": tr_loss_all,
        "te_loss": te_loss_all,
        "tr_acc": tr_acc_all,
        "te_acc": te_acc_all,
        "times": training_time
    }
def run_training_with_scheduler(model, model_name, train_loader, validation_loader, device, epochs, lr):
    """Enhanced training function with ReduceLROnPlateau scheduler"""
    model.to(device)
    
    # Load pretrained CNN weights if available
    try:
        model.cnn.load_state_dict(torch.load(pytorch_state_dict_path), strict=False)
        print("✓ Loaded pretrained CNN weights")
    except Exception as e:
        print(f"Warning: could not load pretrained cnn weights: {e}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # ReduceLROnPlateau scheduler
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                patience=3, min_lr=1e-6)

    best_loss = float('inf')
    tr_loss_all = []
    te_loss_all = []
    tr_acc_all = []
    te_acc_all = []
    training_time = []
    learning_rates = []

    print(f"📊 Initial learning rate: {lr}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        print(f"\nEpoch {epoch:02d}/{epochs:02d} started at {present_time()}")
        
        tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, validation_loader, criterion, device)
        elapsed = time.time() - start_time

        # Step the scheduler
        scheduler.step(te_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        print(
            f"Epoch {epoch:02d} | LR: {current_lr:.6f} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {te_loss:.4f} acc {te_acc:.4f} | "
            f"time {elapsed:.02f}s"
        )

        # Check for learning rate reduction
        if len(learning_rates) > 1 and current_lr < learning_rates[-2]:
            print(f"📉 Learning rate reduced to {current_lr:.6f}")

        tr_loss_all.append(tr_loss)
        te_loss_all.append(te_loss)
        tr_acc_all.append(tr_acc)
        te_acc_all.append(te_acc)
        training_time.append(elapsed)

        if te_loss < best_loss:
            print(f"✓ New best loss ({te_loss:.04f}), saving model")
            best_loss = te_loss
            torch.save(model.state_dict(), model_name)

    print(f"\n📈 Final learning rate: {current_lr:.6f}")
    lr_reductions = sum(1 for i in range(1, len(learning_rates)) if learning_rates[i] < learning_rates[i-1])
    print(f"📊 Learning rate reductions: {lr_reductions}")

    return {
        "tr_loss": tr_loss_all,
        "te_loss": te_loss_all,
        "tr_acc": tr_acc_all,
        "te_acc": te_acc_all,
        "times": training_time,
        "learning_rates": learning_rates
    }
