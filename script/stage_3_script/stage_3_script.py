import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from stage_3_code.script_data_loader import load_data  # This function must load each dataset properly


datasets = ['ORL', 'MNIST', 'CIFAR']

for name in datasets:
    print(f"\n--- Running on {name} Dataset ---")
    X_train, y_train, X_test, y_test, input_channels, num_classes = load_data(name)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = StrongCNN(input_channels, num_classes)
    print(f"{name} label values — min: {y_train.min()}, max: {y_train.max()}, num_classes: {num_classes}")

    trained_model, train_losses, test_accuracies = train_model(model, train_loader, test_loader, epochs=50)

    # Plot training loss
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name} Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/content/drive/MyDrive/ECS189/ECS189G-Project/result/{name}_loss_plot.png')
    plt.show()

    # Plot test accuracy
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{name} Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/content/drive/MyDrive/ECS189/ECS189G-Project/result/{name}_test_accuracy_plot.png')
    plt.show()

    # Evaluate normal model
    report = evaluate_model(trained_model, test_loader)
    print(f"{name} Evaluation Report:")
    for metric in ["accuracy", "precision", "recall", "f1-score"]:
        avg = report["weighted avg" if metric != "accuracy" else "accuracy"]
        score = avg if metric == "accuracy" else avg[metric]
        print(f"{metric.title()}: {score:.4f}")

    # ⬇️ Paste the ABLATION STUDY here ⬇️
    print(f"\n--- Ablation: Running on {name} Dataset ---")
    ablation_model = AblationCNN(input_channels, num_classes)
    ablation_model, ablation_losses, ablation_test_accuracies = train_model(ablation_model, train_loader, test_loader, epochs=30)

    # Plot ablation loss
    plt.plot(ablation_losses)
    plt.title(f"{name} Ablation Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f'/content/drive/MyDrive/ECS189/ECS189G-Project/result/{name}_ablation_loss_plot.png')
    plt.show()

    # Evaluate ablation model
    ablation_report = evaluate_model(ablation_model, test_loader)
    print(f"{name} Ablation Evaluation Report:")
    for metric in ["accuracy", "precision", "recall", "f1-score"]:
        avg = ablation_report["weighted avg" if metric != "accuracy" else "accuracy"]
        score = avg if metric == "accuracy" else avg[metric]
        print(f"{metric.title()}: {score:.4f}")