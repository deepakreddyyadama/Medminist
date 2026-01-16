import matplotlib.pyplot as plt



# Epochs
epochs = [1, 2, 3, 4, 5]

# Training accuracy per epoch
train_acc = [0.9332, 0.9690, 0.9769, 0.9815, 0.9838]

# Validation accuracy per epoch
val_acc   = [0.9451, 0.9557, 0.9761, 0.9606, 0.9695]

# Training loss per epoch
train_loss = [0.1960, 0.0915, 0.0673, 0.0541, 0.0482]

# Best validation and final test accuracy
best_val_acc = 0.9761
test_acc     = 0.8613

#accuracy
plt.figure(figsize=(6, 4))
plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("PathMNIST – Fine-Tuned ResNet18\nTrain vs Validation Accuracy")
plt.ylim(0.8, 1.0)
plt.xticks(epochs)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("train_val_accuracy_curve.png", dpi=300)
plt.close()
#loss curve
plt.figure(figsize=(6, 4))
plt.plot(epochs, train_loss, marker="o")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PathMNIST – Fine-Tuned ResNet18\nTraining Loss per Epoch")
plt.xticks(epochs)
plt.grid(True)
plt.tight_layout()
plt.savefig("train_loss_curve.png", dpi=300)
plt.close()

# 3) Best Val vs Test accuracy

labels = ["Best Val", "Test"]
values = [best_val_acc, test_acc]

plt.figure(figsize=(5, 4))
plt.bar(labels, values)

plt.ylim(0.8, 1.0)
plt.ylabel("Accuracy")
plt.title("PathMNIST – Fine-Tuned ResNet18\nBest Validation vs Test Accuracy")

for i, v in enumerate(values):
    plt.text(i, v + 0.005, f"{v:.4f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("val_vs_test_accuracy.png", dpi=300)
plt.close()

print("Graphs saved as PNG files in this folder.")
