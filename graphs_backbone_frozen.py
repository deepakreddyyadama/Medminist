import matplotlib.pyplot as plt
#frozen run

# Epoch numbers
epochs = [1, 2, 3, 4, 5]

# Training loss per epoch
train_loss = [0.5508, 0.3587, 0.3333, 0.3170, 0.3099]

# Training accuracy per epoch
train_acc = [0.8219, 0.8759, 0.8832, 0.8886, 0.8897]

# Validation accuracy per epoch
val_acc   = [0.8764, 0.8854, 0.8894, 0.8949, 0.8997]

# Best validation accuracy and final test accuracy
best_val_acc = 0.8997
test_acc     = 0.8692

# 1) Train vs Validation Accuracy graph

plt.figure(figsize=(6, 4))
plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("PathMNIST – Frozen Backbone ResNet18\nTrain vs Validation Accuracy")
plt.ylim(0.8, 1.0)
plt.xticks(epochs)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("frozen_train_val_accuracy.png", dpi=300)
plt.close()

# 2) Training Loss 

plt.figure(figsize=(6, 4))
plt.plot(epochs, train_loss, marker="o")

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("PathMNIST – Frozen Backbone ResNet18\nTraining Loss per Epoch")
plt.xticks(epochs)
plt.grid(True)
plt.tight_layout()
plt.savefig("frozen_train_loss.png", dpi=300)
plt.close()


# 3) Best Validation vs Test Accuracy 

labels = ["Best Val", "Test"]
values = [best_val_acc, test_acc]

plt.figure(figsize=(5, 4))
plt.bar(labels, values)

plt.ylim(0.8, 1.0)
plt.ylabel("Accuracy")
plt.title("PathMNIST – Frozen Backbone ResNet18\nBest Validation vs Test Accuracy")

for i, v in enumerate(values):
    plt.text(i, v + 0.005, f"{v:.4f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("frozen_val_vs_test_accuracy.png", dpi=300)
plt.close()

print("Saved graphs: frozen_train_val_accuracy.png, frozen_train_loss.png, frozen_val_vs_test_accuracy.png")
