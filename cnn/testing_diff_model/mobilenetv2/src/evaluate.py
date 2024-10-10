import torch   # type: ignore
from data_loader import get_dataloaders
from model import load_model
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(data_dir, model_path, num_classes=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(num_classes, model_path).to(device)
    _, val_loader = get_dataloaders(data_dir)

    model.eval()
    all_labels = []
    all_preds = []
    threshold = 0.5

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (outputs > threshold).float()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Compute precision, recall, f1-score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro')

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Calculate accuracy
    accuracy = np.sum(all_labels == all_preds) / all_labels.size
    print(f'Accuracy: {accuracy:.4f}')

    # Plot ROC curves for each class
    plot_roc_curve(all_labels, all_preds, num_classes)


def plot_roc_curve(all_labels, all_preds, num_classes):
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'Class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multi-label Classification')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    data_dir = ( 
        '/Users/adityaravi/Desktop/project'
        '/AIDermaVision-MAJOR/implement/cnn'
        '/testing_diff_model/mobilenetv2/dataset'
        )
    model_path = 'mobilenetv2_multilabel.pth'
    evaluate_model(data_dir, model_path)
