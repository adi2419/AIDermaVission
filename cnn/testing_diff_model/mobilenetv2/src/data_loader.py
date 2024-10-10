import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
from PIL import Image  # type: ignore
import pandas as pd  # type: ignore
from torchvision import transforms  # type: ignore


class MultiLabelDataset(Dataset):
    def __init__(self, csv_file, root_dir, class_mapping, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_mapping = class_mapping  # Class name to numeric mapping
        
        # Group labels by image_name and apply list to store multi-labels
        self.labels = (
            self.labels
            .groupby('filename')['class']
            .apply(list)
            .reset_index()
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]  # Image name
        img_path = f"{self.root_dir}/{img_name}"
        image = Image.open(img_path).convert('RGB')   
        # Get class names for the image
        label_names = self.labels.iloc[idx, 1]
        # Initialize a zero vector for multi-label classification
        label_vector = [0] * len(self.class_mapping)
        # Map class names to their corresponding index in the label vector
        for label in label_names:
            if label in self.class_mapping:
                label_vector[self.class_mapping[label]] = 1

        # Convert label_vector to a float tensor
        labels = torch.tensor(label_vector).float()
        if self.transform:
            image = self.transform(image)

        return image, labels


def get_dataloaders(data_dir, batch_size=32):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Define class mapping: adjust based on your classes
    class_mapping = {
        'hyperpigmentation_1': 0,
        'hyperpigmentation_2': 1,
        'hyperpigmentation_3': 2,
        'hyperpigmentation_4': 3,
        'pustules': 4
    }

    train_dataset = MultiLabelDataset(
        csv_file=f"{data_dir}/labels/train_labels_long.csv",
        root_dir=f"{data_dir}/train",
        class_mapping=class_mapping,
        transform=data_transforms['train']
    )
    val_dataset = MultiLabelDataset(
        csv_file=f"{data_dir}/labels/val_labels_long.csv",
        root_dir=f"{data_dir}/val",
        class_mapping=class_mapping,
        transform=data_transforms['val']
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
