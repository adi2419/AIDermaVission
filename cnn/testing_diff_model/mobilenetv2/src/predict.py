# type: ignore

import torch
from PIL import Image
from torchvision import transforms
from model import load_model


def predict_image(image_path, model_path, num_classes=5, threshold=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(num_classes, model_path).to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    img = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        preds = (outputs > threshold).float()

    return preds.cpu().numpy()


if __name__ == "__main__":
    image_path = ('/Users/adityaravi/Desktop/project'
                  '/AIDermaVision-MAJOR/implement/cnn'
                  '/testing_diff_model/mobilenetv2'
                  '/dataset/0aacf1_5002bc1e012c49858c816e51f504967a-mv2_jpg.rf.44320419afa0a8bec375d26ab8b647fd.jpg'
                )
    model_path = 'mobilenetv2_multilabel.pth'
    preds = predict_image(image_path, model_path)
    print(f'Predicted labels: {preds}')
