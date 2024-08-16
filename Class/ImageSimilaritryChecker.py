from colorama import Fore
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import ResNet50_Weights, EfficientNet_B7_Weights
import numpy as np

score = 0.637
class ImageSimilarityChecker:
    def __init__(self, model_name='efficientnet-b7'):
        self.model = self.load_model(model_name)
        self.preprocess = self.get_preprocess_transform()

    def load_model(self, model_name):
        if model_name == 'resnet50':
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
        elif model_name == 'efficientnet-b7':
            model = models.efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
            model = torch.nn.Sequential(*(list(model.children())[:-2]))  # EfficientNet-B7 için son iki katmanı çıkarın
        else:
            raise ValueError("Invalid model name. Supported models: 'resnet50', 'efficientnet-b7'")

        model.eval()
        return model

    def get_preprocess_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_feature_vector(self, image):
        try:
            image_tensor = self.preprocess(image)
            image_tensor = image_tensor.unsqueeze(0)

            with torch.no_grad():
                feature_vector = self.model(image_tensor)
                feature_vector = feature_vector.view(feature_vector.size(0), -1)  # Vektörü düzleştir

            return feature_vector.squeeze().numpy()
        except Exception as e:
            print(Fore.RED + f"Error in extracting feature vector: {e}")
            return None

    def similarity(self, feature_vector1, feature_vector2):
        try:
            if feature_vector1.shape != feature_vector2.shape:
                print(Fore.RED + "Feature vectors do not match in shape.")
                return 0
            similarity_score = cosine_similarity([feature_vector1], [feature_vector2])[0][0]
            return similarity_score
        except Exception as e:
            print(Fore.RED + f"Error in calculating similarity: {e}")
            return 0

    def crop(self, image_path, x1, y1, x2, y2):
        try:
            image = Image.open(image_path)
            cropped_image = image.crop((x1, y1, x2, y2))
            return cropped_image
        except Exception as e:
            print(Fore.RED + f"Error in cropping image: {e}")
            return None

    def control(self, x1, y1, x2, y2, class_id, image_path):
        binary_image = self.crop(image_path, x1, y1, x2, y2)
        if binary_image is None:
            return False

        ref_image_path = 'content/UAP1.jpg' if class_id == 2 else 'content/UA1.jpg'
        ref_image = Image.open(ref_image_path)

        ref_feature_vector = self.get_feature_vector(ref_image)
        compared_feature_vector = self.get_feature_vector(binary_image)

        if ref_feature_vector is None or compared_feature_vector is None:
            return False

        similarity_score = self.similarity(ref_feature_vector, compared_feature_vector)
        if class_id == 2:
            print(Fore.CYAN + f"UAP Similarity score: {similarity_score}")
            if similarity_score > score:
                print(Fore.GREEN + "UAP İNİLEBİLİR")
                return True
        else:
            print(Fore.LIGHTRED_EX + f"UAI Similarity score: {similarity_score}")
            if similarity_score > score:
                print(Fore.GREEN + "UAI İNİLEBİLİR")
                return True
        return False
