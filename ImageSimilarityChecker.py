import cv2
from colorama import Fore
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
score=0.80
class ImageSimilarityChecker:
    def __init__(self, model_name='resnet18'):
        self.model = self.load_model(model_name)
        self.preprocess = self.get_preprocess_transform()

    def load_model(self, model_name):
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            model.eval()
            return model
        else:
            raise ValueError("Invalid model name. Supported models: 'resnet18'")

    def get_preprocess_transform(self):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess

    def get_feature_vector(self, image_tensor):
        with torch.no_grad():
            feature_vector = self.model(image_tensor)
        return feature_vector.squeeze().numpy()

    def similarity(self, feature_vector1, feature_vector2):
        similarity_score = cosine_similarity([feature_vector1], [feature_vector2])[0][0]
        return similarity_score

    def crop(self, image_tensor, x1, y1, x2, y2):
        cropped_image = image_tensor[:, y1:y2, x1:x2]
        return cropped_image

    def control(self, x1, y1, x2, y2, class_id, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0)
        cropped_image_tensor = self.crop(image_tensor, x1, y1, x2, y2)

        ref_image_path = 'content/UAP.jpg' if class_id == 2 else 'content/UAI.jpg'
        ref_image = Image.open(ref_image_path).convert('RGB')
        ref_image_tensor = self.preprocess(ref_image).unsqueeze(0)

        ref_feature_vector = self.get_feature_vector(ref_image_tensor)
        compared_feature_vector = self.get_feature_vector(cropped_image_tensor)

        similarity_score = self.similarity(ref_feature_vector, compared_feature_vector)

        if class_id == 2:
            print(Fore.CYAN)
            print(f"UAP Similarity score: {similarity_score}")
            if similarity_score > score:
                image.show()
                print(Fore.GREEN,"UAP İNİLEBİLİR")
                return True
        else:
            print(Fore.LIGHTRED_EX)
            print(f"UAI Similarity score: {similarity_score}")
            if similarity_score > score:
                image.show()
                print(Fore.GREEN, "UAI İNİLEBİLİR")
                return True
        return False
