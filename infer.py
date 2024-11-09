
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

class BuildingClassifier:
    def __init__(self, model_path, class_names=None):
        """
        Initialize the classifier with a pre-trained model and class names.
        
        Args:
            model_path (str): Path to the saved model weights
            class_names (list): List of class names (building names)
        """
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._create_model(num_classes=len(class_names) if class_names else 10)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Store class names
        self.class_names = class_names if class_names else []
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize(580),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _create_model(self, num_classes):
        """Create a ResNet50 model with custom number of classes"""
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model

    def predict_image(self, image_path):
        """
        Predict the class of a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (predicted_class_name, confidence_score)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class_idx = predicted.item()
                confidence_score = confidence.item()
                
                if self.class_names:
                    predicted_class = self.class_names[predicted_class_idx]
                else:
                    predicted_class = f"Class_{predicted_class_idx}"
                
                return predicted_class, confidence_score
                
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None, 0.0

    def predict_folder(self, folder_path):
        """
        Predict classes for all images in a folder.
        
        Args:
            folder_path (str): Path to the folder containing images
            
        Returns:
            dict: Dictionary of filename to (predicted_class, confidence) pairs
        """
        results = {}
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                image_path = os.path.join(folder_path, filename)
                predicted_class, confidence = self.predict_image(image_path)
                results[filename] = (predicted_class, confidence)
        
        return results

# Example usage
if __name__ == "__main__":
    # Define your class names (buildings)
    CLASS_NAMES = [
        "Butler Hall",
        "Carpenter Hall",
        "Lee Hall",
        "McCain Hall",
        "McCool Hall",
        "Old Main",
        "Simrall Hall",
        "Student Union",
        "Swalm Hall",
        "Walker Hall"
    ]




    
    # Initialize the classifier
    classifier = BuildingClassifier(
        model_path="best_model.pth",   # The path for the pretrained weights should be placed here
        class_names=CLASS_NAMES
    )
    
    # Example: Predict a single image
    image_path = "eg_img.png"
    predicted_class, confidence = classifier.predict_image(image_path)
    print(f"\nSingle image prediction:")
    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    

    # Example: Predict all images in a folder
    folder_path = "pred_folder"
    results = classifier.predict_folder(folder_path)

    print("\nFolder predictions:")
    for filename, (predicted_class, confidence) in results.items():
        print(f"File: {filename}")
        print(f"Predicted class: {predicted_class}")
        # print(f"Confidence: {confidence:.2%}")
        print("-" * 50)