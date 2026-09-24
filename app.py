import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Model Configuration (must match training) ---
target_size = (250, 250)
class_names = ['Cap', 'Helmet'] # Hardcoded from full_dataset.classes

# Define the model architecture (must be identical to the one used for training)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 62 * 62, 128), # Calculated as 250 / 2 / 2 = 62.5 -> 62
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the trained model
model = SimpleCNN(num_classes=len(class_names))
model_save_path = 'simple_cnn_model.pth' # Assuming the model is in the same directory as app.py
model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu'))) # Load to CPU
model.eval() # Set to evaluation mode

# Define the preprocessing transformations (must be identical to testing)
preprocess = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return "Image Classification API. Send a POST request to /predict with an image file."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            image = Image.open(file.stream).convert('RGB')
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model

            with torch.no_grad():
                output = model(input_batch)

            _, predicted_idx = torch.max(output, 1)
            predicted_class = class_names[predicted_idx.item()]

            return jsonify({'prediction': predicted_class}), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For local development, you might use app.run(debug=True)
    # For deployment, listen on all public IPs
    app.run(host='0.0.0.0', port=5000)
