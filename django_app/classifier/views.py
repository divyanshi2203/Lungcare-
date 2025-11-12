# classifier/views.py
from django.shortcuts import render
from .forms import ImageUploadForm  # form for image upload (we'll create this in a moment)
import torch
from PIL import Image
from io import BytesIO
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

# ==== Model Loading (run once) ====
# Ensure the CancerNet class is defined (paste the class definition here or import if defined elsewhere)
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

                                                  
class CancerNet(nn.Module):
    def __init__(self):
        super(CancerNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*13*13, 128)
        self.fc2 = nn.Linear(128, 2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)); x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)); x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate and load weights
MODEL_PATH = "classifier/cancer_detector.pth"  # path to the saved model file
model = CancerNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()  # set to evaluation mode


@csrf_exempt
def chat_view(request):
    """
    Handles AJAX requests for the chat interface.
    """
    if request.method == 'GET':
        # Initial page render
        return render(request, 'classifier/chat.html')

    elif request.method == 'POST':
        # Handle AJAX chat requests
        if 'image' in request.FILES:
            img_file = request.FILES['image']
            img_bytes = img_file.read()
            from PIL import Image
            from io import BytesIO
            import base64
            import torchvision.transforms as transforms

            try:
                image = Image.open(BytesIO(img_bytes))
            except Exception as e:
                return JsonResponse({'error': f'Error opening image: {e}'}, status=400)

            # Preprocess image
            image = image.resize((64, 64))
            image_tensor = transforms.ToTensor()(image)
            image_tensor = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])(image_tensor)
            image_tensor = image_tensor.unsqueeze(0)

            # Prediction
            outputs = model(image_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = 'cancer' if predicted_idx.item() == 0 else 'normal'

            if predicted_class == 'cancer':
                doctors = [
                    {"name": "Dr. Alice Brown", "link": "https://dummydoctor1.com"},
                    {"name": "Dr. Rahul Mehta", "link": "https://dummydoctor2.com"},
                    {"name": "Dr. Emma Chen", "link": "https://dummydoctor3.com"},
                ]
                return JsonResponse({
                    'prediction': 'cancer',
                    'message': "Our model detected signs of cancer. Here are some doctors you can consult:",
                    'doctors': doctors
                })
            else:
                # No cancer found — ask for symptom type
                return JsonResponse({
                    'prediction': 'normal',
                    'message': "No cancer detected. Please select your symptom type (1, 2, or 3).",
                    'symptoms': {
                        1: "Fever, headache, fatigue",
                        2: "Shortness of breath, wheezing, chest tightness",
                        3: "Runny nose, sore throat, mild cough"
                    }
                })

        # Handle symptom selection
        else:
            import json
            data = json.loads(request.body.decode('utf-8'))
            symptom_choice = data.get('symptom_choice')

            if symptom_choice == '1':
                illness = 'COVID-like symptoms'
                doctors = [
                    {"name": "Dr. Mark Patel", "link": "https://covid-doctor1.com"},
                    {"name": "Dr. Sarah Li", "link": "https://covid-doctor2.com"},
                    {"name": "Dr. John Kim", "link": "https://covid-doctor3.com"},
                ]
            elif symptom_choice == '2':
                illness = 'Asthma-like symptoms'
                doctors = [
                    {"name": "Dr. Priya Nair", "link": "https://asthma-doctor1.com"},
                    {"name": "Dr. Leo White", "link": "https://asthma-doctor2.com"},
                    {"name": "Dr. Maria Singh", "link": "https://asthma-doctor3.com"},
                ]
            else:
                illness = 'Common Cold-like symptoms'
                doctors = [
                    {"name": "Dr. James Roy", "link": "https://cold-doctor1.com"},
                    {"name": "Dr. Aisha Khan", "link": "https://cold-doctor2.com"},
                    {"name": "Dr. Ben Wilson", "link": "https://cold-doctor3.com"},
                ]

            return JsonResponse({
                'prediction': 'symptom_followup',
                'message': f"Based on your symptoms, it might be {illness}. Here are some doctors you can consult:",
                'doctors': doctors
            })
