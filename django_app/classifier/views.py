# classifier/views.py
from django.shortcuts import render
from .forms import ImageUploadForm  # form for image upload (we'll create this in a moment)
import torch
from PIL import Image
from io import BytesIO
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, Http404
from django.urls import reverse 
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import redirect
from django.shortcuts import get_object_or_404
from django.core.mail import send_mail
from django.conf import settings

# ==== Model Loading (run once) ====
# Ensure the CancerNet class is defined (paste the class definition here or import if defined elsewhere)
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datetime import datetime, timedelta
from django.utils import timezone
from zoneinfo import ZoneInfo
from .models import Appointment
                                                  
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

# Demo doctor directory (in-memory, no DB for now)
DOCTORS = {
    "alice-brown": {
        "slug": "alice-brown",
        "name": "Dr. Alice Brown",
        "specialization": "Oncologist – Lung Cancer Specialist",
        "hospital": "City Care Hospital",
        "location": "New Delhi, India",
        "experience": "10+ years in thoracic oncology and lung cancer screening.",
        "about": "Dr. Brown focuses on early detection of lung tumors and minimally invasive treatments.",
        "phone": "+91-98765-00001",
        "email": "alice.brown@example.com",
        "consulting_hours": "Mon–Fri · 10:00 AM – 4:00 PM",
    },
    "rahul-mehta": {
        "slug": "rahul-mehta",
        "name": "Dr. Rahul Mehta",
        "specialization": "Pulmonologist",
        "hospital": "Green Valley Medical Center",
        "location": "Mumbai, India",
        "experience": "8+ years treating chronic lung conditions and post-COVID lung issues.",
        "about": "Dr. Mehta combines imaging and lung function tests to plan personalized treatments.",
        "phone": "+91-98765-00002",
        "email": "rahul.mehta@example.com",
        "consulting_hours": "Tue–Sat · 11:00 AM – 6:00 PM",
    },
    "emma-chen": {
        "slug": "emma-chen",
        "name": "Dr. Emma Chen",
        "specialization": "Radiologist – Chest Imaging",
        "hospital": "Sunrise Diagnostics",
        "location": "Bengaluru, India",
        "experience": "12+ years in CT & X-ray interpretation for chest and lung conditions.",
        "about": "Dr. Chen is known for detailed imaging reports and second opinions.",
        "phone": "+91-98765-00003",
        "email": "emma.chen@example.com",
        "consulting_hours": "Mon–Fri · 9:30 AM – 3:30 PM",
    },

    # COVID-like symptoms
    "mark-patel": {
        "slug": "mark-patel",
        "name": "Dr. Mark Patel",
        "specialization": "Internal Medicine",
        "hospital": "Metro Health Clinic",
        "location": "Pune, India",
        "experience": "7+ years in respiratory infections and viral fevers.",
        "about": "Dr. Patel manages COVID-like illnesses and recovery follow-up.",
        "phone": "+91-98765-00004",
        "email": "mark.patel@example.com",
        "consulting_hours": "Mon–Sat · 10:00 AM – 1:00 PM",
    },
    "sarah-li": {
        "slug": "sarah-li",
        "name": "Dr. Sarah Li",
        "specialization": "Infectious Disease Specialist",
        "hospital": "Global Care Hospital",
        "location": "Hyderabad, India",
        "experience": "9+ years in infectious diseases and complex fevers.",
        "about": "She focuses on evidence-based treatment and long-COVID care.",
        "phone": "+91-98765-00005",
        "email": "sarah.li@example.com",
        "consulting_hours": "Mon, Wed, Fri · 3:00 PM – 7:00 PM",
    },
    "john-kim": {
        "slug": "john-kim",
        "name": "Dr. John Kim",
        "specialization": "Pulmonologist",
        "hospital": "North Star Hospital",
        "location": "Chennai, India",
        "experience": "11+ years in acute respiratory infections and ICU care.",
        "about": "Dr. Kim handles severe viral pneumonia and breathing complications.",
        "phone": "+91-98765-00006",
        "email": "john.kim@example.com",
        "consulting_hours": "Tue–Sat · 9:00 AM – 1:00 PM",
    },

    # Asthma-like
    "priya-nair": {
        "slug": "priya-nair",
        "name": "Dr. Priya Nair",
        "specialization": "Pulmonologist – Asthma & Allergy",
        "hospital": "Airway Care Center",
        "location": "Kochi, India",
        "experience": "10+ years in asthma control and allergy management.",
        "about": "She designs long-term asthma action plans and inhaler training.",
        "phone": "+91-98765-00007",
        "email": "priya.nair@example.com",
        "consulting_hours": "Mon–Fri · 4:00 PM – 8:00 PM",
    },
    "leo-white": {
        "slug": "leo-white",
        "name": "Dr. Leo White",
        "specialization": "Allergy & Immunology",
        "hospital": "Clear Breath Clinic",
        "location": "Jaipur, India",
        "experience": "6+ years in allergy testing and asthma triggers.",
        "about": "Dr. White helps patients identify and avoid asthma triggers.",
        "phone": "+91-98765-00008",
        "email": "leo.white@example.com",
        "consulting_hours": "Tue, Thu, Sat · 10:00 AM – 2:00 PM",
    },
    "maria-singh": {
        "slug": "maria-singh",
        "name": "Dr. Maria Singh",
        "specialization": "Pediatric Pulmonologist",
        "hospital": "Children’s Chest Center",
        "location": "Lucknow, India",
        "experience": "9+ years treating asthma in children.",
        "about": "She focuses on child-friendly asthma care and parent education.",
        "phone": "+91-98765-00009",
        "email": "maria.singh@example.com",
        "consulting_hours": "Mon–Fri · 11:00 AM – 5:00 PM",
    },

    # Common cold-like
    "james-roy": {
        "slug": "james-roy",
        "name": "Dr. James Roy",
        "specialization": "General Physician",
        "hospital": "Neighborhood Clinic",
        "location": "Bhopal, India",
        "experience": "5+ years in primary care and common illnesses.",
        "about": "Dr. Roy treats colds, flu, and routine health problems.",
        "phone": "+91-98765-00010",
        "email": "james.roy@example.com",
        "consulting_hours": "Mon–Sat · 9:00 AM – 12:30 PM",
    },
    "aisha-khan": {
        "slug": "aisha-khan",
        "name": "Dr. Aisha Khan",
        "specialization": "ENT Specialist",
        "hospital": "Ear Nose Throat Center",
        "location": "Delhi, India",
        "experience": "8+ years in throat, nose, and sinus problems.",
        "about": "Dr. Khan manages sore throat, blocked nose, and sinusitis.",
        "phone": "+91-98765-00011",
        "email": "aisha.khan@example.com",
        "consulting_hours": "Tue–Sat · 3:00 PM – 7:00 PM",
    },
    "ben-wilson": {
        "slug": "ben-wilson",
        "name": "Dr. Ben Wilson",
        "specialization": "Family Physician",
        "hospital": "Family Health Clinic",
        "location": "Nagpur, India",
        "experience": "7+ years in family and seasonal illness care.",
        "about": "He focuses on supportive care and lifestyle advice.",
        "phone": "+91-98765-00012",
        "email": "ben.wilson@example.com",
        "consulting_hours": "Mon–Fri · 5:00 PM – 9:00 PM",
    },
}


@csrf_exempt
@login_required
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
                cancer_slugs = ["alice-brown", "rahul-mehta", "emma-chen"]
                doctors = []
                for slug in cancer_slugs:
                    doc = DOCTORS[slug]
                    doctors.append({
                        "name": doc["name"],
                        "link": reverse("classifier:doctor_detail", kwargs={"slug": slug}),  # ← here
                    })

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
                        1: "You have mild fever and headache. Also you feel fatigue",
                        2: "Shortness of breath, wheezing, chest tightness",
                        3: "You have a runny nose, sore throat and mild cough"
                    }
                })
        # Handle symptom selection
        else:
            import json
            data = json.loads(request.body.decode('utf-8'))
            symptom_choice = data.get('symptom_choice')

            # Validate input strictly: must be '1', '2', or '3'
            if symptom_choice not in ['1', '2', '3']:
                return JsonResponse({
                    'prediction': 'invalid',
                    'message': "Invalid choice. Please enter 1, 2, or 3."
                })

            # Valid choices
            if symptom_choice == '1':
                illness = 'COVID-like symptoms'
                slugs = ["mark-patel", "sarah-li", "john-kim"]

            elif symptom_choice == '2':
                illness = 'Asthma-like symptoms'
                slugs = ["priya-nair", "leo-white", "maria-singh"]

            elif symptom_choice == '3':
                illness = 'Common Cold-like symptoms'
                slugs = ["james-roy", "aisha-khan", "ben-wilson"]

            doctors = []
            for slug in slugs:
                doc = DOCTORS[slug]
                doctors.append({
                    "name": doc["name"],
                    "link": reverse("classifier:doctor_detail", kwargs={"slug": slug}),
                })

            return JsonResponse({
                'prediction': 'symptom_followup',
                'message': f"Based on your symptoms, it might be {illness}. Here are some doctors you can consult:",
                'doctors': doctors
            })

def doctor_detail(request, slug):
    doctor = DOCTORS.get(slug)
    if not doctor:
        raise Http404("Doctor not found")

    ist = ZoneInfo("Asia/Kolkata")
    today_ist = timezone.now().astimezone(ist)
    tomorrow_date = (today_ist + timedelta(days=1)).date()

    # Generate slots for next day
    slots = []
    for hour in range(9, 17):
        if hour == 13:
            continue
        slot_start = datetime(tomorrow_date.year, tomorrow_date.month, tomorrow_date.day,
                              hour, 0, tzinfo=ist)
        slots.append(slot_start)

    existing = Appointment.objects.filter(
        doctor_name=doctor['name'],
        appointment_datetime__date=tomorrow_date,
    )

    booked_times = {appt.appointment_datetime.time() for appt in existing}

    if request.user.is_authenticated:
        # All bookings for this user tomorrow (any doctor)
        user_tomorrow_all = Appointment.objects.filter(
            user=request.user,
            appointment_datetime__date=tomorrow_date,
        )
        user_booked_times = {appt.appointment_datetime.time() for appt in user_tomorrow_all}

        # Does this user already have one booking with THIS doctor tomorrow?
        doctor_user_has_booking = existing.filter(user=request.user).exists()
    else:
        user_booked_times = set()
        doctor_user_has_booking = False

    return render(request, "classifier/doctor_detail.html", {
        "doctor": doctor,
        "slots": slots,
        "booked_times": booked_times,
        "user_booked_times": user_booked_times,
        "tomorrow_date": tomorrow_date,
        "doctor_user_has_booking": doctor_user_has_booking,
    })



@login_required
def book_appointment(request, slug):
    doctor = DOCTORS.get(slug)
    if not doctor:
        raise Http404("Doctor not found")

    if request.method != 'POST':
        return redirect('classifier:doctor_detail', slug=slug)

    time_str = request.POST.get('slot_time')
    if not time_str:
        messages.error(request, "No time slot selected.")
        return redirect('classifier:doctor_detail', slug=slug)

    try:
        slot_hour, slot_minute = map(int, time_str.split(':'))
    except Exception:
        messages.error(request, "Invalid time format.")
        return redirect('classifier:doctor_detail', slug=slug)

    ist = ZoneInfo("Asia/Kolkata")
    tomorrow_date = (timezone.now().astimezone(ist) + timedelta(days=1)).date()
    slot_datetime = datetime(
        tomorrow_date.year, tomorrow_date.month, tomorrow_date.day,
        slot_hour, slot_minute, tzinfo=ist
    )

    # ❌ 1. Block multiple bookings with the SAME doctor on the same day
    if Appointment.objects.filter(
        user=request.user,
        doctor_name=doctor['name'],
        appointment_datetime__date=tomorrow_date
    ).exists():
        messages.error(request, "You already have an appointment with this doctor for tomorrow.")
        return redirect('classifier:doctor_detail', slug=slug)

    # ❌ 2. Block same time slot with ANY doctor
    if Appointment.objects.filter(
        user=request.user,
        appointment_datetime=slot_datetime
    ).exists():
        messages.error(request, "You already have an appointment at this time with another doctor.")
        return redirect('classifier:doctor_detail', slug=slug)

    # ❌ 3. Block if this specific doctor/time is already taken by someone else
    if Appointment.objects.filter(
        doctor_name=doctor['name'],
        appointment_datetime=slot_datetime
    ).exists():
        messages.error(request, "This time slot is already booked for this doctor.")
        return redirect('classifier:doctor_detail', slug=slug)

    # ✅ All good → create booking
    Appointment.objects.create(
        user=request.user,
        doctor_name=doctor['name'],
        appointment_datetime=slot_datetime
    )

    # ---- Dummy email to doctor ----
    subject = f"New appointment booked for {slot_datetime.strftime('%d %b %Y %I:%M %p')} IST"
    message = (
        f"Dear {doctor['name']},\n\n"
        f"A new appointment has been booked.\n\n"
        f"Patient: {request.user.username}\n"
        f"Doctor: {doctor['name']}\n"
        f"Date & Time: {slot_datetime.strftime('%d %b %Y %I:%M %p')} IST\n\n"
        f"This is a demo notification from the LungCARE app."
    )

    doctor_email = doctor.get('email')  # from your DOCTORS dict

    if doctor_email:
        send_mail(
            subject,
            message,
            settings.DEFAULT_FROM_EMAIL,
            [doctor_email],
            fail_silently=True,  # avoid breaking the flow if email fails
        )
    # ---- End email block ----

    messages.success(
        request,
        f"Appointment booked with Dr. {doctor['name']} at {slot_datetime.strftime('%I:%M %p')} IST."
    )
    return redirect('classifier:doctor_detail', slug=slug)



@login_required
def my_bookings(request):
    bookings = Appointment.objects.filter(user=request.user).order_by('appointment_datetime')

    # Attach doctor info from DOCTORS dict
    for appt in bookings:
        # Find doctor by name
        doc = next((d for d in DOCTORS.values() if d["name"] == appt.doctor_name), None)

        if doc:
            # You can adjust these fields as you like
            appt.doctor_hospital = doc.get("hospital", "")
            appt.doctor_location = doc.get("location", "")
            appt.doctor_phone = doc.get("phone", "")
            appt.doctor_email = doc.get("email", "")
        else:
            appt.doctor_hospital = ""
            appt.doctor_location = ""
            appt.doctor_phone = ""
            appt.doctor_email = ""

    return render(request, "classifier/my_bookings.html", {"bookings": bookings})


@login_required
def delete_booking(request, pk):
    """
    Delete a booking that belongs to the current user.
    """
    booking = get_object_or_404(Appointment, pk=pk, user=request.user)

    if request.method == 'POST':
        booking.delete()
        messages.success(request, "Your appointment has been cancelled.")
        return redirect('classifier:my_bookings')

    # If accessed via GET, just redirect (we don't allow GET delete)
    return redirect('classifier:my_bookings')
