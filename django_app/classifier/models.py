# classifier/models.py
from django.conf import settings
from django.db import models

class Appointment(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    doctor_name = models.CharField(max_length=100)  # name from Doctor profile
    appointment_datetime = models.DateTimeField()   # date and time of the appointment
    booked_at = models.DateTimeField(auto_now_add=True)  # timestamp when booking was made

    class Meta:
        # Ensure no double-booking of the same doctor and time slot:
        constraints = [
            models.UniqueConstraint(fields=['doctor_name', 'appointment_datetime'], 
                                     name='unique_doctor_time_slot')
        ]
        ordering = ['appointment_datetime']  # sort appointments by date/time
