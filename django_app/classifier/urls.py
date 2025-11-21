# classifier/urls.py
from django.urls import path
from . import views

app_name = "classifier"  # <— IMPORTANT

urlpatterns = [
    path("", views.chat_view, name="chat_view"),
    path("doctors/<slug:slug>/", views.doctor_detail, name="doctor_detail"),
    path("doctors/<slug:slug>/book/", views.book_appointment, name="book_appointment"),
    path("my-bookings/", views.my_bookings, name="my_bookings"),
    path("my-bookings/<int:pk>/delete/", views.delete_booking, name="delete_booking"),
]
