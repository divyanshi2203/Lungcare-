# cancer_classification_project/urls.py
from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from classifier import views as classifier_views  # only if you need direct import (not required below)

from accounts import views as accounts_views  # we'll create this app/view in next step
# If you don't want a separate app, we can adjust, but this is clean.

urlpatterns = [
    path('admin/', admin.site.urls),

    # Root: login page
    path('', auth_views.LoginView.as_view(
        template_name='registration/login.html'
    ), name='login'),

    # Register & logout
    path('register/', accounts_views.register, name='register'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),

    # Chat app under /chat/
    path('chat/', include('classifier.urls', namespace='classifier')),
]
