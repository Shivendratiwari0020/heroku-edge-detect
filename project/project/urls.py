"""project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home_view),
    path('api/test_views', views.test_views),
    path('api/upload', views.upload_file),
    path('api/modeling', views.upload_file_get),
    path('api/startmodeling', views.start_modeling),
    path('api/modelevaluation', views.model_evaluation),
    path('api/model_res', views.model_res),
    path('api/image',views.image_eva),
    path('api/ima',views.imagepath),
    path('api/rulebased',views.rulebased),
    path('api/freeze',views.freezedata),
    path('api/sendfreezedata',views.sendfreezedata),
    path('api/ruledata',views.rulesdata),
    path('api/download',views.download),
    path('api/finalimage',views.finalimagepath),
]
