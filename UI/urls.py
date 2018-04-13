from django.conf.urls import url
from UI import views

#TEMPLATE TAGGING

app_name = "UI"

urlpatterns = [
    url(r'^$',views.index,name = 'index'),
    url(r'^facesurveillance/',views.facesurveillance,name = 'facesurveillance'),
    url(r'^facedetected/',views.facedetected,name = 'facedetected'),
]
