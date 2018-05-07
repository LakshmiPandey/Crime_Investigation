from django.shortcuts import render
from UI.faceform import facedata
from django.core.files.storage import FileSystemStorage
import os
# Create your views here.

def index(request):
    return render(request,'UI/index.html')

def facedetected(request,string=None):
    my_dict = {'string':string}
    return render(request,'UI/facedetected.html',my_dict)

def facesurveillance(request):
    form = facedata()
    if request.method == 'POST':
        form = facedata(request.POST,request.FILES)
        if form.is_valid():
            myfile =request.FILES['Image']
            fs = FileSystemStorage(location="media/facetotest") #defaults to   MEDIA_ROOT
            fs.delete("image.jpeg")
            filename = fs.save("image.jpeg", myfile)
            return facedetected(request)
    else:
        form = facedata()
    return render(request,'UI/facesurveillance.html',{'form':form})
