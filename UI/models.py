from django.db import models
import uuid
import os
# Create your models here.

def content_file_name(instance, filename):
    ext = filename.split('.')[-1]
    filename = '%s.%s' % (instance.id, ext)
    return os.path.join('facedatabase',filename)

class facedatabase(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to=content_file_name)
    name = models.CharField(max_length = 264)
    crime = models.CharField(max_length = 264)
    dob = models.DateField()
    dod = models.DateField()
    Injail = models.CharField(max_length = 264)
    gang = models.CharField(max_length = 264)
    address = models.CharField(max_length = 264)
    def __str__(self):
        return self.name
