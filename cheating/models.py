from django.db import models

from django.db import models
from django.contrib.auth.models import User

class Dataset(models.Model):
    email = models.EmailField('Email',max_length=100)
    url_file = models.FileField('Student video (URL)',upload_to='dataset/',blank=False,null=False)
    created_at = models.DateTimeField('Creation date',auto_now_add=True)
    
    def __str__(self):
        return self.email, self.url_file

    class Meta:
        verbose_name ='dataSet'


class Media(models.Model):
    source = models.CharField('Media from', max_length=200)
    model = models.CharField('Choice Model', max_length=200)
    url_file = models.FileField('schedule video or image (URL)',upload_to='multimedia/',blank=False,null=False)
    created_at = models.DateTimeField('Creation date',auto_now_add=True)
    def __str__(self):
        return self

    class Meta:
        verbose_name ='media'


class ModelML(models.Model):
    name = models.CharField('Model Name',max_length=200)
    url_file = models.FileField('Model (URL)',upload_to='models/',blank=False,null=False)
    description = models.TextField('Describe Model',blank=True,null=True)
    created_at = models.DateTimeField('Creation date',auto_now_add=True)
    moified_at = models.DateTimeField('Creation date',auto_now_add=True)

    def __str__(self):
        return self.name, self.url_file

    class Meta:
        verbose_name ='model ML'


class Cheat(models.Model):
    name = models.CharField('Model Name',max_length=200)
    url_file = models.FileField('Cheat image (URL)',upload_to='cheats/',blank=False,null=False)
    media = models.ForeignKey(Media, on_delete= models.CASCADE)

    def __str__(self):
        return self.name, self.url_file

    # change model representation (useful in admin section)
    class Meta:
        verbose_name ='cheat'


class Emotion(models.Model):
    url_file = models.FileField('Cheat image (URL)',upload_to='emotions/',blank=False,null=False)
    media = models.ForeignKey(Media, on_delete= models.CASCADE)
    nb_sad = models.IntegerField('Number Sad', null=True)
    nb_happy = models.IntegerField('Number Happy', null=True)
    nb_disgust = models.IntegerField('Number Disgust', null=True)
    nb_neutral = models.IntegerField('Number Neutral', null=True)
    nb_fear = models.IntegerField('Number Fear', null=True)
    nb_angry = models.IntegerField('Number Angry', null=True)
    nb_surprise = models.IntegerField('Number Surprise', null=True)
   

    def __str__(self):
        return self

    # change model representation (useful in admin section)
    class Meta:
        verbose_name ='emotion'



class Head(models.Model):
    url_file = models.FileField('Cheat image (URL)',upload_to='headPose/',blank=False,null=False)
    media = models.ForeignKey(Media, on_delete= models.CASCADE)
    head_left = models.BooleanField(default=False)
    head_right = models.BooleanField(default=False)
    nb_left = models.IntegerField('Number head left', null=True)
    nb_right = models.IntegerField('Number head right', null=True)
   

    def __str__(self):
        return self

    # change model representation (useful in admin section)
    class Meta:
        verbose_name ='head pose'


