from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django import forms
from django.forms.utils import ErrorList


MODELS_CHOICES= [
    ('FR','Face Recognition Model'),
    ('ER','Expressions Recognition Model'),
    ('HP','Head Pose Detected Model'),
    ('CH','Cheating Activity Model'),
    ]

MEDIA_CHOICES= [
    (0,'Webcam'),
    (1,'Video file'),
    ]


class FormatErrorList(ErrorList):

    def __str__(self):
        return self.as_divs()

    def as_divs(self):
        if not self : return ''
        return '<div class="errorlist">%s</div>' % ''.join(['<p class="error">%s</p>' % e for e in self])


class CreateUserFrom(UserCreationForm):
    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']


class GeStartedForm(forms.Form):
    
    media_type = forms.ChoiceField(label='Media From', choices=MEDIA_CHOICES,initial='0', required = True, widget=forms.Select(attrs={"name": "media",'class': 'form-control form-control-lg'}))

    model_type = forms.ChoiceField(label='Choice Model', choices=MODELS_CHOICES,initial='0', required = True, widget=forms.Select(attrs={"name": "model",'class': 'form-control form-control-lg'}))
    
    video_file = forms.FileField(
        label='Video URL',
        widget=forms.FileInput(attrs={'class': 'form-control form-control-lg'}),
    )
    
    