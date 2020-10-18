from django.shortcuts import render

from django.shortcuts import render, redirect
from django.http import HttpResponse

from .forms import CreateUserFrom, FormatErrorList, GeStartedForm
from django.db import transaction, IntegrityError
from .models import *
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout

from django.contrib import messages

from django.http.response import StreamingHttpResponse
from .camera import MaskDetect

import os
from django.conf import settings
import time

# variable use to load default video streaming
flux_controls = False



def index(request):
    context = {}
    return render(request , 'cheating/index.html', context)

def loginPage(request):
	if request.user.is_authenticated:
		return redirect('cheating:index')
	else:
		if request.method == 'POST':
			username = request.POST.get('username')
			password =request.POST.get('password')

			user = authenticate(request, username=username, password=password)

			if user is not None:
				login(request, user)
				return redirect('cheating:index')
			else:
				messages.info(request, 'Username OR password is incorrect')

		context = {}
		return render(request, 'cheating/login.html', context)


def logoutPage(request):
    logout(request)
    return redirect('login')


def register(request):

    context = {}
    if request.user.is_authenticated:
        return redirect('cheating/index.html')
    else:
   
        if request.method == "POST":
            # get form infos
            form = CreateUserFrom(request.POST, error_class= FormatErrorList)
            
            if form.is_valid():
                form.save()
                # got data from django validation dictionnary 
                email = form.cleaned_data["email"]
                username = form.cleaned_data["username"]
                first_name = form.cleaned_data["first_name"]
                last_name = form.cleaned_data["last_name"]
                password1 = form.cleaned_data["password1"]
                password2 = form.cleaned_data["password2"]
                
                messages.success(request,'Account successfully created for ' + username)
                return redirect('login')
            else:
                context['form_errors'] = form.errors.items()
                

        else:
            form = CreateUserFrom()
    context['form'] = form
    return render(request, 'cheating/register.html', context) 


def listing(request):
    context = {}
    return render(request , 'cheating/thanks.html', context)

# schedule without saved
def webcam(request):
   
    #flux_controls = False
    #model_choiced = 'ras'
    """
    if request.method == "POST":
        media = request.POST.get('media'),
        model = request.POST.get('model'),
        
        flux_controls = True
        context = {
            'flux_controls': flux_controls,
            'model_choiced': model
            }
        return render(request, 'cheating/webcam.html', context)
        #return redirect('cheating:webcam')
    """
    #context = {'flux_controls': flux_controls, 'model_choiced': model_choiced}
    context = {}
    return render(request, 'cheating/webcam.html', context) 

def video(request):
    #flux_controls = False
    """
    if request.method == "POST":
       
        media = request.POST.get('media'),
        video_path = request.POST.get('video'),
        model = request.POST.get('model'),
      
        flux_controls = True
        context = {
            'flux_controls': flux_controls,
            'model_choiced': model,
            'video_url': video_path
            }
        return render(request, 'cheating/video.html', context)
        #return render(request, 'cheating/video.html', context)
    """
    #context = {'flux_controls': flux_controls, 'model_choiced': 'ras'}
    context = {}
    return render(request, 'cheating/video.html', context) 

# schedule and save
def camera(request):
    return HttpResponse('Hello')


def webcam1(request):
    context = {}
    if request.method == "POST":
        media = request.POST.get('media'),
        model = request.POST.get('model'),
        
        VIDEO_PATH = 0
        flux_controls = True
        
        context = {
            'flux_controls': flux_controls,
            'model': model
            }
        return render(request, 'cheating/webcam.html', context)
   
    return render(request, 'cheating/webcam.html', context) 

def video1(request):
    context = {}
    if request.method == "POST":
       
        media = request.POST.get('media'),
        video_path = request.POST.get('video'),
        model = request.POST.get('model'),
      
        VIDEO_PATH = video_path
        flux_controls = True
        
        return render(request, 'cheating/video.html', context)
   
    return render(request, 'cheating/video.html', context) 
   


def details(request):
    return HttpResponse('Hello')


def search(request):
    return HttpResponse('Hello')

# Stream video 

def gen(camera):
    while True:
        frame = camera.get_frame()
        #time.sleep(2.0)
        yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def mask_feed(request):
    VIDEO_PATH = os.path.join(settings.BASE_DIR, 'media/multimedia/test_exam.mp4')
    if request.method == "POST":
        model = request.POST.get('model')
        
        if request.POST.get('video') is not None:
            VIDEO_PATH = request.POST.get('video')
            media = 'video file'
        else:
            VIDEO_PATH = 0
            media = 'webcam'
      
            
        message = [media, model]
            
    
    return StreamingHttpResponse(gen(MaskDetect(VIDEO_PATH)),
                content_type='multipart/x-mixed-replace; boundary=frame')
        

def mask_feed2(request, model= ' '):
    
    VIDEO_PATH = os.path.join(settings.BASE_DIR, 'media/multimedia/test_exam.mp4')
    return StreamingHttpResponse(gen(MaskDetect(VIDEO_PATH)),
                content_type='multipart/x-mixed-replace; boundary=frame')

