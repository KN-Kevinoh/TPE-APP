#import urls from project
from django.conf.urls import url


#import views 
from . import views

urlpatterns = [
    url(r'^$', views.listing, name='listing'),
    url(r'^(?P<id>[0-9]+)/$', views.details, name='details'),
    url(r'^search/$', views.search, name='search'),
    url(r'^index/$', views.index, name='index'),
    url(r'^register/$', views.register, name='register'),
    url(r'^login/$', views.loginPage, name='login'),
    url(r'^logout/$', views.logoutPage, name='logout'),
    url(r'^getstarted/webcam/$', views.webcam, name='webcam'),
    url(r'^getstarted/video/$', views.video, name='video'),
    url(r'^schedule/camera$', views.camera, name='camera'),
    url(r'^schedule/webcam$', views.webcam1, name='webcam_1'),
    url(r'^schedule/video$', views.video1, name='video_1'),
    url(r'^streaming/$', views.mask_feed, name='streamVideo'),
    url(r'^streaming2/$', views.mask_feed2, name='streamVideo2'),
]