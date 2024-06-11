from django.urls import include, path
from inpainting.views import *
from django.conf import settings
from django.conf.urls.static import static

# ... your existing urlpatterns

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


app_name = 'inpainting'

urlpatterns = [

path('image/inpainting',lamaCleaner,name='lamaCleaner'),

]
