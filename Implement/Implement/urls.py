from django.conf.urls import patterns, include, url
from django.contrib import admin
from Implement import settings

# from django.conf.urls.defaults import *
from Implement.views import current_datetime
admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'Implement.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),
    url(r'^$', 'Implement.views.index', name='home'),
    url(r'^admin/', include(admin.site.urls)),
    (r'^time/$', current_datetime),
    # url(r'^search/(?P<keyword>\w*)/$', 'Implement.views.judge'),
    url(r'^add/$', 'Implement.views.add', name='add'),
    url(r'^judge/$', 'Implement.views.judge', name='judge'),
    url(r'^ajax_dict/$', 'Implement.views.ajax_dict', name='ajax-dict'),
    url(r'^ajax_list/$', 'Implement.views.ajax_list', name='ajax-list'),
    url(r'^assets/(?P<path>.*)','django.views.static.serve',{'document_root':settings.ASSETS_URL}),
)
