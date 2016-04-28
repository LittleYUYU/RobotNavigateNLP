from django.shortcuts import render
from django.http import HttpResponse
import datetime
import json
import sys, os
sys.path.append('../seq2seq')
import seq2seq_interface

def add(request):
    a = request.GET['a']
    b = request.GET['b']
    a = int(a)
    b = int(b)
    return HttpResponse(str(a+b))

def ajax_dict(request):
    name_dict = {'twz': 'Love python and Django', 'zqxt': 'I am teaching Django'}
    return HttpResponse(json.dumps(name_dict), content_type='application/json')

def ajax_list(request):
    a = range(100)
    return HttpResponse(json.dumps(a), content_type='application/json')

def index(request):
    return render(request, 'index.html')

def current_datetime(request):
    now = datetime.datetime.now()
    html = "<html><body>It is now %s.</body></html>" % now
    return HttpResponse(html)

def judge(request):
    instruction = request.GET['instruction']
    position = request.GET['position'] #"[[23,17,0]]"
    direction = request.GET['direction'] 
    result = seq2seq_interface.inter_decode(instruction, position, direction)
    # result = seq2seq_interface.test()
    print(result)
    return HttpResponse(result)

def search(request, keyword):
    if request.session.get('UserID', False):
        UserID = request.session['UserID']
        UserType = request.session['UserType']
        UserAccount = request.session['UserAccount']
    else:
        UserID = None
        UserType = None
        UserAccount = None
    # commodityList = Commodity.objects.filter(CommodityName__contains = keyword)
    # commodityList_by_amount = commodityList.order_by('SoldAmount')
    # commodityList_by_price = commodityList.order_by('SellPrice')
    # commodityList_by_counterprice = commodityList.order_by('-SellPrice')
    return render_to_response('Customer_CommodityList.html', locals())