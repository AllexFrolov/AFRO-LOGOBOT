# -*- coding: utf-8 -*-
import numpy as np
import time
import telebot
import logging
from datetime import datetime
import math

chat1_id = -1001450381106
bot_token = "1225147524:AAHSRA9T2dmlSCmHJgtzWvjjlXpvk7OauKM"
null_day = 315
date = time.localtime()
dayMaximum = 4
# logging.basicConfig(filename='{}.log'.format(date.tm_yday - null_day), level=logging.INFO, format='%(asctime)s %(message)s')
def getDate():
    date = time.localtime()
    return date

def daysInBussiness():
    return str(abs(getDate().tm_yday - null_day))

def log_message(msg):
    with open('{0}.log'.format('b' + daysInBussiness()), 'a') as f:
        print('{0} {1} {2}'.format(msg.chat.id, msg.from_user.id, msg.text), datetime.now(), file=f)

def calculate(msg):
    list = []
    with open('Chat1.log', 'r') as f:
        list = [[int(val) for val in row.split()] for row in f.read().split('\n')]
        list.remove([])
        list = np.array(list)
    userId = msg.from_user.id
    print(list)
    if userId in list[:,0]:
        userIdIndex = np.where(list[:,0] == userId)[0]
        if list[userIdIndex, 1] >= dayMaximum:
            return 1
        else:
            list[userIdIndex,1] +=1
    else:
        list = np.vstack( (list, np.array([userId, 1])) )
    with open('Chat1.log', 'w') as f:
        for row in list:
            f.write(str(row[0]) + ' ' + str(row[1]) + '\n')
    return 0

#def loger(orig_func):
#     def wrapper(message):
#         logging.info('{0}, {1}'.format(message.text, message.from_user.id))
#         orig_func()
#     return wrapper
