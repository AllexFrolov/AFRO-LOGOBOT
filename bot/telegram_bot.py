import cv2
import numpy as np
import telebot 
import time
import png
from utils import add_text_to_img
from generator_part.logo_gen import gen_logo_color
from afro_postprocess import superresolute, imgfilter
bot_token = "1225147524:AAHSRA9T2dmlSCmHJgtzWvjjlXpvk7OauKM"
bot = telebot.TeleBot(token = bot_token)



@bot.message_handler(commands = ['help', 'start'])
def send_info(message):
    if message.text=='/start':
        bot.send_message(message.from_user.id, 'Привет!\nЯ умею генерировать логотипы. Давай сгенерируем логотип для твоей компании! 😋\nОтправь мне название твоей компании.')
    elif message.text=='/help':
        bot.send_message(message.from_user.id, 'Отправь мне название твоей компании 👌🏿')


@bot.message_handler()
def company_receiving(message):
    logo = gen_logo_color()
    logo = imgfilter(superresolute(logo))
    
    logo = add_text_to_img(message.text, logo)
    

    print('shape',logo.shape)
    png.from_array(logo, mode="L").save("image.png")
    
    # picture = cv2.imread('image.jpg')
    print(logo.tobytes())
    # sending picture to user
    bot.send_photo(message.from_user.id, logo.tobytes(), caption='TEST')





while True:
    try:
        bot.polling()
    except Exception:
        time.sleep(15)
