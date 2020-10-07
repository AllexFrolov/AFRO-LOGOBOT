import cv2
import numpy as np
import telebot 
import time
import png
from utils import add_text_to_img
from generator_part.logo_gen import gen_logo_color
from afro_postprocess import superresolute, imgfilter
import matplotlib.pyplot as plt

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
    print(message.text)
    # generating logo
    logo = gen_logo_color()
    #applying superresolution and filtering
    logo = superresolute(logo)
    logo = imgfilter(logo)
    # adding company name on the image
    logo = add_text_to_img(message.text, logo)

    cv2.imwrite('image.jpg',logo)
    image = cv2.imread('image.jpg')

    # plt.imshow(image)
    # plt.show()
    
    encoded_image = cv2.imencode('.jpeg', image)[1]
    bytes_logo = encoded_image.tobytes()
    # bytes_logo = open('image.png','rb')
    # picture = cv2.imread('image.jpg')
    # print(logo.shape)
    # bytes_logo = logo.tobytes()
    # sending picture to user
    bot.send_photo(message.from_user.id, bytes_logo, caption=message.text)





while True:
    try:
        bot.polling()
    except Exception:
        time.sleep(1)
