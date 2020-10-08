import time

import requests
import telebot
from PIL import Image
from config import token
from generator_part.logo_gen import gen_logo_color
from telebot import types

from afro_postprocess import superresolute, imgfilter
from utils import add_text_to_img

bot_token = token
bot = telebot.TeleBot(bot_token)


@bot.message_handler(commands=['help', 'start'])
def send_info(message):
    if message.text == '/start':
        bot.send_message(message.from_user.id,
                         'Привет!\nЯ умею генерировать логотипы. '
                         'Давай сгенерируем логотип для твоей компании!'
                         '😋\nОтправь мне название твоей компании.')
    elif message.text == '/help':
        bot.send_message(message.from_user.id, 'Отправь мне название твоей компании 👌🏿')


@bot.message_handler()
def company_receiving(message):
    # FOR DEBUGGING ----------
    print(message.text)
    # ------------------------

    # generating logo
    logo = gen_logo_color()
    # applying superresolution and filtering
    logo = superresolute(logo)
    logo = imgfilter(logo)
    # adding company name on the image
    logo = add_text_to_img(message.text, logo)
    logo = Image.fromarray(logo)

    # create keyboard
    keyboard = types.InlineKeyboardMarkup()
    # create button
    url_button = types.InlineKeyboardButton(text="Примеры", callback_data='123')
    # add button to keyboard
    keyboard.add(url_button)
    # sending picture with keyboard to user
    bot.send_photo(message.from_user.id, logo, caption=message.text,
                   reply_markup=keyboard)


@bot.callback_query_handler(lambda query: query.data == '123')
def process_callback_1(query):
    bot.send_message(query.from_user.id, 'ЕЩе работаем над этим')
    # get link to file
    im_file = bot.get_file(query.message.json['photo'][0]['file_id'])
    # download file
    img = requests.get('https://api.telegram.org/file/bot%s/%s' % (bot_token, im_file.file_path))

    # TO DO: Add some method
    result = img.content
    # return Result

    bot.send_photo(query.from_user.id, result)


while True:
    try:
        bot.polling()
    except Exception:
        time.sleep(1)
