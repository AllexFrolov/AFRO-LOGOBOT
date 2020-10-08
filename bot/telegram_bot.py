import telebot
from telebot import types
import time
from utils import add_text_to_img
from generator_part.logo_gen import gen_logo_color
from afro_postprocess import superresolute, imgfilter
from PIL import Image
import requests
from config import token

bot_token = token
bot = telebot.TeleBot(bot_token)


@bot.message_handler(commands=['help', 'start'])
def send_info(message):
    if message.text == '/start':
        bot.send_message(message.from_user.id,
                         'Привет!\nЯ умею генерировать логотипы. Давай сгенерируем логотип для твоей компании! 😋\nОтправь мне название твоей компании.')
    elif message.text == '/help':
        bot.send_message(message.from_user.id, 'Отправь мне название твоей компании 👌🏿')


@bot.message_handler()
def company_receiving(message):
    print(message.text)
    # generating logo
    logo = gen_logo_color()
    # applying superresolution and filtering
    logo = superresolute(logo)
    logo = imgfilter(logo)
    # adding company name on the image
    logo = add_text_to_img(message.text, logo)
    logo = Image.fromarray(logo)

    # sending picture to user
    keyboard = types.InlineKeyboardMarkup()
    url_button = types.InlineKeyboardButton(text="Примеры", callback_data='123')
    keyboard.add(url_button)
    bot.send_photo(message.from_user.id, logo, caption=message.text,
                   reply_markup=keyboard)


@bot.callback_query_handler(lambda query: query.data == '123')
def process_callback_1(query):
    bot.send_message(query.from_user.id, 'ЕЩе работаем над этим')
    im_file = bot.get_file(query.message.json['photo'][0]['file_id'])
    img = requests.get(f'https://api.telegram.org/file/bot{bot_token}/{im_file.file_path}')
    bot.send_photo(query.from_user.id, img.content)


while True:
    try:
        bot.polling()
    except Exception:
        time.sleep(1)
