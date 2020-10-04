import cv2
import numpy as np
import telebot 

bot_token = "1225147524:AAHSRA9T2dmlSCmHJgtzWvjjlXpvk7OauKM"
bot = telebot.TeleBot(token = bot_token)



@bot.message_handler(commands = ['help', 'start'])
def send_info(message):
    if message.text=='/start':
        bot.send_message(message.from_user.id, 'Привет!\nЯ умею генерировать логотипы. Давай создадим логотип для твоей компании! 😋\nОтправь мне картинку с названием твоей компании.')
    elif message.text=='/help':
        bot.send_message(message.from_user.id, 'Отправь мне картинку с названием твоей компании 👌🏿')


@bot.message_handler(content_types=['photo','document'])
def picture_receiving(message):
    # if document given
    if message.document!=None:
        bot.send_message(message.from_user.id, 'Я не работаю с файлами, дай мне фото')
        return
    image = message.photo[2]
    # image dimensions
    height = image.height
    width = image.width
    # getting the file by id
    file_id_info = bot.get_file(image.file_id)
    file_bytes = bot.download_file(file_id_info.file_path)
    # saving user picture
    with open("image.jpg", 'wb') as new_file:
        new_file.write(file_bytes)
    
    #picture = cv2.imread('image.jpg')

    # sending picture to user
    bot.send_photo(message.from_user.id, file_bytes, caption='TEST')



while True:
    try:
        bot.polling()
    except Exception:
        time.sleep(15)
