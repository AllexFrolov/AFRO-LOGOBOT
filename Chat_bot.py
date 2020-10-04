from config import *
import cv2
import numpy as np
bot = telebot.TeleBot(token = bot_token)

# @bot.message_handler(func=lambda message: message.chat.id==chat1_id, commands = ['start'])
# def send_welcome(message):
#     log_message(message)
#     bot.reply_to(message, 'Greetings.')


@bot.message_handler(commands = ['help', 'start'])
def send_info(message):
    if message.text=='/start':
        bot.send_message(message.from_user.id, 'Привет!\nЯ умею генерировать логотипы. Давай создадим логотип для твоей компании! 😋\nОтправь мне картинку с названием твоей компании.')
    elif message.text=='/help':
        bot.send_message(message.from_user.id, 'Отправь мне картинку с названием твоей компании 👌🏿')
    # bot.send_message(message.from_user.id, "Чем могу быть полезен?\nЗадаш вопрос, вскоре получишь ответ.")


@bot.message_handler(content_types=['photo','document'])
def picture_receiving(message):
    # print(message)
    image = message.photo[2]
    height = image.height
    width = image.width
    file_id_info = bot.get_file(image.file_id)
    print(file_id_info)
    file_bytes = bot.download_file(file_id_info.file_path)
    # saving user picture
    with open("image.jpg", 'wb') as new_file:
        new_file.write(file_bytes)
    
    #picture = cv2.imread('image.jpg')

    # sending picture to user
    bot.send_photo(message.from_user.id, file_bytes, caption='TEST')

    import matplotlib.pyplot as plt
    picture = plt.imread(file_bytes, format='jpeg')
    plt.imshow(picture)
    plt.show()    


while True:
    try:
        bot.polling()
    except Exception:
        time.sleep(15)
