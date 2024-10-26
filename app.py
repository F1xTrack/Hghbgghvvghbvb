import pyTelegramBotAPI as telebot
import torch
import os
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

# Инициализация бота
bot = telebot.TeleBot("7407143171:AAHQz3N_mSTKmR3eRXAT4vJH-hU3JnQQ8gA")

# Проверка и загрузка файла lora.safetensors, если он отсутствует
lora_path = "lora.safetensors"
download_url = "https://cdn-lfs-us-1.hf.co/repos/9b/c7/9bc7e1b664b50c073e263d704d61b0e3ec510dc1a1f8184d7aebb00e164386c6/5ad714d27dea0bfddce0599ab860d249c0356f7e097f9e68ef7160ed535ef93b?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27lora.safetensors%3B+filename%3D%22lora.safetensors%22%3B&Expires=1730240453&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczMDI0MDQ1M319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zLzliL2M3LzliYzdlMWI2NjRiNTBjMDczZTI2M2Q3MDRkNjFiMGUzZWM1MTBkYzFhMWY4MTg0ZDdhZWJiMDBlMTY0Mzg2YzYvNWFkNzE0ZDI3ZGVhMGJmZGRjZTA1OTlhYjg2MGQyNDljMDM1NmY3ZTA5N2Y5ZTY4ZWY3MTYwZWQ1MzVlZjkzYj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=SsodMjOWSRuIZ-R6T%7ENxKzIf3FtlgKGh8q247i2iJyltv1qhz1tADVUZRy%7EiXqeZC4aJ54LDasxCpNNGu5vTSrJm55DEtLRyTj3kzyTckgYE5MQnVS11UZSIesuceMbDnUqkj6SL9Ejff0JV319mkhLeTCLNZFX%7EVCU4MAU6eS3UvRvc8Prob%7EfF5hNuQugy9PNDnDMqc9cO7T5rqdOLoJADSJOZg4pBpyE5OuHaF6gi0h%7E6cfJjH2KjTsu7fltbhgc3sb8f0X6mJ9dVc9EmSZeBqfCFSy0Uoe1w%7Ey0vm%7Er7fSNJ7sQ%7EOWdB0dOm6H4HcP2pygr4P50Ns1Z-Tv4ouQ__&Key-Pair-Id=K24J24Z295AEI9"

if not os.path.exists(lora_path):
    print("Файл lora.safetensors не найден. Скачиваем...")
    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        with open(lora_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Загрузка завершена.")
    else:
        print("Ошибка при загрузке файла.")
        exit(1)

# Инициализация модели
model_name = "./Flux-uncensored"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Загрузка весов LoRA
lora_weights = load_file(lora_path)
model.load_state_dict(lora_weights, strict=False)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Отправь мне текстовый промпт для обработки.")

@bot.message_handler(func=lambda message: True)
def handle_prompt(message):
    prompt = message.text
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    bot.reply_to(message, response)

# Запуск бота
bot.polling()
