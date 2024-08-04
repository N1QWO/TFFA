


#!!! DOESN'T WORK,  I get a message from CryptoPanic,
#  but there is not enough information for a prediction model,
#  you only get noise of [0,0,0,0]



import requests

# Ваш API ключ для CryptoPanic
api_key = '805c06b53909ef358f427fd8ad3f41748b7b2fab'  # Замените на ваш реальный API ключ

# URL для доступа к данным о новостях
url = 'https://cryptopanic.com/api/v1/posts/'

# Параметры запроса
params = {
    'auth_token': api_key,  # Укажите свой API ключ
    'currencies': 'BTC',  # Фильтр по монете
    'public': 'true',  # Публичные новости
}

# Выполнение запроса
response = requests.get(url, params=params)

# Проверка статуса ответа
if response.status_code == 200:
    data = response.json()
    for post in data['results']:
        title = post['title']
        url = post['url']
        published_at = post['published_at']
        votes = post['votes']
        panic_score = post.get('panic_score', 'N/A')
        print(f"Title: {title}\nURL: {url}\nPublished At: {published_at}\nVotes: {votes}\nPanic Score: {panic_score}\n")
else:
    print('Ошибка при получении данных:', response.status_code)
