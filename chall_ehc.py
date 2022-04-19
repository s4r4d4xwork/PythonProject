import requests

url ='http://127.0.0.1:8000/home'

headers = {
    'Host' : '18.162.149.167',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'EHC',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'ja',
    'Connection': 'close',
    'Referer' : 'http://127.0.0.1:8000/',
    'DNT' : '1',
    'Date': '2017'
}

r = requests.get(url, headers=headers)

print(r.text)
