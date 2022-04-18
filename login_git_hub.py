from distutils.log import info
from requests import session
from bs4 import BeautifulSoup as bs
from requests_html import HTMLSession

user = input(str('User:'))
passwd = input(str('Pass:'))

# where request sent not login page
url1 = 'https://github.com/session'

with HTMLSession() as s:  # s = session()
    request1 = s.get(url1).text
    html = bs(request1,features="lxml")
    # tag-field-attr
    # authen
    token = html.find('input', {'name': 'authenticity_token'}).attrs['value']
    # action
    com_mit = html.find("input", {"name": "commit"}).attrs['value']

    login_data = {'login': user,
                  'password': passwd,
                  'commit': com_mit,
                  'authenticity_token': token}

    response1 = s.post(url1, data=login_data)
    soup = bs(response1.content, "html.parser")
    info_user = soup.find(
        'meta', {'name': 'user-login'}).attrs['content']
    if(response1.status_code == 200):
        print('Request Sucess!')
        print('Status_code: ' + str(response1.status_code))
        print('your login username is: '+info_user)
    else:
        print('Status_code: ' + str(response1.status_code))
