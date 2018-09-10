import os
import itchat
import requests

def get_response(msg):
    apiUrl = 'http://www.tuling123.com/openapi/api'
    data = {'key': 'efbbdd6418ac478aa63ce7397aabe96c',
            'info': msg,
            'userid': '本宝宝'}
    r = requests.post(apiUrl, data = data).json()
    return r.get('text')

@itchat.msg_register(itchat.content.TEXT)
def print_content(msg):
    return get_response(msg["Text"])

itchat.auto_login()
itchat.run()

os.system("pause")

itchat.logout()