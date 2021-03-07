import time

import requests
import json
from threading import Thread

# jNfpRoEPKxpfsd4A
api_key = 'jNfpRoEPKxpfsd4A'

response = requests.get('https://lichess.org/api/account',
                        headers={
                            'Authorization': f'Bearer {api_key}'
                            # Need this or you will get a 401: Not Authorized response
                        })
user_data = json.loads(response.content)

print(user_data)

response = requests.post('https://lichess.org/api/challenge/ofekbr',
                         headers={
                             'Authorization': f'Bearer {api_key}'
                             # Need this or you will get a 401: Not Authorized response
                         })

game_challenge = json.loads(response.content)

game_id = game_challenge['challenge']['id']

game_history = []


def get_stream():
    board_url = f'https://lichess.org/api/board/game/stream/{game_id}'

    s = requests.Session()

    resp = s.get(board_url, headers={'Authorization': f'Bearer {api_key}'}, stream=True)

    for line in resp.iter_lines():
        if line:
            game_history.append(line)
            print(line)


# board_response = get_stream(board_url)

time.sleep(10)

thread = Thread(target=get_stream)
thread.start()

# json_board_response = json.loads(board_response)

# board_request = requests.get(f'https://lichess.org/api/board/game/stream/{game_id}',
#                              headers={
#                                  'Authorization': f'Bearer {api_key}'
#                                  # Need this or you will get a 401: Not Authorized response
#                              })

# move=input()
for i in range(10):
    print("AAAAA")
    time.sleep(1)
while True:
    print(len(game_history))
    #move = input()
    move='e7e6'
    move_request = requests.post(f'https://lichess.org/api/board/game/{game_id}/move/{move}',
                                 headers={
                                     'Authorization': f'Bearer {api_key}'
                                     # Need this or you will get a 401: Not Authorized response
                                 })
