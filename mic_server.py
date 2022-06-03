import socket
import pyaudio
import wave
import numpy as np
import time
# from try_audio_google_socket import detect_intent_audio
# Socket
# HOST = socket.gethostname()
# print(HOST)
# HOST = '140.118.121.195'
HOST = '127.0.0.1'
# HOST = '0.0.0.0'
PORT = 5000

from inter import s2t
# from infer_engish import s2t

print(HOST)
def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += ele  
    
    # return string  
    return str1 
# Audio
p = pyaudio.PyAudio()
CHUNK = 1024 
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 15


frames = []
with socket.socket() as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    conn, address = server_socket.accept()
    print("Connection from " + address[0] + ":" + str(address[1]))
#     frames = []
    hello = b''
    data = conn.recv(1024)

    while data != b'':
        data = conn.recv(1024)
 
        frames.append(data)

        hello += data
        
wf = wave.open('ser_output5.wav','wb')

wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
print(s2t())



# print(frames)


