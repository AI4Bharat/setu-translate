# parent_process.py
from subprocess import Popen, PIPE

def handle_output(stream):
    buffer = ''
    while True:
        chunk = stream.read(1)  # Read one byte at a time for more granular control
        if chunk == b'':
            break  # No more data
        character = chunk.decode('utf-8')
        if character == '\r':
            print(buffer, end='\r', flush=True)
            buffer = ''
        elif character == '\n':
            print(buffer, flush=True)
            buffer = ''
        else:
            buffer += character
    if buffer:
        print(buffer, flush=True)  # Print any remaining buffer

with Popen(["python", "child_process.py"], stdout=PIPE) as p:
    handle_output(p.stdout)
    p.wait()  # Wait for the child process to terminate
