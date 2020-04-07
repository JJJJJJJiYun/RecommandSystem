import time
from threading import Thread

from service.collaborative_filtering import CollaborativeFiltering

if __name__ == '__main__':
    cf = CollaborativeFiltering()
    thread_cf = Thread(target=cf.calculate, args=[True])
    thread_cf.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)  # one day in seconds
    except KeyboardInterrupt:
        exit(0)
