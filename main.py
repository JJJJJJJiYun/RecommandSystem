import time
from threading import Thread

from service.collaborative_filtering import CollaborativeFiltering

if __name__ == '__main__':
    cf = CollaborativeFiltering()
    # thread_cf = Thread(target=cf.calculate)
    # thread_cf.start()
    # try:
    #     while True:
    #         time.sleep(60 * 60 * 24)
    # except KeyboardInterrupt:
    #     exit(0)
    cf.eval()
