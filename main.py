import time
from threading import Thread

from service.collaborative_filtering import CollaborativeFiltering
from service.recommand import recommand


def test_recommand():
    while True:
        print(recommand('1', 0, 10))
        time.sleep(0.5)


if __name__ == '__main__':
    cf = CollaborativeFiltering()
    thread_cf = Thread(target=cf.calculate)
    thread_recommand = Thread(target=test_recommand)
    thread_cf.start()
    thread_recommand.start()
