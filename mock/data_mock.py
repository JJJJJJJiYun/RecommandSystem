import random

from utils.redis import redis
from utils.redis_key import *

for user in range(100):
    key = key_of_user_rating_data(user)
    for i in range(10):
        item = random.randint(1, 200)
        redis.zadd(key, {item: random.randint(1, 10)})
    # redis.expire(key, 300)
