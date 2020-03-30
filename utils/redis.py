from redis import StrictRedis

redis = StrictRedis(host='localhost', port=6379, decode_responses=True)
