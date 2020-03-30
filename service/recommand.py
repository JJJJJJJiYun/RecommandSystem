from utils.redis import redis
from utils.redis_key import *


def recommand(user_id, start, end):
    """推荐"""
    # 看缓存池里有没有
    if redis.exists(key_of_user_recommand_result(user_id)):
        return redis.zrevrange(key_of_user_recommand_result(user_id), start, end)
    user_cf_item_score_list = redis.zrevrange(key_of_user_cf_user_item_interest(user_id), 0, -1, withscores=True)
    item_cf_item_score_list = redis.zrevrange(key_of_item_cf_user_item_interest(user_id), 0, -1, withscores=True)
    # 如果 user_cf 没有结果，返回默认推荐
    if len(user_cf_item_score_list) == 0:
        return redis.zrevrange(key_of_default_recommand_result(), start, end)
    # 如果 item_cf 没有结果，返回 user_cf 的推荐
    if len(item_cf_item_score_list) == 0:
        return [item for item, _ in user_cf_item_score_list[start:end]]
    # 计算混合推荐结果
    item_cf_item_score_dict = {item: score for item, score in item_cf_item_score_list}
    item_score_dict = dict()
    for item, score in user_cf_item_score_list:
        item_score_dict[item] = (score + (item_cf_item_score_dict[item] if item in item_cf_item_score_dict else 0)) / 2
    redis.zadd(key_of_user_recommand_result(user_id), item_score_dict)
    redis.expire(key_of_user_recommand_result(user_id), 60 * 60)
    return redis.zrevrange(key_of_user_recommand_result(user_id), start, end)
