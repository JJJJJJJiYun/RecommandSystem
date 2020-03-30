def key_of_user_rating_data(user_id):
    return "rating_data_user:%s" % user_id


def key_of_user_recommand_result(user_id):
    return "recommand_result_user:%s" % user_id


def key_of_default_recommand_result():
    return "default_recommand_result"


def key_of_user_cf_user_item_interest(user_id):
    return "user_cf_user:%s_item_interest" % user_id


def key_of_item_cf_user_item_interest(user_id):
    return "item_cf_user:%s_item_interest" % user_id
