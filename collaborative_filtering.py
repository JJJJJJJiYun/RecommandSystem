import pandas as pd
import sys


class CollaborativeFiltering(object):

    def __init__(self):
        # user-item 表
        self.user_items_dict = dict()
        # user-item 倒排表
        self.item_users_dict = dict()
        # user_cf 计算出的 user 对 item 的兴趣值表
        self.user_cf_users_items_interest_dict = dict()
        # 欧氏距离最近的 user 个数
        self.top_k = 20
        # item_cf 计算出的 user 对 item 的兴趣值表
        self.item_cf_users_items_interest_dict = dict()

    def load_test_data(self):
        """导入测试数据集，测试算法正确性"""
        self.user_items_dict = {
            'A': ['a', 'b', 'd'],
            'B': ['a', 'c'],
            'C': ['b', 'e'],
            'D': ['c', 'd', 'e']
        }
        self.item_users_dict = {
            'a': ['A', 'B'],
            'b': ['A', 'C'],
            'c': ['B', 'D'],
            'd': ['A', 'D'],
            'e': ['C', 'D']
        }

    def load_movie_data(self, filename):
        """导入文件，加载需要的信息到内存中"""
        user_movie_file = pd.read_csv(filename, usecols=['userId', 'movieId'])
        for _, row in user_movie_file.iterrows():
            user_id = str(row['userId'])
            movie_id = str(row['movieId'])
            if user_id not in self.user_items_dict:
                self.user_items_dict[user_id] = set()
            self.user_items_dict[user_id].add(movie_id)
            if movie_id not in self.item_users_dict:
                self.item_users_dict[movie_id] = set()
            self.item_users_dict[movie_id].add(user_id)
        print("load movie file succ")
        print("user count:", len(self.user_items_dict))
        print("item count:", len(self.item_users_dict))

    def user_cf(self):
        """基于用户的协同过滤算法"""
        # 计算 user 间的欧式距离
        users_euclidean_distance_dict = dict()
        for user1, _ in self.user_items_dict.items():
            users_euclidean_distance_dict[user1] = dict()
            for user2, _ in self.user_items_dict.items():
                # 对于每一个 user，计算他和其他所有 user 的欧式距离
                count = 0
                for item, users in self.item_users_dict.items():
                    # 计算出他们的公共 item 数量
                    if user1 in users and user2 in users:
                        count += 1
                euclidean_distance = count / (
                        len(self.user_items_dict[user1]) * len(self.user_items_dict[user2])) ** 0.5
                users_euclidean_distance_dict[user1][user2] = euclidean_distance
            # 排序，并把与自身的欧氏距离去除
            users_euclidean_distance_dict[user1] = sorted(users_euclidean_distance_dict[user1].items(),
                                                          key=lambda d: d[1], reverse=True)[1:]
        print("calculate user's euclidean distance finished")
        print(users_euclidean_distance_dict)
        # 计算 user 对 item 的兴趣排行
        # 统计分数的最大最小值来进行归一化
        user_min_max_score_dict = dict()
        for user1, euclidean_distances in users_euclidean_distance_dict.items():
            self.user_cf_users_items_interest_dict[user1] = dict()
            min_score, max_score = sys.maxsize, -sys.maxsize
            for item, users in self.item_users_dict.items():
                # 计算该 user 对每个 item 的兴趣值
                score = 0
                for user2, euclidean_distance in euclidean_distances[0:self.top_k]:
                    if user2 in users:
                        # 只有欧氏距离与该 user 在一定范围内的 user 对这个 item 有浏览记录，才会加入计算当中
                        score += euclidean_distance
                self.user_cf_users_items_interest_dict[user1][item] = score
                min_score, max_score = min(min_score, score), max(max_score, score)
            user_min_max_score_dict[user1] = (min_score, max_score)
            self.user_cf_users_items_interest_dict[user1] = sorted(
                self.user_cf_users_items_interest_dict[user1].items(),
                key=lambda d: d[1], reverse=True)
        print("calculate user's interest in items finished")
        print(self.user_cf_users_items_interest_dict)
        # 归一化
        self.user_cf_users_items_interest_dict = self.normalize(self.user_cf_users_items_interest_dict,
                                                                user_min_max_score_dict)
        print("normalizatoin finished, min max score dict:", user_min_max_score_dict)
        print(self.user_cf_users_items_interest_dict)

    def item_cf(self):
        """基于物品的协同过滤算法"""
        item_nearest_score_dict = dict()
        for item1, users1 in self.item_users_dict.items():
            item_nearest_score_dict[item1] = dict()
            for item2, users2 in self.item_users_dict.items():
                if item1 == item2:
                    continue
                both_count = 0
                for _, items in self.user_items_dict.items():
                    if item1 in items and item2 in items:
                        both_count += 1
                score = both_count / ((len(users1) * len(users2)) ** 0.5)
                item_nearest_score_dict[item1][item2] = score
            item_nearest_score_dict[item1] = sorted(item_nearest_score_dict[item1].items(),
                                                    key=lambda d: d[1], reverse=True)
        print("calculate item nearest score finished")
        print(item_nearest_score_dict)
        # 计算 user 对 item 的兴趣排行
        # 统计分数的最大最小值来进行归一化
        user_min_max_score_dict = dict()
        for user, items in self.user_items_dict.items():
            self.item_cf_users_items_interest_dict[user] = dict()
            min_score, max_score = sys.maxsize, -sys.maxsize
            for item1, _ in self.item_users_dict.items():
                score = 0
                for item2, nearest_score in item_nearest_score_dict[item1][0:self.top_k]:
                    if item2 in items:
                        score += nearest_score
                self.item_cf_users_items_interest_dict[user][item1] = score
                min_score, max_score = min(min_score, score), max(max_score, score)
            user_min_max_score_dict[user] = (min_score, max_score)
            self.item_cf_users_items_interest_dict[user] = sorted(self.item_cf_users_items_interest_dict[user].items(),
                                                                  key=lambda d: d[1], reverse=True)
        print("calculate user's interest in items finished")
        print(self.item_cf_users_items_interest_dict)
        # 归一化
        self.item_cf_users_items_interest_dict = self.normalize(self.item_cf_users_items_interest_dict,
                                                                user_min_max_score_dict)
        print("normalizatoin finished, min max score dict:", user_min_max_score_dict)
        print(self.item_cf_users_items_interest_dict)

    def recommand(self, user_id, n):
        item_cf_item_score_dict = {item: score for item, score in self.item_cf_users_items_interest_dict[user_id]}
        item_score_dict = dict()
        for item, score in self.user_cf_users_items_interest_dict[user_id]:
            item_score_dict[item] = (score + item_cf_item_score_dict[item]) / 2
        return sorted(item_score_dict.items(), key=lambda d: d[1], reverse=True)[0:n]

    @staticmethod
    def normalize(users_items_interest_dict, user_min_max_score_dict):
        """归一化"""
        for user, items_scores in users_items_interest_dict.items():
            min_score, max_score = user_min_max_score_dict[user]
            for i, (item, score) in enumerate(items_scores):
                if max_score - min_score == 0:
                    users_items_interest_dict[user][i] = (item, 1)
                else:
                    users_items_interest_dict[user][i] = (item, (score - min_score) / (max_score - min_score))
        return users_items_interest_dict
