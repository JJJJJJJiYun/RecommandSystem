import sys
import time

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils.redis import redis
from utils.redis_key import *


class CollaborativeFiltering(object):
    # 取的相似度最近的个数
    TopK = 20

    def __init__(self):
        # user-item 评分表
        self.user_items_score_dict = dict()
        # user-item 倒排表
        self.item_users_dict = dict()
        # user cf 的 user 打分结果
        self.user_cf_users_items_interest_dict = dict()
        # item cf 的 user 打分结果
        self.item_cf_users_items_interest_dict = dict()

    def calculate(self, is_evaluate=False):
        """后台计算任务"""
        while True:
            # 加载数据
            if is_evaluate:
                self.load_movie_data("ratings.csv")
            else:
                self.load_data()
            if len(self.user_items_score_dict) > 1 and len(self.item_users_dict):
                # 进行计算
                self.user_cf()
                # self.item_cf()
            else:
                time.sleep(30)
            if is_evaluate:
                mae_score, rmse_score = self.evaluate(self.user_cf_users_items_interest_dict)
                print("user cf mae score: %f, rmse score: %f" % (mae_score, rmse_score))
                # mae_score, rmse_score = self.evaluate(self.item_cf_users_items_interest_dict)
                # print("item cf mae score: %d, rmse score: %d" % (mae_score, rmse_score))
                exit(0)

    def load_data(self):
        """加载数据"""
        keys = redis.keys(key_of_user_rating_data('*'))
        for key in keys:
            user_id = key.split('_', 3)[2].split(':')[1]
            item_scores = redis.zrange(key, 0, -1, withscores=True)
            for item_id, score in item_scores:
                if item_id not in self.item_users_dict:
                    self.item_users_dict[item_id] = set()
                self.item_users_dict[item_id].add(user_id)
                if user_id not in self.user_items_score_dict:
                    self.user_items_score_dict[user_id] = dict()
                self.user_items_score_dict[user_id][item_id] = score
        print("load data succ, users: %d, items: %d" % (len(self.user_items_score_dict), len(self.item_users_dict)))
        # print(self.user_items_score_dict)

    def load_test_data(self):
        """导入测试数据集，测试算法正确性"""
        self.user_items_score_dict = {
            'A': {'a': 1.0, 'b': 1.0, 'd': 1.0},
            'B': {'a': 1.0, 'c': 1.0},
            'C': {'b': 1.0, 'e': 1.0},
            'D': {'c': 1.0, 'd': 1.0, 'e': 1.0},
            'E': {'a': 1.0, 'b': 1.0, 'd': 1.0},
            'F': {'a': 1.0, 'c': 1.0},
            'G': {'b': 1.0, 'e': 1.0},
            'H': {'c': 1.0, 'd': 1.0, 'e': 1.0},
        }
        self.item_users_dict = {
            'a': ['A', 'B', 'E', 'F'],
            'b': ['A', 'C', 'E', 'G'],
            'c': ['B', 'D', 'F', 'H'],
            'd': ['A', 'D', 'E', 'H'],
            'e': ['C', 'D', 'G', 'H']
        }

    def load_movie_data(self, filename):
        """导入文件，加载需要的信息到内存中"""
        user_movie_file = pd.read_csv(filename, usecols=['userId', 'movieId', 'rating'],
                                      dtype={'userId': str, 'movieId': str, 'rating': float})
        for _, row in user_movie_file.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            rating = row['rating']
            if movie_id not in self.item_users_dict:
                self.item_users_dict[movie_id] = set()
            self.item_users_dict[movie_id].add(user_id)
            if user_id not in self.user_items_score_dict:
                self.user_items_score_dict[user_id] = dict()
            self.user_items_score_dict[user_id][movie_id] = rating
        print("load movie file succ")
        print("user count:", len(self.user_items_score_dict))
        print("item count:", len(self.item_users_dict))

    def user_cf(self):
        """基于用户的协同过滤算法"""
        # k-means 聚类
        user_cluster_result, cluster_user_result = self.k_means_clustering()
        # 计算 user 间的欧式距离（只有同聚类下的 user 会被计算）
        users_euclidean_distance_dict = self.calculate_users_euclidean_distance(user_cluster_result,
                                                                                cluster_user_result)
        # 计算 user 对 item 的兴趣排行
        # 统计分数的最大最小值来进行归一化
        users_min_max_score_dict = dict()
        for user1, user_euclidean_distance_tuples in users_euclidean_distance_dict.items():
            self.user_cf_users_items_interest_dict[user1] = dict()
            min_score, max_score = sys.maxsize, -sys.maxsize
            for item, users in self.item_users_dict.items():
                # 计算该 user 对每个 item 的兴趣值
                score = 0
                for user2, euclidean_distance in user_euclidean_distance_tuples[0:self.TopK]:
                    if user2 in users:
                        # 只有欧氏距离与该 user 在一定范围内的 user 对这个 item 有浏览记录，才会加入计算当中
                        score += euclidean_distance * self.user_items_score_dict[user2][item]
                self.user_cf_users_items_interest_dict[user1][item] = score
                min_score, max_score = min(min_score, score), max(max_score, score)
            users_min_max_score_dict[user1] = (min_score, max_score)
        print("calculate user's interest in items finished")
        # print(self.user_cf_users_items_interest_dict)
        # 归一化
        self.user_cf_users_items_interest_dict = self.normalize(self.user_cf_users_items_interest_dict,
                                                                users_min_max_score_dict)
        # 结果写入 redis
        for user, item_score_dict in self.user_cf_users_items_interest_dict.items():
            redis.zadd(key_of_user_cf_user_item_interest(user), item_score_dict)
            redis.expire(key_of_user_cf_user_item_interest(user), 20)

    def item_cf(self):
        """基于物品的协同过滤算法"""
        # 计算 item 的相似度分数
        item_nearest_score_dict = self.calculate_items_nearest_score()
        # 计算 user 对 item 的兴趣排行
        # 统计分数的最大最小值来进行归一化
        user_min_max_score_dict = dict()
        for user, item_score_dict in self.user_items_score_dict.items():
            self.item_cf_users_items_interest_dict[user] = dict()
            min_score, max_score = sys.maxsize, -sys.maxsize
            for item1, _ in self.item_users_dict.items():
                score = 0
                for item2, nearest_score in item_nearest_score_dict[item1][0:self.TopK]:
                    if item2 in item_score_dict.keys():
                        score += nearest_score * self.user_items_score_dict[user][item2]
                self.item_cf_users_items_interest_dict[user][item1] = score
                min_score, max_score = min(min_score, score), max(max_score, score)
            user_min_max_score_dict[user] = (min_score, max_score)
        print("calculate user's interest in items finished")
        # print(self.item_cf_users_items_interest_dict)
        # 归一化
        self.item_cf_users_items_interest_dict = self.normalize(self.item_cf_users_items_interest_dict,
                                                                user_min_max_score_dict)
        # 结果写入 redis
        for user, item_score_dict in self.item_cf_users_items_interest_dict.items():
            redis.zadd(key_of_item_cf_user_item_interest(user), item_score_dict)
            redis.expire(key_of_item_cf_user_item_interest(user), 20)

    def calculate_users_euclidean_distance(self, user_cluster_result, cluster_user_result):
        """计算 user 间的欧式距离"""
        users_euclidean_distance_dict = dict()
        for user1, _ in self.user_items_score_dict.items():
            users_euclidean_distance_dict[user1] = dict()
            for user2 in cluster_user_result[user_cluster_result[user1]]:
                # 对于每一个 user，计算他在同一聚类中的所有 user 的欧式距离
                if user1 == user2:
                    # 如果是自己不计算
                    continue
                count = 0
                for item, users in self.item_users_dict.items():
                    # 计算出他们的公共 item 数量
                    if user1 in users and user2 in users:
                        count += 1
                euclidean_distance = count / (
                        len(self.user_items_score_dict[user1]) * len(self.user_items_score_dict[user2])) ** 0.5
                users_euclidean_distance_dict[user1][user2] = euclidean_distance
            # 排序
            users_euclidean_distance_dict[user1] = sorted(users_euclidean_distance_dict[user1].items(),
                                                          key=lambda d: d[1], reverse=True)
        print("calculate user's euclidean distance finished")
        # print(users_euclidean_distance_dict)
        return users_euclidean_distance_dict

    def calculate_items_nearest_score(self):
        """计算 item 的相似度分数"""
        item_nearest_score_dict = dict()
        for item1, users1 in self.item_users_dict.items():
            item_nearest_score_dict[item1] = dict()
            for item2, users2 in self.item_users_dict.items():
                if item1 == item2:
                    continue
                both_count = 0
                for _, item_score_dict in self.user_items_score_dict.items():
                    if item1 in item_score_dict.keys() and item2 in item_score_dict.keys():
                        both_count += 1
                score = both_count / ((len(users1) * len(users2)) ** 0.5)
                item_nearest_score_dict[item1][item2] = score
            item_nearest_score_dict[item1] = sorted(item_nearest_score_dict[item1].items(),
                                                    key=lambda d: d[1], reverse=True)
        print("calculate item nearest score finished")
        # print(item_nearest_score_dict)
        return item_nearest_score_dict

    def k_means_clustering(self):
        """K-means 聚类"""
        user_cluster_result, cluster_user_result = dict(), dict()
        # 将 user 对每个 item 的评分降到二维
        pca = PCA(n_components=2)
        data = [[item_score_dict[item] if item in item_score_dict else 0 for item, _ in self.item_users_dict.items()]
                for
                _, item_score_dict in self.user_items_score_dict.items()]
        pca_data = pca.fit_transform(data)
        # plt.scatter(pca_data[:, 0], pca_data[:, 1])
        # plt.show()
        # k-means 聚类
        y_pred = KMeans(n_clusters=4).fit_predict(pca_data)
        # plt.scatter(pca_data[:, 0], pca_data[:, 1], c=y_pred)
        # plt.show()
        # 收集聚类结果
        for i, (user, _) in enumerate(self.user_items_score_dict.items()):
            user_cluster_result[user] = y_pred[i]
            if y_pred[i] not in cluster_user_result:
                cluster_user_result[y_pred[i]] = list()
            cluster_user_result[y_pred[i]].append(user)
        # print(cluster_user_result)
        # print(user_cluster_result)
        print("k-means clustering succ")
        return user_cluster_result, cluster_user_result

    @staticmethod
    def normalize(users_items_interest_dict, user_min_max_score_dict):
        """归一化"""
        for user, item_score_dict in users_items_interest_dict.items():
            min_score, max_score = user_min_max_score_dict[user]
            for item, score in item_score_dict.items():
                if max_score - min_score == 0:
                    the_score = max_score
                else:
                    the_score = (score - min_score) / (max_score - min_score)
                if the_score != 0:
                    users_items_interest_dict[user][item] = the_score
        print("normalizatoin finished")
        # print(user_min_max_score_dict)
        # print(users_items_interest_dict)
        return users_items_interest_dict

    def evaluate(self, users_items_interst_dict):
        mae_scores = list()
        rmse_scrores = list()
        for user, item_score_dict in self.user_items_score_dict.items():
            n1 = 0
            n2 = 0
            for item, score in users_items_interst_dict[user].items():
                n1 += (abs(score - (item_score_dict[item] if item in item_score_dict else 0)))
                n2 += (score - (item_score_dict[item] if item in item_score_dict else 0)) ** 2
            mae_scores.append(n1 / len(users_items_interst_dict[user]))
            rmse_scrores.append((n2 / len(users_items_interst_dict[user])) ** 0.5)
        print(mae_scores)
        print(rmse_scrores)
        return sum(mae_scores) / len(mae_scores), sum(rmse_scrores) / len(rmse_scrores)
