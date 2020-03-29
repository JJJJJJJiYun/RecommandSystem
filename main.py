import collaborative_filtering

if __name__ == '__main__':
    cf = collaborative_filtering.CollaborativeFiltering()
    cf.load_movie_data("ratings.csv")
    # cf.load_test_data()
    cf.collaborative_filtering()
    print("recommand result:", cf.recommand('A', 10))
    # cf.k_means_clustering()
