import collaborative_filtering

if __name__ == '__main__':
    cf = collaborative_filtering.CollaborativeFiltering()
    # cf.load_movie_data("ratings.csv")
    cf.load_test_data()
    cf.user_cf()
    cf.item_cf()
    print(cf.recommand('A', 10))
