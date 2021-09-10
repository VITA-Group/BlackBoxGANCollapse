import os
import pickle
import re

def grep(pat, txt, ind):
    r = re.search(pat, txt)
    return int(r.group(ind))

def compute_age_distribution(path):
    # path = 'monte_carlo_sampling_1m_128_balanced_age/ages/0_1000000'

    pkls = []
    for root, dirs, files in os.walk(path):
        if len(files) != 0:
            pkls.extend([os.path.join(root, file) for file in files])
    # pkls = os.listdir(path)
    pkls.sort(key=lambda txt: grep(r"(\d+)_(\d+)\.pkl", txt, 1))
    print(pkls)
    # sample_lst = []

    young_count, old_count = 0, 0
    for pkl in pkls:
        print(pkl)
        with open(pkl, 'rb') as handle:
            samples = pickle.load(handle)
            for s in samples:
                age = int(s)
                if age <= 30:
                    young_count += 1
                else:
                    old_count += 1
    print(young_count)
    print(old_count)

def compute_gender_distribution(path_lst):
    # path_lst = ['monte_carlo_sampling_celebahq_100k/genders/0_100000']
    male_count_lst, female_count_lst = [], []
    for path in path_lst:
        # path = os.path.join(path, 'genders/0_100000')
        pkls = []
        for root, dirs, files in os.walk(path):
            if len(files) != 0:
                pkls.extend([os.path.join(root, file) for file in files])
        # pkls = os.listdir(path)
        pkls.sort(key=lambda txt: grep(r"(\d+)_(\d+)\.pkl", txt, 1))
        print(pkls)
        sample_lst = []

        male_count, female_count = 0, 0
        for pkl in pkls:
            print(pkl)
            with open(pkl, 'rb') as handle:
                samples = pickle.load(handle)
                for s in samples:
                    gender = s
                    print(gender[0] + gender[1])
                    # if gender == 0:
                    #     female_count += 1
                    # else:
                    #     male_count += 1
                    # female_count_lst.append(female_count)
                    # male_count_lst.append(male_count)
                    # print('Female: {}'.format(female_count))
                    # print('Male: {}'.format(male_count))

                    # print(male_count_lst)
                    # print(female_count_lst)
                    # print(np.mean(male_count_lst))
                    # print(np.std(male_count_lst))
                    # print(np.mean(female_count_lst))
                    # print(np.std(female_count_lst))

def compute_race_distribution():
    import os
    import pickle
    import re
    import numpy as np

    def grep(pat, txt, ind):
        r = re.search(pat, txt)
        return int(r.group(ind))

    # path_lst = ['monte_carlo_sampling_1m_race_128_2', 'monte_carlo_sampling_1m_race_128_3',
    #             'monte_carlo_sampling_1m_race_128_4', 'monte_carlo_sampling_1m_race_128_5']

    # path_lst = ['monte_carlo_sampling_1m_128_1', 'monte_carlo_sampling_1m_128_2', 'monte_carlo_sampling_1m_128_3', 'monte_carlo_sampling_1m_128_4', 'monte_carlo_sampling_1m_128_5',
    #             'monte_carlo_sampling_1m_128_6', 'monte_carlo_sampling_1m_128_7', 'monte_carlo_sampling_1m_128_8', 'monte_carlo_sampling_1m_128_9', 'monte_carlo_sampling_1m_128_10']

    path_lst = ['monte_carlo_sampling_celebahq_100k']

    # asian_count_lst, white_count_lst, black_count_lst = [], [], []
    for path in path_lst:
        path = os.path.join(path, 'races/0_100000')
        pkls = []
        print(path)
        for root, dirs, files in os.walk(path):
            if len(files) != 0:
                pkls.extend([os.path.join(root, file) for file in files])
        # pkls = os.listdir(path)
        pkls.sort(key=lambda txt: grep(r"(\d+)_(\d+)\.pkl", txt, 1))
        # print(pkls)
        # sample_lst = []

        # asian_count, white_count, black_count = 0, 0, 0
        print(pkls)
        for pkl in pkls:
            # print(pkl)
            with open(pkl, 'rb') as handle:
                samples = pickle.load(handle)
                for s in samples:
                    race = s
                    print(race)
                    #             if race == 'Asian':
                    #                 asian_count += 1
                    #             elif race == 'White':
                    #                 white_count += 1
                    #             else:
                    #                 black_count += 1
                    # asian_count_lst.append(asian_count)
                    # white_count_lst.append(white_count)
                    # black_count_lst.append(black_count)

                    # print('Asian:{}'.format(asian_count))
                    # print('White:{}'.format(white_count))
                    # print('Black:{}'.format(black_count))
                    # print(asian_count_lst)
                    # print(white_count_lst)
                    # print(black_count_lst)
                    # print(np.mean(asian_count_lst))
                    # print(np.std(asian_count_lst))
                    # print(np.mean(white_count_lst))
                    # print(np.std(white_count_lst))
                    # print(np.mean(black_count_lst))
                    # print(np.std(black_count_lst))