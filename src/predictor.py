# 今取っている分割データと見つかった読みデータの数から、ボーナス係数を減らしてでも次の分割データを取得するべきかを推定する
# 今まで取得した分割データに重なっていないかつ次の分割データに重なっている分割データがいくつ存在するのかを雑に推定

import math
from itertools import permutations

import numpy as np


def predict_all(match):
    problem = match.current_problem
    points = []
    for i in range(0, problem.num_chunks - problem.num_got_chunks + 1):
        points.append(str(predict(match, i)))
    return ", ".join(points)


def predict_next(match, num_add):
    problem = match.current_problem
    cur_factor = match.bonus_factor[problem.num_got_chunks]
    factor = match.bonus_factor[problem.num_got_chunks + num_add]
    min_points = factor * len(problem.using_speeches)
    est_points = predict(match, num_add)
    max_points = factor * problem.num_total_speeches
    req = math.floor((cur_factor - factor) / factor * len(problem.using_speeches)) + 1
    rem = 88-len(np.unique(match.used_speeches))*2
    if rem == 0:
        prob = 0
    else:
        prob = problem.num_total_speeches / rem
        prob = (prob * match.correct_point - (1 - prob) * match.wrong_penalty)
    return round(min_points, 1), round(est_points, 1), round(max_points, 1), req, round(prob, 1)


def predict(match, num_add):
    problem = match.current_problem
    factor = match.bonus_factor[problem.num_got_chunks + num_add]
    if num_add == 0:
        points = factor * len(problem.using_speeches)
    else:
        rem_space = []
        sids = {}
        sid = 0
        for i in range(problem.num_chunks):
            if problem.chunks[i] is None:
                rem_space.append(i)
                if i == 0 or problem.chunks[i - 1] is not None:
                    sid += 1
                sids[i] = sid
        rem_speeches = problem.num_total_speeches - len(problem.using_speeches)
        sum_chunks = 0
        count = 0
        for per in permutations(rem_space, num_add):
            cs = len(set([sids[i] for i in per]))
            sum_chunks += rem_speeches * cs
            count += sid
        points = factor * (sum_chunks / count + len(problem.using_speeches))
    return round(points, 1)
