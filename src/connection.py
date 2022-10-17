# サーバーとの通信

import json
import os
import wave

import numpy as np
import requests

from src.problem import Problem
from src.match import Match


# prefix = "https://procon33-practice.kosen.work/"
# prefix = "https://procon33-naprock.kosen.work/"
prefix = "http://172.28.1.1:80/"
token = ""


def test():
    url = f"{prefix}test?token={token}"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(response.content.decode().rstrip())
    print(response.text)


def load_token():
    global token
    with open("../TOKEN", "r") as f:
        token = f.readline().rstrip()


def get_match_data():
    url = f"{prefix}match?token={token}"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(response.content.decode().rstrip())
    j = response.json()
    with open(f"../problem_data/match.json", "w") as f:
        f.write(json.dumps(j))
    return Match(j["problems"], j["bonus_factor"], j["change_penalty"], j["wrong_penalty"], j["correct_point"])


def get_problem_data(match):
    url = f"{prefix}problem?token={token}"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(response.content.decode().rstrip())
    j = response.json()
    with open(f"../problem_data/{j['id']}.json", "w") as f:
        f.write(json.dumps(j))
    return Problem(match, j["id"], j["chunks"], j["start_at"], j["time_limit"], j["data"])


def get_chunk_data(problem, i):
    url = f"{prefix}problem/chunks?token={token}&n={i+1}"
    response = requests.post(url)
    if response.status_code != 200:
        raise ValueError(response.content.decode().rstrip())
    j = response.json()
    name = j["chunks"][i]
    cid = int(name[7]) - 1
    url = f"{prefix}problem/chunks/{name}?token={token}"
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(response.content.decode().rstrip())
    loc = f"../problem_data/{problem.problem_id}_{i}_{cid}.wav"
    with open(loc, "wb") as f:
        f.write(response.content)
    with wave.open(loc, "rb") as wf:
        buf = wf.readframes(wf.getnframes())
        data = np.frombuffer(buf, dtype="int16")
    return cid, data


def send_answer(problem_id, answers):
    url = f"{prefix}problem?token={token}"
    body = {"problem_id": problem_id, "answers": answers}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(body), headers=headers)
    if response.status_code != 200:
        raise ValueError(response.content.decode().rstrip())
    j = response.json()
    return j["accepted_at"]


def get_local_match_data():
    with open(f"../problem_data/match.json", "r") as f:
        j = json.load(f)
        return Match(j["problems"], j["bonus_factor"], j["change_penalty"], j["wrong_penalty"], j["correct_point"])


# ローカルに保存した問題でテストする用
problem_ids = ["1-3-1", "1-3-2", "1-3-3"]


def get_local_problem_data(match):
    pid = problem_ids[match.problem_number]
    with open(f"../problem_data/{pid}.json", "r") as f:
        j = json.load(f)
        return Problem(match, j["id"], j["chunks"], j["start_at"], j["time_limit"], j["data"])


def get_local_chunk_data(problem, i):
    name = None
    for name in os.listdir("../problem_data"):
        if name.startswith(f"{problem.problem_id}_{i}_") and name.endswith(".wav"):
            break
    cid = int(name[-5])
    with wave.open(f"../problem_data/{name}", "rb") as wf:
        buf = wf.readframes(wf.getnframes())
        data = np.frombuffer(buf, dtype="int16")
    return cid, data
