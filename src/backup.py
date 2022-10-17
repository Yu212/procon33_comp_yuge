# 問題の途中でエラー落ちして再起動した場合にそれまでの結果を復元するための機能
# 本番一日目に突貫で作ってバグらせるのが怖かったので結局動かしすらしなかった

import json


no_backup = True


def restore():
    with open(f"../problem_data/backup.json", "r") as f:
        return json.load(f)


def write(match):
    if no_backup:
        return
    state = {"used": match.used_speeches,
             "number": match.problem_number,
             "during_problem": match.during_problem}
    if match.during_problem:
        problem = match.current_problem
        state["num_got_chunks"] = problem.num_got_chunks
        for cid in range(problem.num_chunks):
            if problem.chunks[cid] is None:
                continue
            state[str(cid)] = {}
            for name, offset, trim in zip(problem.chunk_using_speeches[cid], problem.speech_offsets[cid], problem.speech_trims[cid]):
                state[str(cid)][name] = {"offset": int(offset), "tl": int(trim[0]), "tr": int(trim[1])}
    with open(f"../problem_data/backup.json", "w") as f:
        f.write(json.dumps(state))
