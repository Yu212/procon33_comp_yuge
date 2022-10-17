# テスト用にGUIを経由しないソルバ

from src import connection, sub_solver
from src.match import Match
from src.problem import Problem
import time
import concurrent
from concurrent.futures import ProcessPoolExecutor, wait, as_completed, FIRST_COMPLETED

from src.utils import *


class NoGui:
    def __init__(self, parallel=True):
        self.hop1 = 2048
        self.hop2 = 256
        self.height1 = 256
        self.height2 = 64
        self.min_window = 24000
        self.max_window = 48000*4
        self.allow_fail = 6
        self.specs = {name: melspectrogram(speech, self.hop1, self.height1) for name, speech in speech_items()}
        self.specs2 = {name: melspectrogram(speech, self.hop1//2, self.height1) for name, speech in speech_items()}
        self.fit = "none"
        if parallel:
            self.max_workers = 6
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
            for i in range(self.max_workers):
                self.executor.submit(lambda: ())

    def start(self):
        n = 48000*4
        m = 20
        c = 5
        s, names, idx, region, trim = generate_problem(n, m, c, same_length=True)
        print(n, m, names)
        self.debug_offsets = {name: region[idx[name]][0] - trim[idx[name]][0] for name in names}
        match = Match(10, [4, 4, 3.3, 2.2, 1.1, 1], 100, 100, 100)
        self.problem = Problem(match, "problem_id", c, time.time(), 1000, m)
        for i in range(c):
            self.problem.add_chunk(i, s[i])
        group = self.problem.make_groups([2])[0]
        return self.solve(group, True, False, 3, {k: v+self.problem.chunk_diff(0, 2) for k, v in self.debug_offsets.items()})
        # return self.solve_parallel(group, {k: v+self.problem.chunk_diff(0, 2) for k, v in self.debug_offsets.items()})


    def solve(self, group, use_v2=True, open_hp=False, sera=3, cor_offsets=None):
        start = time.time()
        successed = 0
        failed = 0
        cand = group.remaining_speeches(False)
        hop = self.hop1
        if group.wf_len() < self.min_window:
            hop //= 2
        matches = self.matchings(group.wf(), cand, self.min_window//hop-2, self.max_window//hop, hop, cor_offsets=cor_offsets)
        print(f"{round(time.time() - start, 1)} sec")
        # return time.time() - start
        cid = 0
        obj = max(self.problem.num_total_speeches, len(self.problem.using_speeches) + 1)
        for name, val, rl, tl in matches:
            if (successed >= 1 and failed >= self.allow_fail) or obj <= len(self.problem.using_speeches):
                break
            if group.using_speeches().size > 0 and (name in group.using_speeches() or toggle_ej(name) in group.using_speeches()):
                continue
            cor = cor_offsets is not None and (cor_offsets[name] if name in cor_offsets and cor_offsets[name] in range(rl*hop-(tl*hop-(-hop*3//2+1)), rl*hop-(tl*hop-(hop*3//2))) else None)
            cid, found, offset, trim = sub_solver.solve(group.wf(), speech_data(name), name, cid, rl*hop, tl*hop, self.fit, self.hop1, self.hop2, self.height2, self.min_window, use_v2, open_hp, sera, cor)
            cid += 1
            print(offset, successed, failed, self.allow_fail)
            # print(f"#{cid} {name} start")
            if offset is None:
                # print(f"{found}, None")
                failed += 1
            else:
                # print(f"{found}, {offset}, {trim}")
                successed += 1
                if group.using_speeches().size > 0 and found in group.using_speeches():
                    # print(f"already used! {found}")
                    continue
                group.use_speech(found, offset, trim)
        print(f"{round(time.time() - start, 1)} sec, {successed}/{successed+failed} successed")
        return time.time() - start, successed, successed+failed


    def matchings(self, wf, cand, minw, maxw, hop, cor_offsets=None):
        spec = melspectrogram(wf, hop, self.height1)
        minw = min(minw, spec.shape[1])
        res = []
        for name in cand:
            sspec = self.specs[name] if hop == self.hop1 else self.specs2[name]
            cdes = sub_solver.matching(spec, sspec, minw, maxw, self.fit)
            best = None
            for cde in cdes:
                if best is None or best[1] > cde[0]:
                    best = (name, *cde)
                res.append((name, *cde))
        res.sort(key=lambda r: r[1])
        for name, val, rl, tl in res[:15]:
            cor = cor_offsets is not None and (name in cor_offsets and cor_offsets[name] in range(rl*hop-(tl*hop-(-hop*3//2+1)), rl*hop-(tl*hop-(hop*3//2))))
            print(name, val, (rl-tl)*hop, cor)
        return res

    def solve_parallel(self, group, cor_offsets=None):
        start = time.time()
        successed = 0
        failed = 0
        cand = group.remaining_speeches(False)
        hop = self.hop1
        if group.wf_len() < self.min_window:
            hop //= 2
        matches = self.matchings_parallel(group.wf(), cand, self.min_window//hop-2, self.max_window//hop, hop, cor_offsets=cor_offsets)
        print(f"{round(time.time() - start, 1)} sec")
        start = time.time()
        futures = {}
        names = {}
        cid = 0
        obj = max(self.problem.num_total_speeches, len(self.problem.using_speeches) + 1)
        for name, val, rl, tl in matches:
            if len(futures) >= self.max_workers:
                ret = wait(futures.values(), return_when=concurrent.futures.FIRST_COMPLETED)
                for future in ret.done:
                    rid, found, offset, trim = future.result()
                    futures.pop(rid)
                    names.pop(rid)
                    if offset is None:
                        print(f"{found}, None")
                        failed += 1
                    else:
                        print(f"{found}, {offset}, {trim}")
                        successed += 1
                        if group.using_speeches().size > 0 and found in group.using_speeches():
                            print(f"already used! {found}")
                            continue
                        group.use_speech(found, offset, trim)
                if (successed >= 1 and failed >= self.allow_fail) or obj <= len(self.problem.using_speeches):
                    break
            if group.using_speeches().size > 0 and (name in group.using_speeches() or toggle_ej(name) in group.using_speeches()):
                continue
            cor = cor_offsets is not None and (cor_offsets[name] if name in cor_offsets and cor_offsets[name] in range(rl*hop-(tl*hop-(-hop*3//2+1)), rl*hop-(tl*hop-(hop*3//2))) else None)
            future = self.executor.submit(sub_solver.solve, group.wf(), speech_data(name), name, cid, rl*hop, tl*hop, self.fit, self.hop1, self.hop2, self.height2, self.min_window, cor)
            futures[cid] = future
            names[cid] = name
            cid += 1
            print(f"#{cid} {name} start: [{','.join(names.values())}]")
        print(f"!{futures.keys()}")
        ret = wait(futures.values(), timeout=1)
        for future in ret.done:
            cid, found, offset, trim = future.result()
            futures.pop(cid)
            names.pop(cid)
            if offset is None:
                failed += 1
                print(f"!{found}, None")
            else:
                print(f"!{found}, {offset}, {trim}")
                successed += 1
                group.use_speech(found, offset, trim)
        for name in names.values():
            print(f"{name} timed out")
        print(f"{round(time.time() - start, 1)} sec, {successed}/{successed+failed} successed")
        return time.time() - start, successed, successed+failed

    def matchings_parallel(self, wf, cand, minw, maxw, hop, cor_offsets=None):
        spec = melspectrogram(wf, hop, self.height1)
        futures = []
        split = max(1, min(6, len(cand)//12))
        for i in range(split):
            sub = cand[i*len(cand)//split:(i+1)*len(cand)//split]
            if len(sub) == 0:
                continue
            print(f"matchings: {sub}")
            specs = [self.specs[name] if hop == self.hop1 else self.specs2[name] for name in sub]
            future = self.executor.submit(sub_solver.matchings, spec, sub, specs, minw, maxw, self.fit)
            futures.append(future)
        res = []
        for future in as_completed(futures):
            bests = {}
            for ncde in future.result():
                if ncde[0] not in bests or bests[ncde[0]][1] > ncde[1]:
                    bests[ncde[0]] = ncde
                res.append(ncde)
            for best in bests.values():
                print(best)
        res.sort(key=lambda r: r[1])
        for name, val, rl, tl in res[:30]:
            cor = cor_offsets is not None and (name in cor_offsets and cor_offsets[name] in range(rl*hop-(tl*hop-(-hop*3//2+1)), rl*hop-(tl*hop-(hop*3//2))))
            print(name, val, (rl-tl)*hop, cor)
        return res


if __name__ == "__main__":
    connection.load_token()
    load_speeches()
    nogui = NoGui(True)
    es = 0
    var = 0
    ts = 0
    for i in range(20):
        elapsed, successed, tried = nogui.start()
        es += elapsed
        var += successed / tried
        ts += tried
    print(es/20, var/20, ts/20)

    # es = 0
    # for i in range(30):
    #     elapsed = nogui.start()
    #     es += elapsed
    # print(es/30)
# 4.016099540392558

# parallel, none, 27.3560972214 0.4977612583494936 13.3
# serial, none, 46.86312787532806 0.5309215784215783 13.0

# parallel, or, 25.993107533454896 0.43059773559773556 12.0
# serial, or, 49.89264304637909 0.5601248751248751 13.8
