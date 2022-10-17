import concurrent
import time
from concurrent.futures import ProcessPoolExecutor, wait, as_completed, FIRST_COMPLETED

from src import sub_solver
from src.utils import *
from pydivsufsort import divsufsort, kasai


class Solver:
    def __init__(self, parallel=True):
        self.problem = None
        self.hop1 = 1024
        self.hop2 = 256
        self.height1 = 256
        self.height2 = 64
        self.min_window = 48000
        self.max_window = 48000*4
        self.allow_fail = 8
        self.interrupt = False
        self.specs = {name: melspectrogram(speech, self.hop1, self.height1) for name, speech in speech_items()}
        self.specs2 = {name: melspectrogram(speech, self.hop1//2, self.height1) for name, speech in speech_items()}
        self.max_workers = 6
        self.fit = "none"
        if parallel:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
            for i in range(self.max_workers):
                self.executor.submit(lambda: ())

    def set_problem(self, problem):
        self.problem = problem

    def set_fit(self, fit):
        self.fit = fit

    def adjust_all(self, open_hp, sera, callback, log):
        groups = self.problem.make_groups(range(self.problem.num_chunks))
        callback()
        for group in groups:
            for name in group.using_speeches():
                if self.interrupt:
                    return
                self.adjust(group, name, open_hp, sera, callback, log)

    def adjust(self, group, name, open_hp, sera, callback, log):
        problem_raw = group.rejected_data(name)
        speech_raw = speech_data(name)
        offset = group.offset(name)
        trim = group.trim(name)
        minw = min(self.min_window, len(problem_raw))

        tl = max(trim[0], (trim[0]+trim[1]-minw)//2)
        rl = tl+offset
        _, _, offset2, trim2 = sub_solver.solve(problem_raw, speech_raw, name, 0, rl, tl, "none", self.hop1, self.hop2, self.height2, self.min_window, True, open_hp, sera, None)
        if offset2 is None:
            log(f"{name} is probably wrong")
            return
        log(f"{group.f}:{group.t}, {name}, {offset},{trim} => {offset2},{trim2}")
        group.adjust_speech(name, trim2, offset=offset2)
        callback()

    def solve_debug(self, group, callback, log):
        n = len(group.wf())
        name = random.choice(group.remaining_speeches(False))
        log(name)
        v = speech_len(name)
        le = random.randrange(24000, min(n, v)+1)
        pl = max(0, min(n - le, random.randrange(-48000, n - le + 48000)))
        sl = max(0, min(v - le, random.randrange(-48000, v - le + 48000)))
        if le < 48000:
            pl = random.randint(0, 1) * (n - le)
        group.use_speech(name, pl - sl, (sl, sl + le))
        callback()

    def solve(self, group, exclude_used_speeches, use_v2, open_hp, sera, callback, log, cor_offsets=None):
        log(f"hp: {open_hp}, sera: {sera}")
        start = time.time()
        successed = 0
        failed = 0
        cand = group.remaining_speeches(exclude_used_speeches)
        hop = self.hop1
        if group.wf_len() < self.min_window:
            hop //= 2
        matches = self.matchings(group.wf(), cand, self.min_window//hop-2, self.max_window//hop, hop, log, cor_offsets=cor_offsets)
        log(f"{round(time.time() - start, 1)} sec")
        start = time.time()
        cid = 0
        obj = max(self.problem.num_total_speeches, len(self.problem.using_speeches) + 1)
        for name, val, rl, tl in matches:
            if self.interrupt:
                break
            if (successed >= 1 and failed >= self.allow_fail) or obj <= len(self.problem.using_speeches):
                break
            if group.using_speeches().size > 0 and (name in group.using_speeches() or toggle_ej(name) in group.using_speeches()):
                continue
            cor = cor_offsets is not None and (cor_offsets[name] if name in cor_offsets and cor_offsets[name] in range(rl*hop-(tl*hop-(-hop*3//2+1)), rl*hop-(tl*hop-(hop*3//2))) else None)
            cid, found, offset, trim = sub_solver.solve(group.wf(), speech_data(name), name, cid, rl*hop, tl*hop, self.fit, self.hop1, self.hop2, self.height2, self.min_window, use_v2, open_hp, sera, cor)
            cid += 1
            log(f"#{cid} {name} start")
            if offset is None:
                log(f"{found}, None")
                failed += 1
            else:
                log(f"{found}, {offset}, {trim}")
                successed += 1
                if group.using_speeches().size > 0 and found in group.using_speeches():
                    log(f"already used! {found}")
                    continue
                group.use_speech(found, offset, trim)
                callback()
        log(f"{round(time.time() - start, 1)} sec, {successed}/{successed+failed} successed")

    def matchings(self, wf, cand, minw, maxw, hop, log, cor_offsets=None):
        spec = melspectrogram(wf, hop, self.height1)
        minw = min(minw, spec.shape[1])
        res = []
        for name in cand:
            if self.interrupt:
                break
            sspec = self.specs[name] if hop == self.hop1 else self.specs2[name]
            cdes = sub_solver.matching(spec, sspec, minw, maxw, self.fit)
            best = None
            for cde in cdes:
                if best is None or best[1] > cde[0]:
                    best = (name, *cde)
                res.append((name, *cde))
            log(best)
        res.sort(key=lambda r: r[1])
        for name, val, rl, tl in res[:30]:
            cor = cor_offsets is not None and (name in cor_offsets and cor_offsets[name] in range(rl*hop-(tl*hop-(-hop*3//2+1)), rl*hop-(tl*hop-(hop*3//2))))
            print(name, val, (rl-tl)*hop, cor)
        return res

    def solve_parallel(self, group, exclude_used_speeches, use_v2, open_hp, sera, callback, log, cor_offsets=None):
        start = time.time()
        successed = 0
        failed = 0
        cand = group.remaining_speeches(exclude_used_speeches)
        hop = self.hop1
        if group.wf_len() < self.min_window:
            hop //= 2
        matches = self.matchings_parallel(group.wf(), cand, self.min_window//hop-2, self.max_window//hop, hop, log, cor_offsets=cor_offsets)
        log(f"{round(time.time() - start, 1)} sec")
        start = time.time()
        futures = {}
        names = {}
        cid = 0
        obj = max(self.problem.num_total_speeches, len(self.problem.using_speeches) + 1)
        for name, val, rl, tl in matches:
            if self.interrupt:
                break
            if len(futures) >= self.max_workers:
                callback()
                ret = wait(futures.values(), return_when=concurrent.futures.FIRST_COMPLETED)
                for future in ret.done:
                    rid, found, offset, trim = future.result()
                    futures.pop(rid)
                    names.pop(rid)
                    if offset is None:
                        log(f"{found}, None")
                        failed += 1
                    else:
                        log(f"{found}, {offset}, {trim}")
                        successed += 1
                        if group.using_speeches().size > 0 and found in group.using_speeches():
                            log(f"already used! {found}")
                            continue
                        group.use_speech(found, offset, trim)
                if (successed >= 1 and failed >= self.allow_fail) or obj <= len(self.problem.using_speeches):
                    break
            if self.interrupt:
                break
            if group.using_speeches().size > 0 and (name in group.using_speeches() or toggle_ej(name) in group.using_speeches()):
                continue
            cor = cor_offsets is not None and (cor_offsets[name] if name in cor_offsets and cor_offsets[name] in range(rl*hop-(tl*hop-(-hop*3//2+1)), rl*hop-(tl*hop-(hop*3//2))) else None)
            future = self.executor.submit(sub_solver.solve, group.wf(), speech_data(name), name, cid, rl*hop, tl*hop, self.fit, self.hop1, self.hop2, self.height2, self.min_window, use_v2, open_hp, sera, cor)
            futures[cid] = future
            names[cid] = name
            cid += 1
            log(f"#{cid} {name} start: [{','.join(names.values())}]")
        log(f"!{futures.keys()}")
        ret = wait(futures.values(), timeout=0 if self.interrupt else 1)
        for future in ret.done:
            cid, found, offset, trim = future.result()
            futures.pop(cid)
            names.pop(cid)
            if offset is None:
                failed += 1
                log(f"!{found}, None")
            else:
                log(f"!{found}, {offset}, {trim}")
                successed += 1
                group.use_speech(found, offset, trim)
        for name in names.values():
            log(f"{name} timed out")
        callback()
        log(f"{round(time.time() - start, 1)} sec, {successed}/{successed+failed} successed")

    def matchings_parallel(self, wf, cand, minw, maxw, hop, log, cor_offsets=None):
        spec = melspectrogram(wf, hop, self.height1)
        minw = min(minw, spec.shape[1])
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
                log(best)
        res.sort(key=lambda r: r[1])
        for name, val, rl, tl in res[:30]:
            cor = cor_offsets is not None and (name in cor_offsets and cor_offsets[name] in range(rl*hop-(tl*hop-(-hop*3//2+1)), rl*hop-(tl*hop-(hop*3//2))))
            print(name, val, (rl-tl)*hop, cor)
        return res

    def solve_sa(self, group, exclude_used_speeches, callback, log):
        min_window = 1200
        start = time.time()
        cand = group.remaining_speeches(exclude_used_speeches)
        problem = group.wf()
        wf_len = len(problem) + sum(speech_len(name) + 1 for name in cand)
        minv = min(problem.min(), *[speech_data(name).min() for name in cand])-2
        wf = np.zeros(wf_len, dtype="int32")
        wf[:len(problem)] = problem
        wf[:len(problem)] -= minv
        wf[len(problem)] = 1
        index = len(problem)+1
        for name in cand:
            wf[index:index+speech_len(name)] = speech_data(name)
            wf[index:index+speech_len(name)] -= minv
            index += speech_len(name)+1
        if self.interrupt:
            return
        sa = divsufsort(wf)
        if self.interrupt:
            return
        lcp = kasai(wf, sa)
        if self.interrupt:
            return
        cond = sa < len(problem)
        offsets = {}
        for i in np.where((lcp >= min_window) & (cond ^ np.append(cond[1:], False)))[0]:
            if i == 0:
                continue
            if sa[i] < len(problem):
                r = sa[i]
                t = sa[i+1]-len(problem)-1
            else:
                r = sa[i+1]
                t = sa[i]-len(problem)-1
            for name in cand:
                slen = speech_len(name)+1
                if t < slen:
                    if name not in offsets or offsets[name][2] < lcp[i]:
                        offsets[name] = (r-t, t, lcp[i])
                    break
                t -= slen
        for name, (offset, tl, length) in offsets.items():
            trim = (tl, tl+length)
            trim2 = sub_solver.find_trim(problem, speech_data(name), offset, "none", self.hop2, self.height2, 12000)
            if trim2 is not None and trim2[0]-2400 <= trim[0] < trim[1] <= trim2[1]+2400:
                trim = trim2
            group.use_speech(name, offset, trim)
        callback()
        log(f"{list(offsets.keys())}, {round(time.time() - start, 1)} sec")
