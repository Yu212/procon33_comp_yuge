import time

import numpy as np

from src import backup
from src.utils import speech_data, speech_names, toggle_ej


class Problem:
    def __init__(self, match, problem_id, num_chunks, starts_at, time_limit, num_total_speeches):
        self.match = match
        self.problem_id = problem_id

        self.num_chunks = num_chunks
        self.starts_at = starts_at
        self.time_limit = time_limit
        self.num_total_speeches = num_total_speeches
        self.num_got_chunks = 0
        self.chunks = np.empty(num_chunks, dtype=np.ndarray)
        self.chunks_removed = np.empty(num_chunks, dtype=np.ndarray)
        self.using_speeches = []
        self.chunk_using_speeches = [[] for _ in range(num_chunks)]
        self.speech_offsets = [[] for _ in range(num_chunks)]
        self.speech_trims = [[] for _ in range(num_chunks)]

    def time_elapsed(self):
        return time.time() - self.starts_at

    def add_chunk(self, cid, data):
        self.chunks[cid] = data
        self.chunks_removed[cid] = data.copy()
        self.num_got_chunks += 1
        backup.write(self.match)

    def width(self, hop):
        last_chunk = -1
        for i in range(self.num_chunks):
            if self.chunks[i] is not None:
                last_chunk = i
        return self.chunk_offset(last_chunk + 1, hop)

    def chunk_offset(self, cid, hop):
        offset = 0
        for i in range(cid):
            if self.chunks[i] is not None:
                offset += len(self.chunks[i]) // hop + 1
            elif i == 0 or self.chunks[i - 1] is not None:
                offset += 100
        return offset

    def chunk_diff(self, i, j):
        sign = np.sign(i - j)
        i, j = sorted([i, j])
        if any(a is None for a in self.chunks[i:j+1]):
            return None
        return sign * sum(map(len, self.chunks[i:j]))

    def use_speech(self, cid, name, offset, trim):
        if name not in self.using_speeches:
            self.using_speeches.append(name)
        speech = speech_data(name)
        self.chunks_removed[cid][offset+trim[0]:offset+trim[1]] -= speech[trim[0]:trim[1]]
        self.chunk_using_speeches[cid].append(name)
        self.speech_offsets[cid].append(offset)
        self.speech_trims[cid].append(trim)

    def adjust_speech(self, cid, name, trim, new_offset=None):
        i = self.chunk_using_speeches[cid].index(name)
        offset = self.speech_offsets[cid][i]
        if new_offset is None:
            new_offset = offset
        old_trim = self.speech_trims[cid][i]
        speech = speech_data(name)
        self.chunks_removed[cid][offset+old_trim[0]:offset+old_trim[1]] += speech[old_trim[0]:old_trim[1]]
        self.chunks_removed[cid][new_offset+trim[0]:new_offset+trim[1]] -= speech[trim[0]:trim[1]]
        self.speech_trims[cid][i] = trim

    def reject_speech(self, name, cid=None):
        if cid is None:
            for i in range(self.num_chunks):
                self.reject_speech(name, cid=i)
            return
        if name in self.chunk_using_speeches[cid]:
            j = self.chunk_using_speeches[cid].index(name)
            offset = self.speech_offsets[cid][j]
            trim = self.speech_trims[cid][j]
            print(j, offset, trim)
            self.chunks_removed[cid][offset+trim[0]:offset+trim[1]] += speech_data(name)[trim[0]:trim[1]]
            self.chunk_using_speeches[cid].pop(j)
            self.speech_offsets[cid].pop(j)
            self.speech_trims[cid].pop(j)
            if all(name not in a for a in self.chunk_using_speeches):
                self.using_speeches.remove(name)

    def make_groups(self, cids):
        groups = []
        f = None
        for i in range(self.num_chunks + 1):
            use = i < self.num_chunks and self.chunks[i] is not None and i in cids
            if not use and f is not None:
                groups.append(self.make_group(f, i))
                f = None
            if use and f is None:
                f = i
        return groups

    def make_group(self, f, t):
        offsets = {}
        for i in range(f, t):
            for name, offset in zip(self.chunk_using_speeches[i], self.speech_offsets[i]):
                diff = self.chunk_diff(f, i)
                if name not in offsets:
                    offsets[name] = offset - diff
                elif offsets[name] != offset - diff:
                    self.reject_speech(name, cid=i)
        return Group(self, f, t)


class Group:
    def __init__(self, problem, f, t):
        self.problem = problem
        self.f = f
        self.t = t

    def using_speeches(self):
        return np.unique(np.hstack(self.problem.chunk_using_speeches[self.f:self.t]))

    def wf(self):
        return np.hstack(self.problem.chunks_removed[self.f:self.t])

    def wf_len(self):
        return sum(len(c) for c in self.problem.chunks_removed[self.f:self.t])

    def offset(self, name):
        for i in range(self.f, self.t):
            if name in self.problem.chunk_using_speeches[i]:
                j = self.problem.chunk_using_speeches[i].index(name)
                diff = self.problem.chunk_diff(self.f, i)
                return self.problem.speech_offsets[i][j] - diff

    def trim(self, name):
        tl = None
        tr = None
        for i in range(self.f, self.t):
            if name in self.problem.chunk_using_speeches[i]:
                j = self.problem.chunk_using_speeches[i].index(name)
                trim = self.problem.speech_trims[i][j]
                tl = trim[0] if tl is None else tl
                tr = trim[1]
        return tl, tr

    def use_speech(self, name, offset, trim):
        for i in range(self.f, self.t):
            diff = self.problem.chunk_diff(self.f, i)
            w = len(self.problem.chunks[i])
            lt = max(-(offset+diff), trim[0])
            rt = min(w-(offset+diff), trim[1])
            if lt < rt:
                self.problem.use_speech(i, name, offset+diff, (lt, rt))

    def rejected_data(self, name):
        data = self.wf()
        speech = speech_data(name)
        for i in range(self.f, self.t):
            if name not in self.problem.chunk_using_speeches[i]:
                continue
            j = self.problem.chunk_using_speeches[i].index(name)
            diff = self.problem.chunk_diff(self.f, i)
            offset = self.problem.speech_offsets[i][j] - diff
            trim = self.problem.speech_trims[i][j]
            # 最後までバグの原因がわからなかったのでログだけ出して諦めた
            if not (0 <= trim[0] < trim[1] <= len(speech)):
                print(f"reject failed 1 {name} {trim[0]}, {trim[1]}")
                continue
            if not (0 <= offset+trim[0] < offset+trim[1] <= len(data)):
                print(f"reject failed 2 {name} {offset+trim[0]}, {offset+trim[1]}")
                continue
            data[offset+trim[0]:offset+trim[1]] += speech[trim[0]:trim[1]]
        return data

    def adjust_speech(self, name, trim, offset=None):
        if offset is None:
            offset = self.offset(name)
        for i in range(self.f, self.t):
            diff = self.problem.chunk_diff(self.f, i)
            w = len(self.problem.chunks[i])
            lt = max(-(offset+diff), trim[0])
            rt = min(w-(offset+diff), trim[1])
            if lt < rt:
                if name in self.problem.chunk_using_speeches[i]:
                    self.problem.adjust_speech(i, name, (lt, rt), offset+diff)
                else:
                    self.problem.use_speech(i, name, offset+diff, (lt, rt))
            elif name in self.problem.chunk_using_speeches[i]:
                self.problem.reject_speech(name, i)

    def remaining_speeches(self, exclude_used_speeches):
        remaining = list(speech_names())
        if exclude_used_speeches:
            for name in self.problem.match.used_speeches:
                remaining.remove("E" + name)
                remaining.remove("J" + name)
        for name in self.using_speeches():
            remaining.remove(name)
            remaining.remove(toggle_ej(name))
        return remaining
