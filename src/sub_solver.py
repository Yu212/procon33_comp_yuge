# 並列化のためにsolver.pyから切り離したコード群

import math

import librosa
import numpy as np


def matchings(spec, cand, specs, minw, maxw, fit):
    res = []
    for name, sspec in zip(cand, specs):
        for cde in matching(spec, sspec, minw, maxw, fit):
            res.append((name, *cde))
        print(f"matching {name} finished")
    return res


def matching(problem, speech, minw, maxw, fit):
    pw = problem.shape[1]
    sw = speech.shape[1]
    sd = speech ** 2
    pd = 1 / (problem + 1) ** 2
    z = []
    sa = np.zeros(sw)
    pa = np.zeros(pw)
    aa = 0
    for d in range(-pw, sw):
        f = max(0, -d)
        t = min(sw-d, pw)
        zz = np.zeros(t - f)
        z.append(zz)
        for x in range(f, t):
            y = d+x
            c = sd[:, y].dot(pd[:, x])
            logc = math.log(c)
            zz[x-f] = logc
            sa[y] += logc
            pa[x] += logc
            aa += logc
    aa /= pw*sw
    sa /= pw
    pa /= sw
    pa = np.clip(pa, 0, 30)
    res = []

    def __matching(mode):
        sl = 0 if mode == "and" or mode == "left" else minw-pw
        sr = sw-pw+1 if mode == "and" or mode == "right" else sw-minw+1
        for d in range(sl, sr):
            f = max(0, -d)
            t = min(sw-d, pw)
            v = z[d+pw] - sa[d+f:d+t] - pa[f:t] + aa
            sv, i = find_min_avg_window_2(v, minw, min(len(v), maxw), mode)
            res.append((sv, i+f, i+f+d))

    if fit == "or":
        __matching("left")
        __matching("right")
    else:
        __matching(fit)
    res.sort(key=lambda r: r[0])
    res2 = []
    used = set()
    for val, rl, tl in res:
        if all(rl-tl+i not in used for i in range(-5, 6)):
            res2.append((val, rl, tl))
            if len(res2) >= 5:
                break
        used.add(rl-tl)
    return res2


def solve(problem_raw, speech_raw, name, cid, rl, tl, fit, hop1, hop2, height2, min_window, use_v2, open_hp, sera, cor):
    min_window = min(len(problem_raw), min_window)
    print(f"solve #{cid} {name}, {rl}, {tl}, {cor}")
    if use_v2:
        offset = find_offset_v2(problem_raw, speech_raw, rl, tl, name, hop1, hop2, height2, min_window, open_hp, sera, cor)
    else:
        offset = find_offset(problem_raw, speech_raw, rl, tl, name, hop1, hop2, height2, min_window, cor)
    if offset is None:
        return cid, name, None, None
    trim = find_trim(problem_raw, speech_raw, offset, fit, hop2, height2, min_window)
    if trim is None:
        return cid, name, None, None
    return cid, name, offset, trim


def find_offset_v2(problem_raw, speech_raw, rl, tl, name, hop1, hop2, height2, min_window, open_hp, sera, cor):
    ws = min_window
    pw = len(problem_raw)
    sw = len(speech_raw)
    if pw < ws:
        ws = pw
        hop1 //= 2
    if pw < rl+ws:
        tl = max(0, tl-(rl-(pw-ws)))
        rl = pw-ws
    if sw < tl+ws:
        rl = max(0, rl-(tl-(sw-ws)))
        tl = sw-ws
    bspec = np.absolute(problem_raw[rl:rl+ws]).sum()

    def f(d):
        removed = problem_raw[rl:rl+ws].copy()
        if pw < ws + d or sw < ws - d:
            return math.inf
        if rl-d < 0:
            removed[d-rl:] -= speech_raw[0:rl-d+ws]
        elif sw < rl-d+ws:
            removed[:sw-(rl-d)] -= speech_raw[rl-d:sw]
        else:
            removed -= speech_raw[rl-d:rl-d+ws]
        rspec = np.absolute(removed).sum()
        return rspec-bspec

    low = -hop1*sera+1
    high = hop1*sera
    a = np.zeros(high - low)
    md1 = None
    mv1 = -20*ws
    for i in range(low, high):
        d = rl-(tl-i)
        a[i-low] = f(d)
        if mv1 > a[i-low]:
            mv1 = a[i-low]
            md1 = d
    mi = np.argpartition(a, 10)[:10]
    mi = mi[np.argsort(a[mi])]

    if open_hp:
        if md1 is None or a[mi[9]]/mv1 > 0.99 or a[mi[9]]-mv1 < 8*ws:
            return None
    else:
        if md1 is None or a[mi[9]]/mv1 > 0.96 or a[mi[9]]-mv1 < 8*ws:
            return None

    bspec = melspectrogram_db(problem_raw[rl:rl+ws], hop2, height2)

    def g(i):
        d = i+low+rl-tl
        removed = problem_raw[rl:rl+ws].copy()
        if pw < ws + d or sw < ws - d:
            return -math.inf
        if rl-d < 0:
            removed[d-rl:] -= speech_raw[0:rl-d+ws]
        elif sw < rl-d+ws:
            removed[:sw-(rl-d)] -= speech_raw[rl-d:sw]
        else:
            removed -= speech_raw[rl-d:rl-d+ws]
        rspec = melspectrogram_db(removed, hop2, height2)
        return ((bspec-rspec)**3).sum()

    mig = sorted([(-g(i), i+low+rl-tl) for i in mi])
    vi = -mig[0][0]
    vj = -mig[3][0]
    if open_hp:
        if 30000 < vi and vi > max(vj, 1) * 1.02:
            return mig[0][1]
    else:
        if 30000 < vi and vi > max(vj, 1) * 1.05:
            return mig[0][1]
    return None


def find_offset(problem_raw, speech_raw, rl, tl, name, hop1, hop2, height2, min_window, cor):
    ws = min_window
    pw = len(problem_raw)
    sw = len(speech_raw)
    if pw < ws:
        ws = pw
        hop1 //= 2

    if pw < rl+ws:
        tl = max(0, tl-(rl-(pw-ws)))
        rl = pw-ws
    if sw < tl+ws:
        rl = max(0, rl-(tl-(sw-ws)))
        tl = sw-ws
    bspec = melspectrogram_db(problem_raw[rl:rl+ws], hop2, height2)

    def f(d):
        removed = problem_raw[rl:rl+ws].copy()
        if pw < ws + d or sw < ws - d:
            return -math.inf
        if rl-d < 0:
            removed[d-rl:] -= speech_raw[0:rl-d+ws]
        elif sw < rl-d+ws:
            removed[:sw-(rl-d)] -= speech_raw[rl-d:sw]
        else:
            removed -= speech_raw[rl-d:rl-d+ws]
        rspec = melspectrogram_db(removed, hop2, height2)
        return ((bspec-rspec)**3).sum()

    low = -hop1*3//2+1
    high = hop1*3//2
    searched = 0
    a = np.array([None] * (high - low))
    md = None
    vi = 0
    vj = None
    vk = None
    mi = high
    mj = high
    early = [False] * (high - low)
    for j in range(150):
        i = high*(j*2+1)//150+low
        early[i-low] = True
        d = rl-(tl-i)
        a[i-low] = f(d)
        searched += 1
        if j == 0 or vi < a[i-low]:
            vk = vj
            vj = vi
            vi = a[i-low]
            mj = mi
            mi = i
            md = d
        elif vj is None or vj < a[i-low]:
            mj = i
            vk = vj
            vj = a[i-low]
        elif vk is None or vk < a[i-low]:
            vk = a[i-low]
    center = mi
    for i in sorted(range(low, high), key=lambda k: abs(k-center)):
        d = rl-(tl-i)
        if not early[i-low]:
            a[i-low] = f(d)
            searched += 1
            if vi < a[i-low]:
                vk = vj
                vj = vi
                vi = a[i-low]
                mj = mi
                mi = i
                md = d
            elif vj < a[i-low]:
                mj = i
                vk = vj
                vj = a[i-low]
            elif vk < a[i-low]:
                vk = a[i-low]
        dist = abs(i-center)-abs(mi-center)
        if dist > 100 and vi > max(0, vk)+1000000 and vi > vk*2:
            break
        if dist > 300 and vi > max(0, vk)+100000 and vi > vk*1.5:
            break
        if dist > 500 and vi > max(0, vk)+100000 and vi > vk*1.2:
            break
        if searched > 600 and vi < -100000:
            md = None
            break
        if searched > 1200 and vi < -10000:
            md = None
            break
        if searched > 2000 and vi < 0:
            md = None
            break
        if searched > high:
            break
    if vi <= max(0, vk)+100000 and (vi <= max(0, vk)+50000 or vi <= vk*1.1):
        md = None
    return md


def find_trim(problem_raw, speech_raw, md, fit, hop2, height2, min_window):
    pw = len(problem_raw)
    sw = len(speech_raw)
    if md < 0:
        w = min(sw + md, pw)
        removed = problem_raw[0:w] - speech_raw[-md:w-md]
        dspec = melspectrogram_db(problem_raw[0:w], hop2, height2) - melspectrogram_db(removed, hop2, height2)
        rl = 0
    else:
        w = min(pw - md, sw)
        removed = problem_raw[md:md+w] - speech_raw[0:w]
        dspec = melspectrogram_db(problem_raw[md:md+w], hop2, height2) - melspectrogram_db(removed, hop2, height2)
        rl = md
    a = dspec.sum(axis=0)
    if fit == "or":
        r1 = find_max_sum_window_2(a, min_window//hop2, "left")
        r2 = find_max_sum_window_2(a, min_window//hop2, "right")
        sv, i, j = max(r1, r2)
    else:
        sv, i, j = find_max_sum_window_2(a, min_window//hop2, fit)
    if sv <= 0:
        return None
    return rl+i*hop2-md, min(sw, rl+j*hop2-md)


def melspectrogram(data, hop, n_mels=128):
    return librosa.feature.melspectrogram(y=data.astype(float), sr=48000, n_fft=hop*4, hop_length=hop, n_mels=n_mels)


def melspectrogram_db(data, hop, n_mels=128):
    return librosa.power_to_db(melspectrogram(data, hop, n_mels))


def find_max_sum_window(a, m, fit):
    n = len(a)
    if fit == "and":
        return a.sum(), 0, n
    elif fit == "left":
        cs = a.cumsum()
        sr = cs[m-1:].argmax()
        return cs[sr+m-1], 0, sr+m
    elif fit == "right":
        cs = a.cumsum()
        sl = cs[:n-m].argmin()
        if cs[sl] > 0:
            return cs[-1], 0, n
        return cs[-1]-cs[sl], sl+1, n
    else:
        cs = a.cumsum()
        sl = 0
        sr = cs[m-1:].argmax()+m
        sv = cs[sr-1]
        for i in range(m, n):
            r = cs[i:].argmax()
            v = cs[i+r]-cs[i-m]
            if sv < v:
                sv = v
                sl = i+1-m
                sr = i+1+r
        return sv, sl, sr


def find_min_avg_window(a, m1, m2, fit):
    n = len(a)
    if fit == "and":
        return a.mean(), 0
    elif fit == "left":
        cm = a.cumsum()[m1-1:m2]/np.arange(m1, m2+1)
        sr = cm.argmin()
        return cm[sr], 0
    elif fit == "right":
        cm = a[::-1].cumsum()[m1-1:m2]/np.arange(m1, m2+1)
        sr = cm.argmin()
        return cm[sr], n-sr-m1
    elif n < 50:
        cs = a.cumsum()
        si = 0
        sr = (cs[m1-1:m2]/np.arange(m1, m2+1)).argmin()+m1
        sv = cs[sr-1]/sr
        for i in range(m1, n):
            to = min(n, i-m1+m2+1)
            r = ((cs[i:to]-cs[i-m1])/np.arange(m1, m1+to-i)).argmin()
            v = (cs[i+r]-cs[i-m1])/(m1+r)
            if sv > v:
                sv = v
                si = i+1-m1
        return sv, si
    else:
        ll, rr = a.min()-0.1, a.max()
        bi = None
        for i in range(20):
            mid = (ll + rr) / 2
            ml = 0
            vl = 0
            sl = 0
            sr = (a[:m1]-mid).sum()
            for j in range(m1, n):
                if sr < vl:
                    bi = ml
                    rr = mid
                    break
                sl += a[j-m1]-mid
                sr += a[j]-mid
                if vl < sl:
                    vl = sl
                    ml = j-m1+1
            else:
                if sr < vl:
                    bi = ml
                    rr = mid
                else:
                    ll = mid
        return ll, bi


def find_max_sum_window_2(a, m, fit):
    n = len(a)
    if fit == "and":
        return a.sum(), 0, n
    elif fit == "left":
        sv = -math.inf
        s = a[0:m-1].sum()
        sr = 0
        for j in range(m, n+1):
            s += a[j-1]
            if sv < s:
                sv = s
                sr = j
        return sv, 0, sr
    elif fit == "right":
        sv = -math.inf
        sl = 0
        s = a[n-m+1:n].sum()
        for j in range(m, n+1):
            s += a[n-j]
            if sv < s:
                sv = s
                sl = n-j
        return sv, sl, n
    else:
        sv = -math.inf
        sl = 0
        sr = 0
        for i in range(0, n-m+1):
            s = a[i:i+m-1].sum()
            for j in range(i+m-1, n):
                s += a[j]
                if sv < s:
                    sv = s
                    sl = i
                    sr = j
        return sv, sl, sr


def find_min_avg_window_2(a, m, m2, fit):
    n = len(a)
    if fit == "and":
        return a.mean(), 0
    elif fit == "left":
        sv = math.inf
        s = a[0:m-1].sum()
        for j in range(m, n+1):
            s += a[j-1]
            if sv*j > s:
                sv = s/j
        return sv, 0
    elif fit == "right":
        sv = math.inf
        si = 0
        s = a[n-m+1:n].sum()
        for j in range(m, n+1):
            s += a[n-j]
            if sv*j > s:
                sv = s/j
                si = n-j
        return sv, si
    elif n < 40:
        sv = math.inf
        si = 0
        for i in range(0, n-m+1):
            s = a[i:i+m-1].sum()
            for j in range(i+m-1, n):
                s += a[j]
                if sv*(j+1-i) > s:
                    sv = s/(j+1-i)
                    si = i
        return sv, si
    else:
        ll, rr = a.min()-0.1, a.max()
        bi = None
        for i in range(20):
            mid = (ll + rr) / 2
            ml = 0
            vl = 0
            sl = 0
            sr = (a[:m]-mid).sum()
            for j in range(m, n):
                if sr < vl:
                    bi = ml
                    rr = mid
                    break
                sl += a[j-m]-mid
                sr += a[j]-mid
                if vl < sl:
                    vl = sl
                    ml = j-m+1
            else:
                if sr < vl:
                    bi = ml
                    rr = mid
                else:
                    ll = mid
        return ll, bi
