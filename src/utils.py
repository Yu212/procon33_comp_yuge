import random
import struct
import time
import wave
import numpy as np
import librosa
import librosa.display
from PIL import Image, ImageTk
from matplotlib import pyplot as plt


def __open_speech_data(name):
    with wave.open(f"../sample_data/speech_data/{name}.wav", "rb") as wf:
        buf = wf.readframes(wf.getnframes())
        return np.frombuffer(buf, dtype="int16")


def load_speeches():
    global speeches
    speeches = {}
    speeches.update({f"E{i:02}": __open_speech_data(f"E{i:02}") for i in range(1, 45)})
    speeches.update({f"J{i:02}": __open_speech_data(f"J{i:02}") for i in range(1, 45)})
    return speeches


def speech_data(name):
    return speeches[name]


def speech_len(name):
    return len(speeches[name])


def speech_items():
    return speeches.items()


def speech_names():
    return speeches.keys()


def toggle_ej(name):
    return ("J" if name[0] == "E" else "E") + name[1:]


def random_choose_speeches(num_speeches, cand="EJ"):
    ids = random.sample(range(44), num_speeches)
    chosen_names = [f"{random.choice(list(cand))}{ind+1:02}" for ind in ids]
    chosen_names.sort()
    return chosen_names


def melspectrogram(data, hop, n_mels=128):
    return librosa.feature.melspectrogram(y=data.astype(float), sr=48000, n_fft=hop*4, hop_length=hop, n_mels=n_mels)


def melspectrogram_db(data, hop, n_mels=128):
    return librosa.power_to_db(melspectrogram(data, hop, n_mels))


def save_wav(wav, name):
    with wave.open(f"../{name}.wav", "wb") as wf:
        out = struct.pack("h" * len(wav), *wav)
        wf.setnchannels(1)
        wf.setframerate(48000)
        wf.setsampwidth(2)
        wf.writeframes(out)


def gen_image(spec, cmap='magma', swap_rb=True):
    img = librosa.display.specshow(spec, x_axis='time', y_axis='mel', sr=48000, cmap=cmap)
    arr = img.to_rgba(img.get_array().reshape(spec.shape))
    if swap_rb:
        arr[:, :, [0, 2]] = arr[:, :, [2, 0]]
    arr = (arr[::-1, :, :] * 255).astype(np.uint8)
    return arr


def waveform_image(waveform, width, height):
    fig = plt.figure(figsize=(width/100, height/100))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.plot(waveform)
    ax.set_xlim(0, len(waveform))
    ax.set_ylim(-25000, 25000)
    fig.canvas.draw()
    plt.clf()
    plt.close()
    return ImageTk.PhotoImage(image=Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()))


def generate_problems(match):
    from src.problem import Problem
    rem = 44
    cands = random_choose_speeches(44)
    random.shuffle(cands)
    names = []
    while rem > 3 and len(names) < 10:
        m = random.randint(3, min(rem, 20))
        names.append(cands[rem-m:rem])
        rem -= m
    random.shuffle(names)
    problems = []
    sp = random.randint(2, 5)
    ns = [random.randint(sp * 24000, sp * 48000 * 4) for i in range(len(names))]
    for i, n, na in zip(range(len(names)), ns, names):
        m = len(na)
        print(m, na)
        s, na, idx, region, trim = generate_problem(n, m, sp, names=na, same_length=True)
        problem = Problem(match, f"problem_{i}", sp, time.time(), 600, m)
        offsets = {name: region[idx[name]][0] - trim[idx[name]][0] for name in na}
        problems.append((problem, np.random.permutation(range(sp)), s, offsets))
        print(na)
    return problems


def generate_problem(n, m, sp=1, names=None, cand="EJ", same_length=False):
    if names is None:
        names = random_choose_speeches(m, cand)
    idx = {name: i for i, name in enumerate(names)}
    region = []
    trim = []
    s = np.zeros(n, np.int16)
    for i, name in enumerate(names):
        v = speech_len(name)
        le = random.randrange(48000, min(n, v)+1)
        # le = min(n, v)
        pl = max(0, min(n - le, random.randrange(-48000, n - le + 48000)))
        sl = max(0, min(v - le, random.randrange(-48000, v - le + 48000)))
        region.append((pl, pl + le))
        trim.append((sl, sl + le))
        s[pl:pl+le] += speech_data(name)[sl:sl+le]
    if same_length:
        split = [n * i // sp for i in range(sp + 1)]
    else:
        while True:
            split = [0] + sorted(random.sample(range(n), sp-1)) + [n]
            if all(split[i+1]-split[i] >= 24000 for i in range(sp)):
                break
    s = [s[split[i]:split[i+1]] for i in range(sp)]
    return s, names, idx, region, trim
