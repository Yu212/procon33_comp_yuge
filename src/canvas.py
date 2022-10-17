from tkinter import *

from src.utils import *


class GuiCanvas:
    def __init__(self, gui):
        self.gui = gui
        self.canvas_speech = gui.canvas_speech
        self.canvas_problem = gui.canvas_problem
        self.canvas_speech_name = gui.canvas_speech_name
        self.hop = 512
        self.height = 60
        self.chunk_images = None
        self.wf_images = None
        self.speech_images_gray = {}
        self.speech_images_magma = {}
        for name, data in speech_items():
            spec = melspectrogram_db(data, self.hop, n_mels=self.height)
            spec_img_gray = gen_image(spec, swap_rb=False, cmap="gray")
            spec_img_magma = gen_image(spec, swap_rb=False, cmap="magma")
            self.speech_images_gray[name] = Image.fromarray(spec_img_gray)
            self.speech_images_magma[name] = Image.fromarray(spec_img_magma)
        self.image_cache = []

    def clicked_chunk(self, event):
        if self.gui.solving:
            return
        x = int(self.canvas_problem.canvasx(event.x))
        if not self.gui.match.during_problem:
            return
        problem = self.gui.match.current_problem
        for i in range(problem.num_chunks):
            if problem.chunks[i] is None:
                continue
            co = problem.chunk_offset(i, self.hop)
            cw = len(problem.chunks[i])//self.hop+1
            if co <= x < co + cw:
                if i in self.gui.selected_chunks:
                    self.gui.selected_chunks.remove(i)
                    self.canvas_problem.itemconfig(f"chunk_text_{i}", text=f"#{i+1}", fill="black")
                else:
                    self.gui.selected_chunks.append(i)
                    self.canvas_problem.itemconfig(f"chunk_text_{i}", text=f"#{i+1}*", fill="red")
                break

    def clicked_reject_button(self, event):
        if self.gui.solving:
            return
        y = int(self.canvas_speech_name.canvasy(event.y)) // self.height
        problem = self.gui.match.current_problem
        rejected = sorted(problem.using_speeches)[y]
        problem.reject_speech(rejected)
        self.gui.set_button_state("11110011")
        self.gui.draw()

    def start_problem(self, problem):
        self.chunk_images = [None] * problem.num_chunks
        self.wf_images = [None] * problem.num_chunks

    def draw(self):
        def text_background(canvas, item, bg):
            canvas.create_rectangle(canvas.bbox(item), fill=bg, width=0)
            canvas.tag_raise(item)
        self.image_cache.clear()
        h = self.height
        if self.gui.match is None or not self.gui.match.during_problem:
            problem = None
            canvas_width = 1150
            canvas_height = 600
        else:
            problem = self.gui.match.current_problem
            canvas_width = max(1150, problem.width(self.hop))
            canvas_height = max(600, len(problem.using_speeches)*h)
        self.canvas_speech.delete(ALL)
        self.canvas_problem.delete(ALL)
        self.canvas_speech_name.delete(ALL)
        self.canvas_speech.config(scrollregion=(0, 0, canvas_width, canvas_height))
        self.canvas_problem.config(scrollregion=(0, 0, canvas_width, 0))
        self.canvas_speech_name.config(scrollregion=(0, 0, 0, canvas_height))
        for i in range(0, canvas_height, h*2):
            self.canvas_speech_name.create_rectangle(0, i, 250, i+h, fill="#ddd", width=0)
            self.canvas_speech.create_rectangle(0, i, canvas_width, i+h, fill="#ddd", width=0)
        self.canvas_speech_name.create_line(200, 0, 200, canvas_height, fill="black", width=6, tags="border")
        self.canvas_problem.create_line(0, 380, canvas_width, 380, fill="black", width=6, tags="border")
        self.canvas_problem.create_line(0, 20+125, canvas_width, 20+125, fill="black", width=1, tags="border")
        if problem is not None:
            lines = {}
            for i, name in enumerate(sorted(problem.using_speeches)):
                lines[name] = i
                self.canvas_speech_name.create_text(40, i*h+h//2, text=f"#{i+1:02}", font=("Consolas", 25))
                self.canvas_speech_name.create_text(105, i*h+h//2, text=name, font=("Consolas", 25))
                reject_button_color = "#888" if self.gui.solving else "#fff"
                reject_text_color = "#800" if self.gui.solving else "#f00"
                self.canvas_speech_name.create_rectangle(154, i*h+14, 186, i*h+46, fill=reject_button_color, width=3, tags="reject_button")
                self.canvas_speech_name.create_text(170, i*h+31, text=f"✕", fill=reject_text_color, font=("Consolas", 20), tags="reject_button")
            for i in range(problem.num_chunks):
                co = problem.chunk_offset(i, self.hop)
                if problem.chunks[i] is None:
                    nid = next((cid for cid in range(i+1, problem.num_chunks) if problem.chunks[cid] is not None), None)
                    if i == 0 or problem.chunks[i-1] is None or nid is None:
                        continue
                    predicted_length = None
                    for name, offset in zip(problem.chunk_using_speeches[i-1], problem.speech_offsets[i-1]):
                        if name not in problem.chunk_using_speeches[nid]:
                            continue
                        k = problem.chunk_using_speeches[nid].index(name)
                        print(problem.speech_offsets[nid][k], offset, len(problem.chunks[i-1]))
                        predicted_length = offset-problem.speech_offsets[nid][k]-len(problem.chunks[i-1])
                    if predicted_length is not None:
                        self.canvas_problem.create_text(co+8, 27, text=f"{predicted_length/48000:.2}", font=("Consolas", 10), anchor=NW)
                        self.canvas_problem.create_text(co+8, 40, text=f"{predicted_length}", font=("Consolas", 10), anchor=NW)
                    continue
                cw = len(problem.chunks[i])//self.hop+1
                spec_img = gen_image(melspectrogram_db(problem.chunks_removed[i], self.hop, n_mels=100), swap_rb=False)
                self.chunk_images[i] = ImageTk.PhotoImage(image=Image.fromarray(spec_img))
                self.canvas_problem.create_image(co, 380-100-3, image=self.chunk_images[i], anchor=NW)
                self.wf_images[i] = waveform_image(problem.chunks_removed[i], cw, 250)
                self.canvas_problem.create_image(co, 20, image=self.wf_images[i], anchor=NW)
                text = self.canvas_problem.create_text(co+8, 5, text=f"#{i+1}*", font=("Consolas", 16), anchor=NW, fill="red", tags=f"chunk_text_{i}")
                text_background(self.canvas_problem, text, "white")
                if i not in self.gui.selected_chunks:
                    self.canvas_problem.itemconfig(text, text=f"#{i+1}", fill="black")
                text = self.canvas_problem.create_text(co+8, 27, text=f"{len(problem.chunks[i])/48000:.2f}", font=("Consolas", 10), anchor=NW)
                text_background(self.canvas_problem, text, "white")
                text = self.canvas_problem.create_text(co+8, 40, text=f"{len(problem.chunks[i])}", font=("Consolas", 10), anchor=NW)
                text_background(self.canvas_problem, text, "white")
                for j in range(0, canvas_height):
                    self.canvas_speech.create_rectangle(co, j*h, co+cw, j*h+h, fill="#999" if j % 2 == 0 else "#bbb", width=0)
                if 0 < i and problem.chunks[i - 1] is None:
                    self.canvas_speech.create_line(co-1, 0, co-1, canvas_height, fill="black", width=1, tags="border")
                    self.canvas_problem.create_line(co-1, 0, co-1, 380, fill="black", width=1, tags="border")
                if i + 1 < problem.num_chunks or problem.width(self.hop) < 1150:
                    self.canvas_speech.create_line(co+cw-1, 0, co+cw-1, canvas_height, fill="black", width=1, tags="border")
                    self.canvas_problem.create_line(co+cw-1, 0, co+cw-1, 380, fill="black", width=1, tags="border")
                for name, so, st in zip(problem.chunk_using_speeches[i], problem.speech_offsets[i], problem.speech_trims[i]):
                    so = so//self.hop
                    tl = st[0]//self.hop
                    tr = st[1]//self.hop+1
                    image_gray = self.speech_images_gray[name]
                    image_gray = image_gray.crop((max(0, -so), 0, min(image_gray.width, cw-so), 128))
                    image_gray = ImageTk.PhotoImage(image=image_gray)
                    self.image_cache.append(image_gray)
                    self.canvas_speech.create_image(co+max(0, so), lines[name]*h, image=image_gray, anchor=NW)
                    image_magma = self.speech_images_magma[name]
                    image_magma = image_magma.crop((tl, 0, tr, 64))
                    image_magma = ImageTk.PhotoImage(image=image_magma)
                    self.image_cache.append(image_magma)
                    self.canvas_speech.create_image(co+so+tl, lines[name]*h, image=image_magma, anchor=NW)
        self.canvas_speech.tag_raise("border")
        self.canvas_problem.tag_raise("border")
