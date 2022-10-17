import re
import threading
from decimal import Decimal
from tkinter import *

from src import connection, backup
from src.canvas import GuiCanvas
from src.match import Match
from src.predictor import predict_all, predict_next
from src.solver import Solver
from src.utils import *


class Gui:
    def __init__(self):
        self.root = Tk()
        self.left_frame = Frame(self.root)
        self.center_frame = Frame(self.root)
        self.right_frame = Frame(self.root)
        self.problem_info_border = Frame(self.left_frame, width=200, height=380, bg="black")
        self.problem_info = Frame(self.problem_info_border, width=200-3, height=380-3)
        self.canvas_speech_name = Canvas(self.left_frame, width=200, height=600, highlightthickness=0, bg="white")
        self.canvas_problem = Canvas(self.center_frame, width=1150, height=380, highlightthickness=0, bg="white")
        self.canvas_speech = Canvas(self.center_frame, width=1150, height=600, highlightthickness=0, bg="white")
        self.scroll_h_frame = Frame(self.center_frame, height=30)
        self.scroll_h = Scrollbar(self.scroll_h_frame, width=30, orient=HORIZONTAL)
        self.chunk_info = Frame(self.right_frame, height=180)
        self.buttons_frame_1 = Frame(self.right_frame, height=75)
        self.buttons_frame_2 = Frame(self.right_frame, height=75)
        self.button_solve1 = Button(self.buttons_frame_1, text="SOLVE1", font=("Consolas", 20))
        self.button_solve2 = Button(self.buttons_frame_1, text="SOLVE2", font=("Consolas", 20))
        self.button_adjust = Button(self.buttons_frame_1, text="ADJUST", font=("Consolas", 20))
        self.button_sa = Button(self.buttons_frame_1, text="SA", font=("Consolas", 20))
        self.button_get = Button(self.buttons_frame_2, text="GET", font=("Consolas", 20))
        self.button_add = Button(self.buttons_frame_2, text="ADD", font=("Consolas", 20))
        self.button_send = Button(self.buttons_frame_2, text="SEND", font=("Consolas", 20))
        self.button_match = Button(self.buttons_frame_2, text="MATCH", font=("Consolas", 20))
        self.scroll_v_frame = Frame(self.right_frame, width=30, height=630)
        self.scroll_v = Scrollbar(self.scroll_v_frame, width=30, orient=VERTICAL)
        self.speech_list_frame = Frame(self.right_frame, height=160)
        self.control_frame = Frame(self.right_frame, height=100)
        self.fit_buttons_frame = Frame(self.control_frame, bg="black")
        self.fit_button_left = Label(self.fit_buttons_frame, width=4, text="L", bg="white", font=("Consolas", 12))
        self.fit_button_right = Label(self.fit_buttons_frame, width=4, text="R", bg="white", font=("Consolas", 12))
        self.fit_button_or = Label(self.fit_buttons_frame, width=5, text="OR", bg="white", font=("Consolas", 12))
        self.fit_button_none = Label(self.fit_buttons_frame, width=5, text="NONE", bg="#aaa", font=("Consolas", 12))
        self.fit_button_and = Label(self.fit_buttons_frame, width=5, text="AND", bg="white", font=("Consolas", 12))
        self.parameters_canvas = Canvas(self.control_frame, highlightthickness=0)
        self.log_frame = Frame(self.right_frame)
        self.command_entry = Entry(self.log_frame, font=("Consolas", 12))
        self.log_box = Text(self.log_frame, width=70)
        self.log_scroll = Scrollbar(self.log_frame, orient=VERTICAL, command=self.log_box.yview)
        self.problem_number_label = Label(self.problem_info, font=("Consolas", 32))
        self.problem_id_label = Label(self.problem_info, font=("Consolas", 17))
        self.timer_label = Label(self.problem_info, font=("Consolas", 20))
        self.points_label = Label(self.problem_info, font=("Consolas", 17))
        self.num_using_speeches_label = Label(self.problem_info, font=("Consolas", 25))
        self.num_chunks_label = Label(self.chunk_info, font=("Consolas", 30))
        self.bonus_factor_label = Label(self.chunk_info, font=("Consolas", 20))
        self.got_chunks_label = Label(self.chunk_info, font=("Consolas", 20))

        self.root.state("zoomed")
        self.root.title("procon33")
        self.left_frame.pack(side=LEFT, fill=Y)
        self.center_frame.pack(side=LEFT, fill=Y)
        self.right_frame.pack(fill=BOTH, expand=True)
        self.problem_info_border.pack()
        self.problem_info.pack(padx=(0, 3), pady=(0, 3))
        self.problem_info.pack_propagate(False)
        self.canvas_speech_name.pack()
        self.canvas_problem.pack()
        self.canvas_speech.pack()
        self.scroll_h_frame.pack(fill=X, anchor=NW)
        self.scroll_h.place(x=0, y=0, relwidth=1)
        self.chunk_info.pack(fill=X)
        self.chunk_info.pack_propagate(False)
        self.buttons_frame_1.pack(fill=X, padx=25, pady=5)
        self.buttons_frame_2.pack(fill=X, padx=25, pady=5)
        self.button_solve1.place(relx=0, rely=0, anchor=NW, relheight=1, relwidth=0.22)
        self.button_solve2.place(relx=0.48, rely=0, anchor=NE, relheight=1, relwidth=0.22)
        self.button_adjust.place(relx=0.52, rely=0, anchor=NW, relheight=1, relwidth=0.22)
        self.button_sa.place(relx=1, rely=0, anchor=NE, relheight=1, relwidth=0.22)
        self.button_match.place(relx=0, rely=0, anchor=NW, relheight=1, relwidth=0.22)
        self.button_get.place(relx=0.48, rely=0, anchor=NE, relheight=1, relwidth=0.22)
        self.button_add.place(relx=0.52, rely=0, anchor=NW, relheight=1, relwidth=0.22)
        self.button_send.place(relx=1, rely=0, anchor=NE, relheight=1, relwidth=0.22)
        self.scroll_v_frame.pack(side=LEFT, anchor=NW)
        self.scroll_v.place(x=0, y=30, height=600)
        self.speech_list_frame.pack(anchor=W)
        self.control_frame.pack(fill=X)
        self.control_frame.pack_propagate(False)
        self.fit_buttons_frame.pack(side=LEFT, padx=(50, 0))
        self.fit_button_left.pack(side=LEFT, fill=Y, padx=(2, 1), pady=2)
        self.fit_button_right.pack(side=RIGHT, fill=Y, padx=(1, 2), pady=2)
        self.fit_button_or.pack(side=TOP, fill=X, pady=(2, 0))
        self.fit_button_none.pack(side=TOP, fill=X, pady=1)
        self.fit_button_and.pack(side=TOP, fill=X, pady=(0, 2))
        self.parameters_canvas.pack(side=LEFT)
        self.log_frame.pack(fill=Y, expand=True, padx=10, pady=(10, 20))
        self.command_entry.pack(side=BOTTOM, fill=X)
        self.log_box.pack(side=LEFT, fill=Y)
        self.log_scroll.pack(fill=Y, expand=True)
        self.problem_number_label.pack(pady=(50, 5))
        self.problem_id_label.pack(pady=5)
        self.timer_label.pack(pady=5)
        self.points_label.pack(pady=5)
        self.num_using_speeches_label.pack(pady=5)
        self.num_chunks_label.pack(pady=(20, 0))
        self.bonus_factor_label.pack(pady=5)
        self.got_chunks_label.pack()

        self.en_label = Label(self.speech_list_frame, text="E", font=("Consolas", 30))
        self.jp_label = Label(self.speech_list_frame, text="J", font=("Consolas", 30))
        self.en_label.grid(row=0, rowspan=4, column=0, padx=(0, 5))
        self.jp_label.grid(row=0, rowspan=4, column=12, padx=(10, 5))
        self.speech_name_labels = {}
        for name in speech_names():
            label = Label(self.speech_list_frame, highlightthickness=2, borderwidth=0, text=name[1:], font=("Consolas", 10))
            sid = int(name[1:])-1
            x = sid % 11 + (1 if name[0] == "E" else 13)
            y = sid // 11
            label.grid(row=y, column=x)
            self.speech_name_labels[name] = label
        self.canvas = GuiCanvas(self)
        self.solving = False
        self.exclude_used_speeches = True
        self.selected_chunks = []
        self.match = None
        self.open_hp = False
        self.sera = 3
        self.use_parallel = True
        self.use_debug = True
        self.use_local = False
        self.solver = Solver()

        self.log_box.config(state=DISABLED)
        self.scroll_h.config(command=lambda *args: (self.canvas_speech.xview(*args), self.canvas_problem.xview(*args)))
        self.scroll_v.config(command=lambda *args: (self.canvas_speech.yview(*args), self.canvas_speech_name.yview(*args)))
        self.canvas_speech.config(xscrollcommand=self.scroll_h.set)
        self.canvas_problem.config(xscrollcommand=self.scroll_h.set)
        self.canvas_speech.config(yscrollcommand=self.scroll_v.set)
        self.canvas_speech_name.config(yscrollcommand=self.scroll_v.set)
        self.canvas_speech.bind("<Shift-MouseWheel>", self.mouse_wheel_h)
        self.canvas_problem.bind("<Shift-MouseWheel>", self.mouse_wheel_h)
        self.canvas_speech_name.bind("<Shift-MouseWheel>", lambda: ())
        self.canvas_speech.bind("<MouseWheel>", self.mouse_wheel_v)
        self.canvas_speech_name.bind("<MouseWheel>", self.mouse_wheel_v)
        self.log_box.config(yscrollcommand=self.log_scroll.set)
        self.button_solve1.config(command=self.pressed_button_solve1)
        self.button_solve2.config(command=self.pressed_button_solve2)
        self.button_adjust.config(command=self.pressed_button_adjust)
        self.button_sa.config(command=self.pressed_button_sa)
        self.button_match.config(command=self.pressed_button_match)
        self.button_add.config(command=self.pressed_button_add)
        self.button_get.config(command=self.pressed_button_get)
        self.button_send.config(command=self.pressed_button_send)
        self.fit_button_left.bind("<Button-1>", lambda event: self.pressed_button_fit("left", self.fit_button_left))
        self.fit_button_right.bind("<Button-1>", lambda event: self.pressed_button_fit("right", self.fit_button_right))
        self.fit_button_or.bind("<Button-1>", lambda event: self.pressed_button_fit("or", self.fit_button_or))
        self.fit_button_none.bind("<Button-1>", lambda event: self.pressed_button_fit("none", self.fit_button_none))
        self.fit_button_and.bind("<Button-1>", lambda event: self.pressed_button_fit("and", self.fit_button_and))
        self.canvas_speech_name.tag_bind("reject_button", "<Button-1>", self.canvas.clicked_reject_button)
        self.canvas_problem.bind("<Button-1>", self.canvas.clicked_chunk)
        self.command_entry.bind("<Return>", self.command_execute)
        self.dummy_button = Button(self.root, command=self.canvas.draw)

        self.init_control_frame()
        self.set_button_state("00001000")
        self.update_timer()
        self.draw()
        self.update_chunk_info()
        self.root.mainloop()

    def init_control_frame(self):
        self.parameters_canvas.create_text(30, 8, tags="min_window", anchor=NW, font=("Consolas", 11))
        self.parameters_canvas.create_text(30, 28, tags="max_window", anchor=NW, font=("Consolas", 11))
        self.parameters_canvas.create_text(30, 48, tags="predicted", anchor=NW, font=("Consolas", 11))
        self.parameters_canvas.create_text(30, 68, tags="predicted_next", anchor=NW, font=("Consolas", 11))
        self.update_control_frame()

    def update_control_frame(self):
        self.parameters_canvas.itemconfig("min_window", text=f"min_window: {self.solver.min_window} {self.open_hp}")
        self.parameters_canvas.itemconfig("max_window", text=f"max_window: {self.solver.max_window} {self.sera}")
        if self.match is None or not self.match.during_problem or self.match.current_problem.num_got_chunks in [0, self.match.current_problem.num_chunks]:
            self.parameters_canvas.itemconfig("predicted", text="")
            self.parameters_canvas.itemconfig("predicted_next", text="")
        else:
            self.parameters_canvas.itemconfig("predicted", text=f"{predict_all(self.match)}")
            self.parameters_canvas.itemconfig("predicted_next", text=f"{predict_next(self.match, 1)}")

    def set_button_state(self, state):
        problem = None if self.match is None else self.match.current_problem
        solve1_enabled = state[0] == '1'
        solve2_enabled = state[1] == '1'
        adjust_enabled = state[2] == '1'
        sa_enabled = state[3] == '1'
        match_enabled = state[4] == '1'
        get_enabled = state[5] == '1'
        add_enabled = state[6] == '1' and problem.num_got_chunks < problem.num_chunks
        send_enabled = state[7] == '1' and len(problem.using_speeches) <= problem.num_total_speeches
        self.button_solve1.config(state=NORMAL if solve1_enabled else DISABLED)
        self.button_solve2.config(state=NORMAL if solve2_enabled else DISABLED)
        self.button_adjust.config(state=NORMAL if adjust_enabled else DISABLED)
        self.button_sa.config(state=NORMAL if sa_enabled else DISABLED)
        self.button_match.config(state=NORMAL if match_enabled else DISABLED)
        self.button_get.config(state=NORMAL if get_enabled else DISABLED)
        self.button_add.config(state=NORMAL if add_enabled else DISABLED)
        self.button_send.config(state=NORMAL if send_enabled else DISABLED)

    def draw(self):
        self.dummy_button.invoke()
        self.update_problem_info()
        self.update_control_frame()
        self.update_speech_list()

    def mouse_wheel_v(self, event):
        self.canvas_speech.yview_scroll(-event.delta // 120, UNITS)
        self.canvas_speech_name.yview_scroll(-event.delta // 120, UNITS)

    def mouse_wheel_h(self, event):
        self.canvas_speech.xview_scroll(-event.delta // 120, UNITS)
        self.canvas_problem.xview_scroll(-event.delta // 120, UNITS)

    def command_execute(self, _):
        command = self.command_entry.get()
        self.command_entry.delete(0, END)
        if match := re.match(r"minw (\d+)", command):
            self.solver.min_window = int(match.group(1)) * 1000
            self.log(f"min_window set to {self.solver.min_window}")
            self.update_control_frame()
        if match := re.match(r"maxw (\d+)", command):
            self.solver.max_window = int(match.group(1)) * 1000
            self.log(f"max_window set to {self.solver.max_window}")
            self.update_control_frame()
        if match := re.match(r"hop1 (\d+)", command):
            self.solver.hop1 = int(match.group(1))
            self.log(f"hop1 set to {self.solver.hop1}")
        if match := re.match(r"hop2 (\d+)", command):
            self.solver.hop2 = int(match.group(1))
            self.log(f"hop2 set to {self.solver.hop2}")
        if match := re.match(r"height1 (\d+)", command):
            self.solver.height1 = int(match.group(1))
            self.log(f"height1 set to {self.solver.height1}")
        if match := re.match(r"height2 (\d+)", command):
            self.solver.height2 = int(match.group(1))
            self.log(f"height2 set to {self.solver.height2}")
        if match := re.match(r"fail (\d+)", command):
            self.solver.allow_fail = int(match.group(1))
            self.log(f"allow_fail set to {self.solver.allow_fail}")
        if re.match(r"interrupt", command):
            self.set_button_state("00000000")
            self.solver.interrupt = True
            self.log("interrupting")
        if re.match(r"debug", command):
            threading.Thread(target=self.debug_start, daemon=True).start()
        if re.match(r"exclude used", command):
            self.exclude_used_speeches = True
            self.log("exclude used speeches")
        if re.match(r"include used", command):
            self.exclude_used_speeches = False
            self.log("include used speeches")
        if re.match(r"draw", command):
            self.draw()
            self.update_chunk_info()
        if re.match(r"next", command):
            if self.solving:
                self.solver.interrupt = True
            self.selected_chunks.clear()
            self.match.end_problem([])
            self.draw()
            self.update_chunk_info()
            self.set_button_state("00000100")
        if re.match(r"send", command):
            self.pressed_button_send()
        if re.match(r"restore", command):
            self.restore_state()
        if re.match(r"pick", command):
            pass
        if re.match(r"use ([EJ]\d{2})", command):
            name = match.group(1)
            pass
        if re.match(r"parallel true", command):
            self.use_parallel = True
            self.log("parallel enabled")
        if re.match(r"parallel false", command):
            self.use_parallel = False
            self.log("parallel disabled")
        if re.match(r"backup true", command):
            backup.no_backup = False
            self.log("backup enabled")
        if re.match(r"backup false", command):
            backup.no_backup = True
            self.log("backup disabled")
        if re.match(r"hp open", command):
            self.open_hp = True
            self.log("hp opened")
            self.update_control_frame()
        if re.match(r"hp close", command):
            self.open_hp = False
            self.log("hp closed")
            self.update_control_frame()
        if match := re.match(r"sera (\d+)", command):
            self.sera = int(match.group(1))
            self.log(f"sera: {self.sera}")
            self.update_control_frame()
        if match := re.match(r"skip (\d+)", command):
            self.match.problem_number += int(match.group(1))
            self.draw()

    def restore_state(self):
        state = backup.restore()
        if self.match is None:
            backup.no_backup = True
            if state["during_problem"]:
                self.log(f"skip {state['number']-1} and get {state['num_got_chunks']} chunks")
            else:
                self.log(f"skip {state['number']}")
            return
        problem = self.match.current_problem
        self.match.used_speeches = state["used"]
        for cid in range(problem.num_chunks):
            if str(cid) not in state:
                continue
            for name, value in state[str(cid)].items():
                offset = value["offset"]
                trim = (value["tl"], value["tr"])
                problem.use_speech(cid, name, offset, trim)
        backup.no_backup = False
        self.draw()

    def log(self, message):
        scroll_end = self.log_box.yview()[1] == 1
        self.log_box.configure(state=NORMAL)
        self.log_box.insert(END, f"{message}\n")
        if scroll_end:
            self.log_box.see(END)
        self.log_box.configure(state=DISABLED)

    def solve1_start(self):
        self.solving = True
        self.button_solve1.config(text="STOP")
        self.set_button_state("10000001")
        self.draw()
        problem = self.match.current_problem
        groups = problem.make_groups(self.selected_chunks)
        self.draw()
        self.log(f"solve start: {sorted(self.selected_chunks)}")
        for group in groups:
            if self.use_debug:
                ofs = None
                # ofs = {k: v+problem.chunk_diff(0, group.f) for k, v in self.debug_offsets.items()}
                if self.use_parallel:
                    self.solver.solve_parallel(group, self.exclude_used_speeches, False, self.open_hp, self.sera, self.draw, self.log, ofs)
                else:
                    self.solver.solve(group, self.exclude_used_speeches, False, self.open_hp, self.sera, self.draw, self.log, ofs)
            else:
                if self.use_parallel:
                    self.solver.solve_parallel(group, self.exclude_used_speeches, False, self.open_hp, self.sera, self.draw, self.log)
                else:
                    self.solver.solve(group, self.exclude_used_speeches, False, self.open_hp, self.sera, self.draw, self.log)
            if self.solver.interrupt:
                self.log("solve interrupted")
                break
        else:
            self.log("solve finished")
        self.solving = False
        self.solver.interrupt = False
        self.button_solve1.config(text="SOLVE1")
        if self.match.during_problem:
            self.set_button_state("11110011")
        self.draw()

    def solve2_start(self):
        self.solving = True
        self.button_solve2.config(text="STOP")
        self.set_button_state("01000001")
        self.draw()
        problem = self.match.current_problem
        groups = problem.make_groups(self.selected_chunks)
        self.draw()
        self.log(f"solve start: {sorted(self.selected_chunks)}")
        for group in groups:
            if self.use_debug:
                ofs = None
                # ofs = {k: v+problem.chunk_diff(0, group.f) for k, v in self.debug_offsets.items()}
                if self.use_parallel:
                    self.solver.solve_parallel(group, self.exclude_used_speeches, True, self.open_hp, self.sera, self.draw, self.log, ofs)
                else:
                    self.solver.solve(group, self.exclude_used_speeches, True, self.open_hp, self.sera, self.draw, self.log, ofs)
            else:
                if self.use_parallel:
                    self.solver.solve_parallel(group, self.exclude_used_speeches, True, self.open_hp, self.sera, self.draw, self.log)
                else:
                    self.solver.solve(group, self.exclude_used_speeches, True, self.open_hp, self.sera, self.draw, self.log)
            if self.solver.interrupt:
                self.log("solve interrupted")
                break
        else:
            self.log("solve finished")
        self.solving = False
        self.solver.interrupt = False
        self.button_solve2.config(text="SOLVE2")
        if self.match.during_problem:
            self.set_button_state("11110011")
        self.draw()
        backup.write(self.match)

    def adjust_start(self):
        self.solving = True
        self.button_adjust.config(text="STOP")
        self.set_button_state("00100001")
        self.draw()
        problem = self.match.current_problem
        groups = problem.make_groups(self.selected_chunks)
        self.draw()
        self.log(f"adjust start: {sorted(self.selected_chunks)}")
        for group in groups:
            for name in group.using_speeches():
                self.solver.adjust(group, name, self.open_hp, self.sera, self.draw, self.log)
                if self.solver.interrupt:
                    self.log("adjust interrupted")
                    break
            if self.solver.interrupt:
                break
        else:
            self.log("adjust finished")
        self.solving = False
        self.solver.interrupt = False
        self.button_adjust.config(text="ADJUST")
        if self.match.during_problem:
            self.set_button_state("11110011")
        self.draw()
        backup.write(self.match)

    def debug_start(self):
        self.solving = True
        self.set_button_state("00000001")
        self.draw()
        problem = self.match.current_problem
        groups = problem.make_groups(self.selected_chunks)
        self.draw()
        self.log(f"debug start: {sorted(self.selected_chunks)}")
        for group in groups:
            self.solver.solve_debug(group, self.draw, self.log)
            if self.solver.interrupt:
                self.log("debug interrupted")
                break
        else:
            self.log("debug finished")
        self.solving = False
        self.solver.interrupt = False
        if self.match.during_problem:
            self.set_button_state("11110011")
        self.draw()
        backup.write(self.match)

    def sa_start(self):
        self.solving = True
        self.button_sa.config(text="STOP")
        self.set_button_state("00010001")
        self.draw()
        problem = self.match.current_problem
        groups = problem.make_groups(self.selected_chunks)
        self.draw()
        self.log(f"sa start: {sorted(self.selected_chunks)}")
        for group in groups:
            self.solver.solve_sa(group, self.exclude_used_speeches, self.draw, self.log)
            if self.solver.interrupt:
                self.log("sa interrupted")
                break
        else:
            self.log("sa finished")
        self.solving = False
        self.solver.interrupt = False
        self.button_sa.config(text="SA")
        if self.match.during_problem:
            self.set_button_state("11110011")
        self.draw()
        backup.write(self.match)

    def pressed_button_fit(self, fit, pressed):
        self.fit_button_left.config(bg="white")
        self.fit_button_right.config(bg="white")
        self.fit_button_or.config(bg="white")
        self.fit_button_none.config(bg="white")
        self.fit_button_and.config(bg="white")
        pressed.config(bg="#aaa")
        self.solver.set_fit(fit)

    def pressed_button_solve1(self):
        self.log("solve1 pressed")
        if self.solving:
            self.set_button_state("00000001")
            self.solver.interrupt = True
        else:
            threading.Thread(target=self.solve1_start, daemon=True).start()

    def pressed_button_solve2(self):
        self.log("solve2 pressed")
        if self.solving:
            self.set_button_state("00000001")
            self.solver.interrupt = True
        else:
            threading.Thread(target=self.solve2_start, daemon=True).start()

    def pressed_button_adjust(self):
        self.log("adjust pressed")
        if self.solving:
            self.set_button_state("00000001")
            self.solver.interrupt = True
        else:
            threading.Thread(target=self.adjust_start, daemon=True).start()

    def pressed_button_sa(self):
        self.log("sa pressed")
        if self.solving:
            self.set_button_state("00000001")
            self.solver.interrupt = True
        else:
            threading.Thread(target=self.sa_start, daemon=True).start()

    def pressed_button_match(self):
        self.log("match pressed")
        try:
            if self.use_debug:
                self.match = Match(10, [4, 4, 3.3, 2.2, 1.1, 1], 30, 40, 100)
                self.debug_problems = generate_problems(self.match)
            elif self.use_local:
                self.match = connection.get_local_match_data()
            else:
                self.match = connection.get_match_data()
        except ValueError as e:
            self.log(e)
            return
        self.draw()
        self.update_chunk_info()
        self.set_button_state("00000100")
        self.log("match end")

    def pressed_button_add(self):
        self.log("add pressed")
        if self.match is None:
            self.log("match not started")
            return
        if not self.match.during_problem:
            self.log("problem not started")
            return
        problem = self.match.current_problem
        try:
            if self.use_debug:
                cid = self.debug_problems[self.match.problem_number-1][1][problem.num_got_chunks]
                data = self.debug_problems[self.match.problem_number-1][2][cid]

                # cid = [0, 2, 4, 3, 1][problem.num_got_chunks]
                # cid = problem.num_got_chunks
                # data = self.debug_problem[cid]
                # names = [""]
                # with wave.open(names[cid], "rb") as wf:
                #     buf = wf.readframes(wf.getnframes())
                #     data = np.frombuffer(buf, dtype="int16")
            elif self.use_local:
                cid, data = connection.get_local_chunk_data(problem, problem.num_got_chunks)
            else:
                cid, data = connection.get_chunk_data(problem, problem.num_got_chunks)
        except ValueError as e:
            self.log(e)
            return
        self.match.current_problem.add_chunk(cid, data)
        self.selected_chunks.append(cid)
        self.draw()
        self.update_chunk_info()
        self.set_button_state("11110011")
        self.log("add end")

    def pressed_button_get(self):
        self.log("get pressed")
        if self.match is None:
            self.log("match not started")
            return
        try:
            if self.use_debug:
                problem = self.debug_problems[self.match.problem_number][0]
                problem.starts_at = time.time()
                self.debug_offsets = self.debug_problems[self.match.problem_number][3]

                # # n = random.randint(200000, 1000000)
                # # m = random.randint(10, 20)
                # n = 48000*10
                # m = 20
                # c = 5
                # s, names, idx, region, trim = generate_problem(n, m, c, same_length=True)
                # self.log(f"{n}, {m}, {names}")
                # print(n, m, names)
                # print("|".join(names))
                # print([region[idx[name]][0] - trim[idx[name]][0] for name in names])
                # print([region[idx[name]][0] for name in names])
                # print([trim[idx[name]][0] for name in names])
                # print("|".join([name for name in names if region[idx[name]][0] < len(s[0]) - 24000]))
                # # cv2.imwrite(f"../temp/problem.png", gen_image(melspectrogram_db(s[0], hop=2048)))
                # for name in names:
                #     if region[idx[name]][0] == 0:
                #         # sp = np.zeros(len(s[0]))
                #         # rf = region[idx[name]][0]
                #         # rt = min(region[idx[name]][1],len(s[0]))
                #         # tf = trim[idx[name]][0]
                #         # tt = tf + rt - rf
                #         # sp[rf:rt] = speech_data(name)[tf:tt]
                #         # cv2.imwrite(f"../temp/{name}.png", gen_image(melspectrogram_db(sp, hop=2048)))
                #         print(name, region[idx[name]][0] - trim[idx[name]][0])
                # self.debug_problem = s
                # self.debug_offsets = {name: region[idx[name]][0] - trim[idx[name]][0] for name in names}
                # problem = Problem(self.match, "problem_id", c, time.time(), 1000, m)
            elif self.use_local:
                problem = connection.get_local_problem_data(self.match)
            else:
                problem = connection.get_problem_data(self.match)
        except ValueError as e:
            self.log(e)
            return
        self.match.start_problem(problem)
        self.solver.set_problem(problem)
        self.canvas.start_problem(problem)
        self.draw()
        self.update_chunk_info()
        self.update_timer(loop=False)
        self.set_button_state("00000010")
        self.log("get end")

    def pressed_button_send(self):
        self.log("send pressed")
        problem = self.match.current_problem
        for name in problem.using_speeches:
            if toggle_ej(name) in problem.using_speeches:
                self.log(f"{name[1:]} is used twice!")
                return
        answer = [name[1:] for name in problem.using_speeches]
        self.log(f"send: {answer}")
        if self.solving:
            self.solver.interrupt = True
        current_bonus_factor = Decimal(str(self.match.bonus_factor[problem.num_got_chunks]))
        num_using_speeches = Decimal(len(problem.using_speeches))
        point = current_bonus_factor * num_using_speeches
        try:
            if self.use_debug or self.use_local:
                elapsed = int(time.time() - problem.starts_at)
            else:
                accepted_at = connection.send_answer(problem.problem_id, answer)
                elapsed = int(accepted_at - problem.starts_at)
        except ValueError as e:
            self.log(e)
            return
        self.log(f"send accepted at {elapsed//60:02}:{elapsed%60:02}, {point} pts")
        self.selected_chunks.clear()
        self.match.end_problem(answer)
        self.draw()
        self.update_chunk_info()
        self.set_button_state("00000100")
        self.log("send end")

    def update_chunk_info(self):
        if self.match is None:
            self.num_chunks_label.config(text=f"")
            self.bonus_factor_label.config(text=f"")
            self.got_chunks_label.config(text=f"")
        elif not self.match.during_problem:
            self.num_chunks_label.config(text=f"")
            self.bonus_factor_label.config(text=f"{self.match.bonus_factor}")
            self.got_chunks_label.config(text=f"")
        else:
            problem = self.match.current_problem
            got_chunks = list([cid+1 for cid in range(problem.num_chunks) if problem.chunks[cid] is not None])
            self.num_chunks_label.config(text=f"{problem.num_got_chunks}/{problem.num_chunks}")
            self.bonus_factor_label.config(text=f"{self.match.bonus_factor}")
            self.got_chunks_label.config(text=f"{got_chunks}")

    def update_problem_info(self):
        if self.match is None:
            self.problem_number_label.config(text=f"")
            self.problem_id_label.config(text=f"")
            self.points_label.config(text=f"")
            self.num_using_speeches_label.config(text=f"")
        elif not self.match.during_problem:
            self.problem_number_label.config(text=f"#{self.match.problem_number}/{self.match.num_problems}")
            self.problem_id_label.config(text=f"")
            self.points_label.config(text=f"")
            self.num_using_speeches_label.config(text=f"")
        else:
            problem = self.match.current_problem
            current_bonus_factor = Decimal(str(self.match.bonus_factor[problem.num_got_chunks]))
            num_using_speeches = Decimal(len(problem.using_speeches))
            final_bonus_factor = Decimal(str(self.match.bonus_factor[problem.num_chunks]))
            num_total_speeches = Decimal(problem.num_total_speeches)
            point = current_bonus_factor * num_using_speeches
            max_point = final_bonus_factor * num_total_speeches
            self.problem_number_label.config(text=f"#{self.match.problem_number}/{self.match.num_problems}")
            self.problem_id_label.config(text=f"{problem.problem_id}")
            self.points_label.config(text=f"{point}/{max_point} pts")
            self.num_using_speeches_label.config(text=f"{num_using_speeches}/{problem.num_total_speeches}")

    def update_timer(self, loop=True):
        if self.match is None:
            timer_text = ""
        elif not self.match.during_problem:
            timer_text = ""
        else:
            problem = self.match.current_problem
            elapsed = int(time.time() - problem.starts_at)
            timer_text = f"{elapsed//60:02}:{elapsed%60:02}/{problem.time_limit//60:02}:{problem.time_limit%60:02}"
        self.timer_label.config(text=timer_text)
        if loop:
            self.root.after(1000, self.update_timer)

    def update_speech_list(self):
        used_speeches = []
        using_speeches = []
        if self.match is not None:
            used_speeches = self.match.used_speeches
            if self.match.during_problem:
                using_speeches = self.match.current_problem.using_speeches
        for name, label in self.speech_name_labels.items():
            bg_color = "#b0b0b0" if name[1:] in used_speeches else "#f0f0f0"
            if name in using_speeches:
                border_color = "#000000"
            elif toggle_ej(name) in using_speeches:
                border_color = "#d0d0d0"
            else:
                border_color = "#f0f0f0"
            label.config(bg=bg_color, highlightbackground=border_color)
