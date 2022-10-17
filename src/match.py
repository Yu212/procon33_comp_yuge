from src import backup


class Match:
    def __init__(self, num_problems, bonus_factor, change_penalty, wrong_penalty, correct_point):
        self.num_problems = num_problems
        self.bonus_factor = bonus_factor
        self.change_penalty = change_penalty
        self.wrong_penalty = wrong_penalty
        self.correct_point = correct_point
        self.current_problem = None
        self.problem_number = 0
        self.during_problem = False
        self.used_speeches = []

    def start_problem(self, problem):
        self.during_problem = True
        self.problem_number += 1
        self.current_problem = problem
        backup.write(self)

    def end_problem(self, answer):
        self.used_speeches += answer
        self.current_problem = None
        self.during_problem = False
        backup.write(self)
