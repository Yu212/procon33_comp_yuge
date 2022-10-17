import os

from src import connection
from src.gui import Gui
from src.utils import load_speeches

if __name__ == "__main__":
    for name in os.listdir("../temp"):
        os.remove(f"../temp/{name}")
    connection.load_token()
    connection.test()
    load_speeches()
    gui = Gui()
