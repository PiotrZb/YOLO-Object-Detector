import multiprocessing
multiprocessing.freeze_support()

import sys

from io import StringIO
memory_buffer = StringIO()
sys.stdout = memory_buffer


from app import App


def main():
    application = App()
    application.run()


if __name__ == "__main__":
    main()