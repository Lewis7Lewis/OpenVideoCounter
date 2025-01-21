"""loger Module"""

from math import inf
import queue
from threading import Thread
import csv


class Loger:
    """A threaded worker to Log and gat nices tables"""

    def __init__(self, filename, fifo: queue.Queue):
        self.file = open(filename, "w", newline="", encoding="utf-8")
        self.fifo = fifo
        self.csv = csv.writer(self.file)
        self.init_file()
        self.i = 0
        self.stopped = True
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads keep running in the background
        # while the program is executing
        self.i = 0

    def init_file(self):
        """Init the nice table"""
        self.csv.writerow(["Timing (secondes)", "people"])

    def rec(self, timing, people):
        """Add a record"""
        self.i += 1
        self.csv.writerow([timing, people])
        if self.i % 5 == 0:
            self.flush()

    def close(self):
        """Close the file"""
        self.file.close()

    def update(self):
        """The threaded main process"""
        while not self.stopped:
            t, p = self.fifo.get(True)
            if t == inf:
                self.stop()
            else:
                self.rec(t, p)

    def flush(self):
        """Flush datas to the disk"""
        self.file.flush()

    def start(self):
        """Start the worker"""
        self.stopped = False
        self.t.start()

    def stop(self):
        """Stop the worker"""
        print("[Loger Stop]")
        self.stopped = True

    def join(self):
        """Wait until finish"""
        self.t.join()
