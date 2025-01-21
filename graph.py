from queue import Queue
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from time import time

class Graph :
    def __init__(self,qimgs:Queue,qpreds:Queue,qpeople:Queue):
        self.qimgs= qimgs
        self.qpreds= qpreds
        self.qpeople= qpeople
        self.img = []
        self.pred = []
        self.people = []
        self.time = []
    
    def add_rec(self):
        self.img.append(self.qimgs.qsize())
        self.pred.append(self.qpreds.qsize())
        self.people.append(self.qpeople.qsize())
        self.time.append(time())

    def make_graph(self,fig:Figure):
        reltime = [i-self.time[0] for i in self.time]
        ax = fig.add_subplot(111)
        ax.plot(reltime,self.img,label="Images Queue Size")
        ax.plot(reltime,self.pred,label="Predictions Queue Size")
        ax.plot(reltime,self.people,label="People Queue Size")
        ax.legend(loc="best")
        ax.set_xlabel("Time(s)")
        ax.set_ylabel("Queue Size")
        ax.set_title("Queue Size over time")
        return fig




