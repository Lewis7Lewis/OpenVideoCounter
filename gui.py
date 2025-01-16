from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import tkinter.ttk as ttk


from threadcounter import ThreadCounter


class Gui(Tk):
    def __init__(self):
        super().__init__()

        self.inputs = LabelFrame(self, text="IO")
        self.LabelVideo = Label(self.inputs, text="Video :")
        self.LabelVideo.grid(column=1, row=1, sticky=NSEW)
        self.urlVideo = Entry(self.inputs, width=50)
        self.urlVideo.grid(column=2, row=1, sticky=NSEW)
        self.clickVideo = Button(
            self.inputs, text="Open Video", command=self.select_video
        )
        self.clickVideo.grid(column=3, row=1, sticky=NSEW)

        self.LabelOutput = Label(self.inputs, text="CSV file")
        self.LabelOutput.grid(row=2, column=1, sticky=NSEW)
        self.urlOutput = Entry(self.inputs, width=50)
        self.urlOutput.grid(row=2, column=2, sticky=NSEW)
        self.clickOutput = Button(
            self.inputs, text="Save CSV file", command=self.select_output
        )
        self.clickOutput.grid(row=2, column=3, sticky=NSEW)

        self.params = LabelFrame(self, text="Paramètres")
        self.LabelNet = Label(self.params, text="Model Net")
        self.LabelNet.grid(row=1, column=1, sticky=NSEW)
        self.urlNet = Entry(self.params, width=50)
        self.urlNet.grid(row=1, column=2, sticky=NSEW)
        self.urlNet.insert(0, "Models/yolov8n.onnx")

        self.clickNet = Button(
            self.params, text="Open onnx model", command=self.select_net
        )
        self.clickNet.grid(row=1, column=3, sticky=NSEW)

        self.video = ttk.Checkbutton(self.params, text="Enable Video output",state=())
        self.video.grid(row=3, column=1, columnspan=3, sticky=NSEW)

        self.configuration = LabelFrame(self, text="Configuration")

        self.runs = LabelFrame(self, text="Run")

        self.startstop = Button(
            self.runs, bg="Green", activebackground="lightgreen", text="Start",command=self.start
        )
        self.startstop.grid(row=1, column=1, columnspan=2, rowspan=2, sticky=NSEW)
        self.showgraph = Button(self.runs,text="Show Graph",state="disabled",command=self.show_graph)
        self.showgraph.grid(row=1,column=3,rowspan=2, sticky=NSEW)
        self.pourcentVar = Variable(self.runs,value=0,name="progress")
        self.pourcent = ttk.Progressbar(self.runs, maximum=1,variable=self.pourcentVar)
        self.pourcent.grid(row=3, column=1, columnspan=3, sticky=NSEW)

        self.inputs.grid(row=1, column=1, sticky=NSEW)
        self.inputs.grid_rowconfigure([1,2],weight=1)
        self.inputs.grid_columnconfigure([1,3],weight=1)
        self.inputs.grid_columnconfigure(2,weight=2)

        self.params.grid(row=2, column=2, sticky=NSEW)
        self.params.grid_rowconfigure([1,2],weight=1)
        self.params.grid_columnconfigure([1,3],weight=1)
        self.params.grid_columnconfigure(2,weight=2)

        self.configuration.grid(row=2, column=1, sticky=NSEW)
        self.runs.grid(row=1, column=2, sticky=NSEW)
        self.runs.grid_rowconfigure([1,2,3],weight=1)
        self.runs.grid_columnconfigure([1,2,3],weight=1)

        self.grid_columnconfigure([1,2],weight=1)
        self.grid_rowconfigure([1,2],weight=1)

        self.TD = None

    def select_video(self):
        Video_file = filedialog.askopenfilename(title="Select a VideoFile")
        if Video_file != "" :
            self.urlVideo.delete(0, END)
            self.urlVideo.insert(0, Video_file)

    def select_output(self):
        csvfilename = filedialog.asksaveasfilename(
            title="Save CSV name",
            filetypes=(("CSV File", "*.csv"), ("all files", "*.*")),
        )
        if csvfilename != "":
            self.urlOutput.delete(0, END)
            self.urlOutput.insert(0, csvfilename)

    def select_net(self):
        netfilename = filedialog.askopenfilename(
            title="Open onnx model",
            filetypes=(("ONNX model", "*.onnx"), ("all files", "*.*")),
        )
        if netfilename != "":
            self.urlNet.delete(0, END)
            self.urlNet.insert(0, netfilename)

    def start(self):
        self.TD = ThreadCounter(
            "config.toml",self.urlVideo.get(),self.urlOutput.get(),
            size=30,
            net=self.urlNet.get(),
            show= 'selected' in self.video.state()
        )
        print(self.video.state())
        self.startstop.config(bg="red", activebackground="#822", text="Stop",command=self.stop)
        self.showgraph.config(state="disabled")
        self.TD.start()
        self.TD.graph.add_rec()
        self.update()

    def update(self):
        self.pourcentVar.set(self.TD.pourcent())

        if self.TD.runing :
            self.TD.graph.add_rec()
            self.after(10,self.update)
        else :
            self.start_config()

    def stop(self):
        self.TD.stop()


    def start_config(self):
        self.startstop.config(bg="Green", activebackground="lightgreen", text="Start",command=self.start)
        self.showgraph.config(state="normal")
        self.pourcentVar.set(0)

    
    def show_graph(self):
        self.TD.show_graph()



if __name__ == "__main__":
    app = Gui()
    app.mainloop()