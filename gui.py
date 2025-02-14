from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import tkinter.ttk as ttk

import cv2

from threadcounter import ThreadCounter
from analyzer import Fence,Crops,Filters,Analyser

class LabelSpinbox(Frame):
    def __init__(self, master, text,minmaxinc=[0,100,1],*args,**kwargs):
        super().__init__(master,*args,**kwargs)
        self.Label =Label(self,text=text)
        self.Label.grid(row=1,column=1,sticky=NS+W)
        self.Spinbox = Spinbox(self,from_=minmaxinc[0],to=minmaxinc[1],increment=minmaxinc[2])
        self.Spinbox.grid(row=1,column=2,sticky=NSEW)
        self.grid_columnconfigure([2],weight=1)
        self.grid_rowconfigure([1],weight=1)


class Configuration(LabelFrame):
    def __init__(self, master,url:Entry,*args,**kwargs):
        self.url = url
        super().__init__(master,text="Configuration",*args,**kwargs)


        self.IOFrame = Frame(self)
        self.Bopen = Button(self.IOFrame,text="Open Config",command=self.openfile)
        self.Bsave = Button(self.IOFrame,text="Save config",command=self.savefile)
        self.Bopen.grid(row=0,column=0)
        self.Bsave.grid(row = 0,column=1)

        self.crop = LabelFrame(self,text="Crop")
        self.cropleft = LabelSpinbox(self.crop,"Left")
        self.cropleft.grid(row=1,column=1,sticky=W)
        self.cropright = LabelSpinbox(self.crop,text="Right")
        self.cropright.grid(row=2,column=1,sticky=W)
        self.croptop = LabelSpinbox(self.crop,text="Top")
        self.croptop.grid(row=3,column=1,sticky=W)
        self.cropbotton = LabelSpinbox(self.crop,text="Bottom")
        self.cropbotton.grid(row=4,column=1,sticky=W)
        self.crop.grid_columnconfigure([1],weight=1)
        self.crop.grid_rowconfigure([1,2,3,4],weight=1)

        self.filters = LabelFrame(self,text="Filters")
        self.maxsize = LabelSpinbox(self.filters,text="Maximun Size")
        self.maxsize.grid(row=1,sticky=W)
        self.trackingdistance = LabelSpinbox(self.filters,text="Tracking distance")
        self.trackingdistance.grid(row=2,sticky=W)
        self.maxTrackingDistance = LabelSpinbox(self.filters,text="Maximun tracking distance")
        self.maxTrackingDistance.grid(row=3,sticky=W)
        self.minWalkingDistance = LabelSpinbox(self.filters,text="Minimun walking distance")
        self.minWalkingDistance.grid(row=4,sticky=W)

        self.filters.grid_columnconfigure([0],weight=1)
        self.filters.grid_rowconfigure([1,2,3,4],weight=1)


        self.fence = LabelFrame(self,text="Fence")
        self.fencel = LabelSpinbox(self.fence,text="Left Fence")
        self.fencel.grid(row=1,sticky=W)
        self.fencer = LabelSpinbox(self.fence,text="Right Fence")
        self.fencer.grid(row=2,sticky=W)
        self.fenceangle = LabelSpinbox(self.fence,text="Angle")
        self.fenceangle.grid(row=3,sticky=W)

        self.fence.grid_columnconfigure([0],weight=1)
        self.fence.grid_rowconfigure([1,2,3],weight=1)


        self.test = Button(self,text="Show Config",command=self.show_config)
        self.test.grid(row=4,column=1,sticky=NSEW)

        
        self.IOFrame.grid(row=0,column=1,sticky=NSEW)
        self.crop.grid(row=1,column=1,sticky=NSEW)
        self.filters.grid(row=2,column=1,sticky=NSEW)
        self.fence.grid(row=3,column=1,sticky=NSEW)


        self.grid_columnconfigure([1],weight=1)
        self.grid_rowconfigure([0,1,2,3,4],weight=1)

    def show_config(self):
        a = self.get_datas()
        ok,error = a.check()
        if not ok:
            messagebox.showerror("Bad configuration",error)
            return
        if (url:=self.url.get()) == "":
            url = filedialog.askopenfilename(title="Select a VideoFile")

        if url != "" :
            
            if url.split(".")[-1] == "mp4":
                cam = cv2.VideoCapture(url)
                ok, img = cam.read()
                if ok :
                    cam.release()
                else :
                    return
            else:
                img = cv2.imread(url)
            cv2.imshow("Croping",a.draw_settings(a.crop_scale_inferance(img)))
            cv2.waitKey(0)

            

        

    def openfile(self):
        filename = filedialog.askopenfilename(
            title="Open Config",
            filetypes=(("TOML config", "*.toml"), ("all files", "*.*")),
        )
        if filename != "":
            a = Analyser(filename)
            a.open()
            self.load(a)

    def savefile(self):
        filename = filedialog.asksaveasfilename(
            title="Save Config",
            filetypes=(("TOML config", "*.toml"), ("all files", "*.*")),
        )
        if filename != "":
            a = self.get_datas()
            a.save()

    def filter_float(self,data=str):
        inte = ""
        for v in data:
            if v.isnumeric():
                inte += v

        return float(inte)

    def get_datas(self):
        #crop 
        crops = Crops()
        crops.left = self.filter_float(self.cropleft.Spinbox.get())
        crops.right = self.filter_float(self.cropright.Spinbox.get())
        crops.top = self.filter_float(self.croptop.Spinbox.get())
        crops.bottom = self.filter_float(self.cropbotton.Spinbox.get())

        #filters
        filters = Filters()
        filters.maxsize = self.filter_float(self.maxsize.Spinbox.get())
        filters.tracking_distance = self.filter_float(self.trackingdistance.Spinbox.get())
        filters.max_tracking_distance = self.filter_float(self.maxTrackingDistance.Spinbox.get())
        filters.min_walking_distance = self.filter_float(self.minWalkingDistance.Spinbox.get())

        #fence 
        fence= Fence()
        fence.l = self.filter_float(self.fencel.Spinbox.get())
        fence.r = self.filter_float(self.fencer.Spinbox.get())
        fence.angle = self.filter_float(self.fenceangle.Spinbox.get())

        analys =Analyser()
        analys.fence = fence
        analys.filters = filters
        analys.crops = crops

        return analys

    def load(self,analys:Analyser):
        crops = analys.crops
        self.cropleft.Spinbox.delete(0,END)
        self.cropleft.Spinbox.insert(0,str(crops.left))
        self.cropright.Spinbox.delete(0,END)
        self.cropright.Spinbox.insert(0,str(crops.right))
        self.croptop.Spinbox.delete(0,END)
        self.croptop.Spinbox.insert(0,str(crops.top))
        self.cropbotton.Spinbox.delete(0,END)
        self.cropbotton.Spinbox.insert(0,str(crops.bottom))

        filters = analys.filters
        self.maxsize.Spinbox.delete(0,END)
        self.maxsize.Spinbox.insert(0,str(filters.maxsize))
        self.trackingdistance.Spinbox.delete(0,END)
        self.trackingdistance.Spinbox.insert(0,str(filters.tracking_distance))
        self.maxTrackingDistance.Spinbox.delete(0,END)
        self.maxTrackingDistance.Spinbox.insert(0,str(filters.max_tracking_distance))
        self.minWalkingDistance.Spinbox.delete(0,END)
        self.minWalkingDistance.Spinbox.insert(0,str(filters.min_walking_distance))

        fence = analys.fence
        self.fencel.Spinbox.delete(0,END)
        self.fencel.Spinbox.insert(0,str(fence.l))
        self.fencer.Spinbox.delete(0,END)
        self.fencer.Spinbox.insert(0,str(fence.r))
        self.fenceangle.Spinbox.delete(0,END)
        self.fenceangle.Spinbox.insert(0,str(fence.angle))

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

        self.video = ttk.Checkbutton(self.params, text="Enable Video output")
        self.video.grid(row=3, column=1, columnspan=3, sticky=NSEW)

        self.configuration = Configuration(self,self.urlVideo)

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
        if self.TD is None or True:
            self.TD = ThreadCounter(
                self.configuration.get_datas(),self.urlVideo.get(),self.urlOutput.get(),
                size=30,
                net=self.urlNet.get(),
                show= 'selected' in self.video.state()
            )
        else :
            ### Can't restart thread
            self.TD.reset(self.configuration.get_datas(),self.urlVideo.get(),self.urlOutput.get(),
                size=30,
                net=self.urlNet.get(),
                show= 'selected' in self.video.state())
        self.startstop.config(bg="red", activebackground="#822", text="Stop",command=self.stop)
        self.showgraph.config(state="disabled")
        self.TD.start()
        self.TD.graph.add_rec()
        self.update()

    def update(self):
        self.pourcentVar.set(self.TD.pourcent())

        if self.TD.is_running() :
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