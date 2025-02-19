"""Inference module"""
from math import inf
import queue
from threading import Thread
import time
import os

import cv2
import numpy as np
import onnxruntime as onx

from analyzer import Analyser
from fpsmeter import FPSMeter
from functions import beautifultime
from static import INFOTIME


trt_ep_options = {
    "trt_timing_cache_enable": True,
    "trt_max_workspace_size": 2147483648,
    "trt_fp16_enable": True,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": ".\trt_engines",
}
cuda_ep_option = {}


if "CUDA_PATH" in os.environ :
    provider = [
        ("CUDAExecutionProvider", cuda_ep_option),("CUDAExecutionProvider", cuda_ep_option),
    #('TensorrtExecutionProvider',trt_ep_options)
    ]
else :
    provider = [onx.get_available_providers()[0]]

class Inferance:
    """A threaded worker to perform detection inference"""

    def __init__(
        self,
        imgsfifo: queue.Queue,
        predfifo: queue.Queue,
        analys: Analyser,
        videoinfos=None,
        net="yolov8n.onnx",
        size=30,
        index = None,
        providers = None
    ) -> None:
        if index is not None :
            self.index = str(index)
        else :
            self.index = ""
        self.size = size
        self.analys = analys
        self.fps = FPSMeter()
        if videoinfos is None:
            self.videoinfos = [20, 120000]
        else:
            self.videoinfos = videoinfos
        self.imgsfifo = imgsfifo
        self.predfifo = predfifo
        self.predfifo.lastid = 0

        if providers is None :
            providers = provider

        self.providers = providers
        self.init_net(net,providers)

        # self.stopped is set to False when frames are being read from self.vcap stream
        self.stopped = True
        # reference to the thread for reading next available frame from input stream
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads keep running in the background
        # while the program is executing
        self.batch = []

    def init_net(self,net,providers):
        self.net = net
                ### Net 
        sess_options = onx.SessionOptions()
        sess_options.enable_profiling = False
        #sess_options.execution_mode = onx.ExecutionMode.ORT_PARALLEL
    
        self.net = onx.InferenceSession(
            net, providers=providers, sess_options=sess_options
        )  # ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.io_binds = self.net.io_binding()


        self.input_name = self.net.get_inputs()[0].name
        self.label_name = self.net.get_outputs()[0].name

    def reset(self,videoinfos=None,net=None,size=None):
        if net is not None and net != self.net :
            self.init_net(net,self.providers)
        if size is not None:
            self.size = size
        if videoinfos is None:
            self.videoinfos = [20, 120000]
        else:
            self.videoinfos = videoinfos
            
        self.predfifo.lastid = 0
        self.batch = []
         
    def predict(self, imgs):
        """Make the prediction from the model"""
        blob = cv2.dnn.blobFromImages(imgs, 1 / 255, (640, 640), (0, 0, 0), swapRB=True)
        self.io_binds.bind_cpu_input(self.input_name,blob.astype(np.float32))
        self.io_binds.bind_output(self.label_name)
        self.net.run_with_iobinding(self.io_binds)
        data = self.io_binds.copy_outputs_to_cpu()
        return data

    def start(self):
        """Start the worker"""
        self.stopped = False
        self.t.start()

    def update(self):
        """The thread main process"""
        while not self.stopped:
            try:
                i, img = self.imgsfifo.get(True, 1)

                if i == inf :
                    self.do_batch()
                    while self.predfifo.lastid < self.videoinfos[1]-1 :
                        time.sleep(0.001)
                    self.predfifo.put((i, [], []), True)
                    print("[End INFERANCE]")
                    self.stopped = True
                else:
                    # print(img.shape)
                    self.batch.append((i, img))
            except queue.Empty:
                pass
            else:
                if len(self.batch) == self.size:
                    self.do_batch()
                    self.batch = []
                if i != inf and i % (INFOTIME * self.videoinfos[0]) == 0:
                    print(
                        f"Inference {self.index}: {i} images ({beautifultime(i//self.videoinfos[0])}) (FPS:{self.fps.fps:.2f})"
                    )

    def do_batch(self):
        """Agregate images to make batch"""
        if len(self.batch) > 0:
            self.fps.update(1)
            predictions = self.predict([img for _, img in self.batch])
            self.fps.update(len(self.batch))
            for (i, img), prediction in zip(self.batch, predictions[0]):
                while self.predfifo.lastid < i-1 :
                    time.sleep(0.001)
                self.predfifo.put((i, img, prediction), True)
                self.predfifo.lastid = i

    def stop(self):
        """Stop the worker"""
        print(f"[Inferance {self.index} Stop]")
        self.stopped = True

    def join(self):
        """Wait until finish task"""
        self.t.join()
