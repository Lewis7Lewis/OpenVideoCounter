"""Add args to the simple coutner"""

import argparse
import pathlib
import time

from counter_yolo import Detector, Loger, camera

# Arg Parsing

parser = argparse.ArgumentParser()
parser.add_argument_group("Mandatory")
parser.add_argument("Videofilename", type=pathlib.Path, help="Videofile")

parser.add_argument("Output_file", type=pathlib.Path, help="csv fileoutput")

args = parser.parse_args()


print(f"Video file : {args.Videofilename.absolute()}")
print(f"Log File : {args.Output_file.absolute()}")

cam = camera(str(args.Videofilename.absolute()))

log = Loger(args.Output_file)

Detectorator = Detector(feed=cam, loging=log)
prog_start = time.time()
Detectorator.do_batch(30, False)
cam.release()
log.rec(Detectorator.frame_count // Detectorator.framerate, Detectorator.counter)
log.close()
print(
    f"Compute time of {time.time()-prog_start:.2f}s ~  {(time.time()-prog_start)/60:.1f} minutes "
)
print(f"Le système denombrer {Detectorator.counter} Personnes entrante")
