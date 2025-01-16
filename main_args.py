"""Add args to the simple coutner"""

import argparse
import pathlib

from threadcounter import ThreadCounter

# Arg Parsing

parser = argparse.ArgumentParser()
parser.add_argument_group("Mandatory")
parser.add_argument("Videofilename", type=pathlib.Path, help="Videofile")

parser.add_argument("Output_file", type=pathlib.Path, help="csv fileoutput")

args = parser.parse_args()


print(f"Video file : {args.Videofilename.absolute()}")
print(f"Log File : {args.Output_file.absolute()}")

VIDEO_FILE = str(args.Videofilename.absolute())
CSVFILENAME = str(args.Output_file.absolute())

Detectorator = ThreadCounter(
    "config.bak.toml",
    VIDEO_FILE,
    CSVFILENAME,
    size=30,
    net="Models/yolov8n.onnx",
    show=False,
)
count, duration = Detectorator.run()
factor = Detectorator.factorspeed()

print(f"Le système à denombrer {count} Personnes entrante")
print(f"Le job a pris {duration},(x{factor:.2f})")

Detectorator.show_graph()
