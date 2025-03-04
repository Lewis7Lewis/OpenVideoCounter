"""Add args to the simple coutner"""
import logging
import argparse
import pathlib

from threadcounter import ThreadCounter
from analyzer import Analyser

# Arg Parsing
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument_group("Mandatory")
parser.add_argument("Videofilename", type=pathlib.Path, help="Videofile")

parser.add_argument("Output_file", type=pathlib.Path, help="csv fileoutput")

args = parser.parse_args()


logger.info(f"Video file : {args.Videofilename.absolute()}")
logger.info(f"Log File : {args.Output_file.absolute()}")

VIDEO_FILE = str(args.Videofilename.absolute())
CSVFILENAME = str(args.Output_file.absolute())


analyse = Analyser("config.bak.toml")
analyse.open()
Detectorator = ThreadCounter(
    analyse,
    VIDEO_FILE,
    CSVFILENAME,
    size=30,
    net="Models/yolov8n.onnx",
    show=True,
)
count, duration = Detectorator.run()
factor = Detectorator.factorspeed()

logger.info(f"Le système à denombrer {count} Personnes entrante")
logger.info(f"Le job a pris {duration},(x{factor:.2f})")

Detectorator.show_graph()
