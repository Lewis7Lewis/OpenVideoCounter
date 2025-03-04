"""The main program to use the counter"""
import logging
from tkinter import filedialog
from tkinter import messagebox


from threadcounter import ThreadCounter
from analyzer import Analyser

logger = logging.getLogger(__name__ if __name__ != "__main__" else "Main")

if __name__ == "__main__":
    Video_file = filedialog.askopenfilename(title="Select a VideoFile")
    csvfilename = filedialog.asksaveasfilename(
        title="Save CSV name", filetypes=(("CSV File", "*.csv"), ("all files", "*.*"))
    )
    affichage = messagebox.askyesno("Affichage", "Voulez-vous afficher la vidéo")
    analyse = Analyser("config.toml")
    analyse.open()

    Detectorator = ThreadCounter(
        analyse,
        Video_file,
        csvfilename,
        size=32,
        net="Models/yolov8n.onnx",
        show=False,
    )
    logger.info("Starting")
    count, duration = Detectorator.run()
    factor = Detectorator.factorspeed()

    logger.info(f"Le système à denombrer {count} Personnes entrante")
    logger.info(f"Le job a pris {duration},(x{factor:.2f})")

    Detectorator.show_graph()

