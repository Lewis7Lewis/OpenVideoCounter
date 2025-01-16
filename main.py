"""The main program to use the counter"""

from tkinter import filedialog
from tkinter import messagebox


from threadcounter import ThreadCounter

if __name__ == "__main__":
    Video_file = filedialog.askopenfilename(title="Select a VideoFile")
    csvfilename = filedialog.asksaveasfilename(
        title="Save CSV name", filetypes=(("CSV File", "*.csv"), ("all files", "*.*"))
    )
    affichage = messagebox.askyesno("Affichage", "Voulez-vous afficher la vidéo")
    Detectorator = ThreadCounter(
        "config.toml",
        Video_file,
        csvfilename,
        size=32,
        net="Models/yolov8n.onnx",
        show=False,
    )
    print("Starting")
    count, duration = Detectorator.run()
    factor = Detectorator.factorspeed()

    print(f"Le système à denombrer {count} Personnes entrante")
    print(f"Le job a pris {duration},(x{factor:.2f})")

    Detectorator.show_graph()

