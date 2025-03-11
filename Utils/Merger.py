import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import messagebox
from tkinter import simpledialog
import time
import datetime
from matplotlib.dates import DateFormatter

def get_data(files:list[str],formattime = None,correctiontime = 0) -> tuple[np.ndarray,np.ndarray] :
    t,p = np.array([]),np.array([])
    for i,f in enumerate(files) :
        if i == 0 :
            p = np.append(p,[0],0)
            if formattime is not None:
                starttime = datetime.datetime.strptime(".".join(f.split("/")[-1].split(".")[:-1]),formattime) + datetime.timedelta(hours=correctiontime)
                starttime = starttime.timestamp()
                t = np.append(t,[0 + int(starttime)],0)
            else :
                t = np.append(t,[0])

        d = np.loadtxt(f,skiprows=1,delimiter=",",dtype=np.int64)
        p = np.append(p,(d[:,1]+int(p[-1])),0)
        if formattime is None :
            t = np.append(t,(d[:,0]+int(t[-1])),0)
        else :
            starttime = datetime.datetime.strptime(".".join(f.split("/")[-1].split(".")[:-1]),formattime) + datetime.timedelta(hours=correctiontime)
            starttime = starttime.timestamp()
            t = np.append(t,d[:,0] + int(starttime))
    return t,p

filespath = filedialog.askopenfilenames(title="Datas files",filetypes=(("CSV","*.csv"),))

is_formatime = messagebox.askyesno("Time","le nom des fichiers est il formaté par le temps")
if is_formatime :
    formattime = simpledialog.askstring("Format",f"Quel est le format du temps pour les fichiers \n{filespath[0].split('/')[-1]}\nExemple: %Y-%m-%d %H:%M:%S")
    correctiontime = simpledialog.askinteger("Corection d'heure","Combien d'heure de décalage y a t'il entre le temps du fichier et le temps réel")
else :
    formattime = None
    correctiontime = 0


def derive(t,p):
    return t[1:-1],(p[2:]-p[:-2])/(t[2:]-t[:-2])

def todatetime(t):
    return [datetime.datetime.fromtimestamp(i) for i in t]

t,p= get_data(filespath,formattime=formattime,correctiontime=correctiontime)
fig1 = plt.figure(1)
plt.plot(todatetime(t),p,label=f"Nombre de personnes {p.max()}")
plt.xlabel("Horaire")
plt.ylabel("Personnes")
plt.legend(loc="best")
plt.gcf().autofmt_xdate()
plt.gcf().axes[0].xaxis.set_major_formatter(DateFormatter("%H:%M"))


fig2 =plt.figure(2)
plt.plot(todatetime(t)[1:-1],derive(t,p)[1]*60,label="Derivé")
plt.ylabel("Personnes par Minutes")
plt.xlabel("Horaire")
plt.legend(loc="best")
plt.gcf().autofmt_xdate()
plt.gcf().axes[0].xaxis.set_major_formatter(DateFormatter("%H:%M"))

plt.show()

filesavepath = filedialog.asksaveasfilename(title="Save File for excels",filetypes=(("CSV","*.csv"),))

if messagebox.askyesno("Save","Voulez vous sauvegarder les graphiques ?"):
    fig1.savefig(filesavepath.replace(".csv","_Global.png"))
    fig2.savefig(filesavepath.replace(".csv","_Derivative.png"))

#bad timezone gestion
np.savetxt(filesavepath,np.array([[d[0]/86400+25569 + 1/24,d[1]] for d in zip(t,p)]),delimiter=";",header="Horaire (Format Excel);Personnes",comments="",fmt=("%1f","%i"))