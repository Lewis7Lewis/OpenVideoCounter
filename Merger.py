import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog


def get_data(files):
    datas = [np.loadtxt(f,skiprows=1,delimiter=",",dtype=np.int64) for f in files]
    t,p = np.array([0]),np.array([0])
    for d in datas :
        t = np.append(t,(d[:,0]+int(t[-1])),0)
        p = np.append(p,(d[:,1]+int(p[-1])),0)

    return t//60,p

filespath = filedialog.askopenfilenames(title="Datas files",filetypes=(("CSV","*.csv"),))


def derive(t,p):
    return t[1:-1],(p[2:]-p[:-2])/(t[2:]-t[:-2])

t,p= get_data(filespath)
plt.figure(1)
plt.plot(t,p,label=f"Nombre de personnes {p.max()}")
plt.xlabel("Temps en minutes")
regr = np.polyfit(t,p,1)
plt.plot(t,np.polyval(regr,t),label=f"P = {regr[0]:.2f}*t + {regr[1]:.2f}")
plt.ylabel("Personnes")
plt.legend(loc="best")


plt.figure(2)
plt.plot(*derive(t,p),label="Derivé")
plt.ylabel("Personnes par Minutes")
plt.xlabel("Temps en minutes")
plt.legend(loc="best")

plt.show()

filesavepath = filedialog.asksaveasfilename(title="Save File",filetypes=(("CSV","*.csv"),))

np.savetxt(filesavepath,np.array([d for d in zip(t,p)]),fmt="%i")
