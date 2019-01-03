#/usr/bin/python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import datetime as dt
import time

style.use('fivethirtyeight')
dashboard = plt.figure()
ax1=dashboard.add_subplot(1,1,1)
ax1.patch.set_facecolor('xkcd:black')
#fig, axes = plt.subplots(nrows=2, ncols=5)
#ax0, ax1, ax2, ax3 = axes.flatten()
csfont = {'fontname':'Helvetica'}

def animate(i):
  graph_data=open('predictions.dat', 'r').read()
  lines = graph_data.split('\n')
  #ys1, ys2, ys3, ys4, ys5, ys6, ys7, ys8, ys9, ys10 = []
  ys1 = []
  ys2 = []
  ys3 = []
  ys4 = []
  ys5 = []
  ys6 = []
  ys7 = []
  ys8 = []
  ys9 = []
  ys10 = []
  xs=[]
  xlimit=9
  
  for line in lines:
      if len(line) > 1 :
        #print(line)
        y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = line.split(',')
        xs.append(dt.datetime.now().strftime('%H:%M:%S:%f'))
        #date = time.strftime("%H:%M:%S")
        #xs.append(date)
        if len(xs) > xlimit : del xs[0] 
        ys1.append(float(y1))
        if len(ys1) > xlimit : del ys1[0] 
        ys2.append(float(y2))
        if len(ys2) > xlimit : del ys2[0] 
        ys3.append(float(y3))
        if len(ys3) > xlimit : del ys3[0] 
        ys4.append(float(y4))
        if len(ys4) > xlimit : del ys4[0] 
        ys5.append(float(y5))
        if len(ys5) > xlimit : del ys5[0] 
        ys6.append(float(y6))
        if len(ys6) > xlimit : del ys6[0] 
        ys7.append(float(y7))
        if len(ys7) > xlimit : del ys7[0] 
        ys8.append(float(y8))
        if len(ys8) > xlimit : del ys8[0] 
        ys9.append(float(y9))
        if len(ys9) > xlimit : del ys9[0] 
        ys10.append(float(y10))
        if len(ys10) > xlimit : del ys10[0] 
  ax1.clear()
  ax1.plot(xs,ys1, label="Analysis", color="white")
  ax1.plot(xs,ys2, label="Backdoor", color="blue")
  ax1.plot(xs,ys3, label="DoS", color="red")
  ax1.plot(xs,ys4, label="Exploits", color="darkviolet")
  ax1.plot(xs,ys5, label="Fuzzers", color="yellow")
  ax1.plot(xs,ys6, label="Generic", color="violet")
  ax1.plot(xs,ys7, label="Normal", color="lime")
  ax1.plot(xs,ys8, label="Reconnaissance",color="aqua")
  ax1.plot(xs,ys9, label="Shellcode", color="pink")
  ax1.plot(xs,ys10, label="Worms", color="peru")
  ax1.set_ylim(0, 2)
  plt.title('Realtime attack probability using 2D CNN', style='italic')
  plt.xlabel('Time', style='italic')
  plt.legend(loc="upper left")
  plt.ylabel('Probability')
  ax1.patch.set_facecolor('xkcd:black')


ani=animation.FuncAnimation(dashboard, animate, interval=10)
plt.gca().axes.get_yaxis().set_visible(False)
plt.show()
