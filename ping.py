import pythonping
import time
import numpy as np
import csv

hours = 12
delay = 5
numPings = round(int((12 * 60 * 60) / delay))
pings = {}
numSucess, numFail = 0, 0
for i in range(numPings):
    try:
        ping = pythonping.ping("www.google.com").rtt_avg * 1000
        numSucess += 1
        print(f'Ping at {time.ctime()} = {round(ping, 2)} milliseconds, numSuccess: {numSucess}, numFail: {numFail}')
    except:
        ping = np.nan
        numFail += 1
        print(f'Ping at {time.ctime()} = NO CONNECTION!!!!!!!!')
    pings[time.ctime()] = ping

    time.sleep(delay)
    try:
        w = csv.writer(open("pings.csv", "w"))
        for key, val in pings.items():
            w.writerow([key, val])
    except:
        print('ah, failed writing to file...')
