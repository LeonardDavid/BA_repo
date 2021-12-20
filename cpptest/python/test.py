import numpy as np

nr = np.arange(0,676)

for y in nr:
    print(y," ",int(y/26)," ",int(y/26)+2," # ",y%26," ",y%26+2)
    if(y>0 and (y+1)%26==0):
        print("\n")
