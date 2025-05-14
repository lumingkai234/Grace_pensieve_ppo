LD_LIBRARY_PATH=libs/:libs/backup/:${LD_LIBRARY_PATH} python3 grace-gpu.py
LD_LIBRARY_PATH=libs/:libs/backup/:${LD_LIBRARY_PATH} python3 h26x.py
LD_LIBRARY_PATH=libs/:libs/backup/${LD_LIBRARY_PATH} python3 pretrained-gpu.py
LD_LIBRARY_PATH=libs/:libs/backup/:${LD_LIBRARY_PATH} python3 grace_new.py
