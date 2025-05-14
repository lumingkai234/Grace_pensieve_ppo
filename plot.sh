LD_LIBRARY_PATH=libs/:libs/backup/:${LD_LIBRARY_PATH} python3 grace_new.py
cd results
python plot.py --lamda 16384
cd ..