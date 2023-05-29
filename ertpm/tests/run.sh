python ../ertprocess.py \
-fname electra/*.ele \
-ftype electra \
-rec_check \
-err_rec \
-err 3 \
-w_err \
-plot \
-export simpeg pygimli res2dinv csv

python ../ertprocess.py \
-fname mpt/*.Data \
-ftype labrecque \
-rec_check \
-err 3 \
-err_rec \
-w_err \
-plot \
-export simpeg pygimli res2dinv csv

python ../ertprocess.py \
-fname syscal/*.csv \
-ftype syscal \
-rec_check \
-err_rec \
-err 3 \
-w_err \
-plot \
-export simpeg pygimli res2dinv csv
