python ../ertprocess.py \
-fname electra/*.ele \
-ftype electra \
-rec_check \
-rec_quantities r \
-method fft \
-rec_max 8 \
-rec_couple \
-ctc_check \
-ctc_max 10 \
-err 0.5 \
-err_rec \
-w_err \
-plot \
-export_simpeg \
-export_pygimli \
-export_res2dinv \
-export_csv

# python ../ertprocess.py \
# -fname mpt/*.Data \
# -ftype labrecque \
# -rec_check \
# -rec_quantities r \
# -rec_max 5 \
# -rec_couple \
# -ctc_check \
# -ctc_max 10 \
# -err 1 \
# -err_rec \
# -w_err \
# -plot \
# -export_simpeg \
# -export_pygimli \
# -export_res2dinv \
# -export_csv

python ../ertprocess.py \
-fname syscal/*.csv \
-ftype syscal \
-rec_check \
-rec_max 5 \
-rec_quantities r \
-rec_couple \
-err_rec \
-err 2 \
-w_err \
-plot \
-export_simpeg \
-export_pygimli \
-export_res2dinv \
-export_csv

# -rec_couple \
# -rhoa_chec \
# -rec_unpaired \
