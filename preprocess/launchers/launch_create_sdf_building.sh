source ./isosurface/LIB_PATH
dset='building'
reduce=4
python -u create_sdf.py --dset ${dset} --thread_num 9 --reduce ${reduce}