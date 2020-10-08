# Prepare files for Zenodo

```bash
# on Osiris
cd DMS/mesaruns/
mkdir zenodo_Raen2020
# copy inlist templates, and src dir to newdir/supporting_code
# copy model data from RUNS_FINAL to newdir/model_data

# on Osiris
tar -czvf supporting_code.tar.gz supporting_code
# on Roy
scp tjr63@osiris-inode01.phyast.pitt.edu:/home/tjr63/DMS/mesaruns/zenodo_Raen2020/supporting_code.tar.gz Downloads/zenodo_Raen2020/.

# on Osiris
# having trouble uploading the full tarball; exclude large files
tar -czvf model_data.tar.gz --exclude=model_data/1.0_Msun_data/log_GammaB_6/LOGS/history.data --exclude=model_data/1.0_Msun_data/log_GammaB_6/LOGS/STD.out --exclude=model_data/1.0_Msun_data/log_GammaB_4/LOGS/history.data --exclude=model_data/*/*/photos model_data
# on Roy
scp tjr63@osiris-inode01.phyast.pitt.edu:/home/tjr63/DMS/mesaruns/zenodo_Raen2020/model_data.tar.gz Downloads/zenodo_Raen2020/.

```
