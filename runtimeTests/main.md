

# Test varying OMP_NUM_THREADS

- [x]  Set up `run_osirisRTT.sh` to do m1p0c6
- [x]  Set up `run_osirisRTT_caller.sh` to do 3 simultaneous runs
- [x]  Add stopping condition to `defDM` inlist: `xa_central_lower_limit_species(1) = 'h1', xa_central_lower_limit(1) = 0.6`

```bash
# test 12, 9, 6 for OMP_NUM_THREADS
cd bash_scripts
nohup nice ./run_osirisRTT_caller.sh "_default_plus_DM" 12
```


# Look at data

```python
%run fncs

rcdf, pidf = load_all_data(dr=dr, rk='_default_plus_DM')

plot_avg_runtimes(rcdf, save=None)

### SAND



```
