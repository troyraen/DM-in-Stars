
```python
# use dataProc.md to create files/data derived from MESA output
# (descDF and isochrones)
import plot_fncs as pf
```


# MStau

```python
descdf = pf.get_descdf(fin=pf.fdesc) # get the descdf
pf.plot_delta_tau(descdf, cctrans_frac=0.01, which='avg')
```
