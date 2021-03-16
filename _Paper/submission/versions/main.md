

# `diff` two versions
- [Install `latexdiff`](https://www.overleaf.com/learn/latex/Articles/Using_Latexdiff_For_Marking_Changes_To_Tex_Documents).
- ~[Install `latexdiffcite`](https://latexdiffcite.readthedocs.io/en/latest/)~ Couldn't get latexdiffcite to compile the diff. Instead, separated long citations into multiple `\cite{}`s.

```bash
cd ~/Osiris/DMS/mesaruns_analysis/_Paper/submission/versions
# dmsenv
diffname='diff-arXivfeb8-arXivmar16'
draftdir='DM-in-Stars-arXiv-feb25'
revisiondir='DM-in-Stars-arXiv-mar16'

draft=${draftdir}'/main.tex'
revision=${revisiondir}'/main.tex'
mkdir ${diffname}

# do the diff
latexdiff ${draft} ${revision} > ${diffname}/diff.tex

# copy some files so we can compile a pdf
cp -r ${revisiondir}/plots ${diffname}/.
cp ${revisiondir}/macros.sty ${revisiondir}/references.bib ${diffname}/.

# now compile the diff in TexShop or Overleaf
```
