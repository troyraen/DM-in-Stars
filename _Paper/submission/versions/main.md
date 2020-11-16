# Install `latexdiff`
see instructions [here](https://www.overleaf.com/learn/latex/Articles/Using_Latexdiff_For_Marking_Changes_To_Tex_Documents).

# `diff` two versions

```bash
# dmsenv
cd Osiris/DMS/mesaruns_analysis/_Paper/versions/
diffname='diff-sept10-sept25'
draft='DM-in-Stars-sept10/main.tex'
revisiondir='DM-in-Stars-sept25'
revision=${revisiondir}'/main.tex'
mkdir ${diffname}

# do the diff
latexdiff ${draft} ${revision} > ${diffname}/diff.tex

# copy some files so we can compile a pdf
cp -r ${revisiondir}/plots ${diffname}/.
cp ${revisiondir}/macros.sty ${revisiondir}/references.bib ${diffname}/.

# now compile the diff in TexShop
```
