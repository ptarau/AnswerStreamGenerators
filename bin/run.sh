export PFILE=ansgens
# make code extractor
gcc -O -o toc ../src/toc.c

# make latex
pushd .
cd ../src
rm -f $PFILE.pdf
pdflatex $PFILE.tex
pdflatex $PFILE.tex
bibtex $PFILE
pdflatex $PFILE.tex
ls -l  $PFILE.pdf

# extract Prolog
rm -f  $PFILE.pro
../bin/toc < $PFILE.tex > $PFILE.pro
ls -l $PFILE.pro
swipl -O -s $PFILE.pro
popd

