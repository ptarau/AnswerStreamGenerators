export PFILE=ansgens

# make latex
pushd .
cd ../src
rm -f $PFILE.pdf
pdflatex $PFILE.tex
pdflatex $PFILE.tex
bibtex $PFILE
pdflatex $PFILE.tex
ls -l  $PFILE.pdf
popd

# extract Prolog
rm -f  ../src/$PFILE.pro
gcc -O -o toc ../src/toc.c
./toc < ../src/$PFILE.tex > ../src/$PFILE.pro
pushd .
cd ../src
ls -l $PFILE.pro
swipl -O -s $PFILE.pro
popd

