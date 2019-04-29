swipl -s lazy_streams.pl -g "open('../../src/doc.tex',write,S),latex_for_file('lazy_streams.pl',S,[]),close(S),halt"

