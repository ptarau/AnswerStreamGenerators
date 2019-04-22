# AnswerStreamGenerators
Literate Prolog program and LaTeX sources for paper on Answer Stream Generators

To generate the paper and extract the Prolog code from it, go to directory 

bin 

and type

./run.sh

The code is contained a SWI Prolog package.

To install it, type:

```
pack_install('https://github.com/ptarau/AnswerStreamGenerators/raw/master/lazy_streams-0.5.0.zip').
```

followed by

```
?-use_module(library(lazy_streams)).

?-tests.
```

to see some tests and examples of use.
