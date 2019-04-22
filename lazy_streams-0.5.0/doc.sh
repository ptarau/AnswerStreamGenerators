pushd .
cd prolog
swipl -s lazy_streams.pl -g "doc_save(., [recursive(true)]),halt"
popd
open prolog/doc/index.html
