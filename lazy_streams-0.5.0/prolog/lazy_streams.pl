/**
<module> Lazy Stream Generators of finite or infinite sequences and their operations.

Lazy stream generators provide a unified interface to, stateful computations, I/O operations as well as algorithms producing finite or infinite sequences and answers of first class Prolog engines. We expose them to the application programmer either as lazy lists or through an abstract sequence manipulation API.

 We define an algebra of stream generator operations that extends Prolog via an embedded language interpreter providing a compact notation for composition mechanisms and supports moving between isomorphic sequence representations.

 Lazy Stream Generators operations have  isomorphic lazy list counterparts, the main difference being the use of a generic sequence manipulation API instead the use of a concrete list syntax in the case of lazy list. This isomorphism supports delegating operations between lazy lists and lazy streams.
 
As a special instance, we introduce answer stream generators that  encapsulate the work of coroutining logic engines with support for expendable or reusable answer streams.
*/

:- module(lazy_streams,[
  ask/2, % ask for next element of a lazy stream generator
  ask_/2, % like ask, but working in a stream expression
  stop/1, % stop/1 marks a generator as done so its resources can be freed
  is_done/1, % true if generator is done,
  empty/1, % empty stream generator, set to "done" up front
  op(800,xfx,(in)), % exports in/2
  op(800,xfx,(in_)), % exports in_/2
  in/2, % like member/2, for lists - test or generates
  in_/2, % like in/2, but working on a stream expression
  show/1, % prints a list representation of an initial segment of a stream
  show/2, % prints the first K elements of a stream as a list
  eng/3, % engine-based answer stream generator
  ceng/3, % clonable engine-based answer stream generator with copy of goal and template - assumes side-effect free goal
  ceng_clone/2, % creates a clone using the goal + template wrapped in a generator
  const/2, % constant infinite stream
  rand/1, % random floating point number in 0..1 generator
  nat/1, % stream of natural numbers starting at 0
  pos/1, % stream of positive natural numbers starting at 1
  neg/1, % stream of negative integers starting at -1
  list/2, % creates a stream from a finite or infinite (lazy) list
  cycle/2, % creates infinite cyclic list from finite list
  clause_stream/2, % generator for all clauses matching a given head
  take/3, % generator for finite initial segment of given length
  drop/3, % skips initial segment of given length
  slice/4, % extracts finite slice between bounds, right bound excluded
  range/3, % generator for integers betwen two numbers, larger excluded
  setify/2, % morphs a gnerator into one that produces distinct elements
  cat/2, % concatenates a list of gnerators - all but last should be finite
  conv/1, % generator for N * N self-convolution
  sum/3, % generator for direct sum of two finite or infinite streams
  prod_/3, % generator for direct product of two finite or infinite streams, engine based
  prod/3, % generator for direct product of two finite or infinite streams
  cantor_pair/3, % Cantor's pairing function
  cantor_unpair/3, % Cantor's unpairing function
  int_sqrt/2, % integer square root - with Newton's method
  map/3, % generator obtained by applying a predicate to a stream
  map/4, % % generator obtained by applying a predicate to two streams
  zipper_of/3, % forms pairs from corresponding elements of two streams
  reduce/4, % reduces/folds a stream with given predicate of arity 2
  arith_sum/3, % adds 2 streams element by element
  mult/3, % product of two numbers
  arith_mult/3, % multiplies 2 streams element by element
  chain/3, % generator derived from other by applying a transformer of arity 2
  chains/3, % generator piping a stream through a list of transformers of arity 2
  mplex/3, % multiplexes a stream to by applying to its output a list of generators
  split/3,
  lazy_nats/1, % infinite lazy list of natural numbers starting at 0
  lazy_maplist/3, % maplist working also on an infinite lazy list
  lazy_maplist/4, % maplist working also on two infinite lazy lists
  gen2lazy/2, % turns a generator into an isomorphic lazy list
  lazy2gen/2, % turns a lazy list into and isomorphic generator
  iso_fun/5, % functor transporting predicates of arity 2 between isomorphic domains
  iso_fun/6, % functor transporting predicates of arity 3 between isomorphic domains
  iso_fun_/6, % functor transporting predicates of arity 3 between isomorphic domains on ar in 2 out
  lazy_conv/3, % convolution of lazy lists
  convolution/3, % convolution of streams, borrowed from lazy lists
  eval_stream/2, % evaluates stream generator expression to generator
  do/1, % runs a goal exclusively for its side effects
  term_reader/2, % term stream generator, by reading from a Prolog file
  fact/2, % generator for infinite stream of factorials
  fibo/1, % generator for infinite stream of Fibonacci numbers
  prime/1, % simple engine-based generator for the infinite stream of prime numbers
  tests/0, % run all tests, with output to tests.txt
  bm/0 % run all benchmarks
]).

:-use_module(dynamic_arrays).
:-use_module(library(lazy_lists)).
:-use_module(library(solution_sequences)).
%:-include(test_lazy_streams).
:-ensure_loaded(test_lazy_streams).

%! ask(+Generator, -NextValue) 
%
% the Generator generator protocol works as follows:
% A generator step is a call to a closure that moves its state forward
% defining a generator simply stores it as a Prolog fact.
%
% The predicate ask/2 queries a generator if it is not done and
% it marks a generator as "done" after its first failure.
% This ensures it can be garbage collected
% by making its handle unreacheable.
%
% ask/2 extracts X by calling state transformer E.
ask(E,_):-is_done(E),!,fail.
ask(E,R):-call(E,X),!,R=X.
ask(E,_):-stop(E),fail.

%! stop(+Generator) 
%
% stop/1 marks a generator as done.
% Future calls to it will fail
stop(E):-
  arg(1,E,Handle),
  (is_engine(Handle),engine_destroy(Handle) ; true),
  nb_linkarg(1,E,done).

%! is_done(+Generator)
%
% checks if a generator is done
is_done(E):-arg(1,E,done).

%! empty(-Done)
%
% empty stream generator, set to "done" up front
 
empty(state(done)).

:-op(800,xfx,(in)).

%! in(-Value, +Generator)
%
% in/2 backtracks over progressively advancing states.
% in/2 is an xfx infix operator of priority 800

X in Gen:-ask(Gen,A),select_from(Gen,A,X).

select_from(_,X,X).
select_from(Gen,_,X):-X in Gen.

%! show(+NumberOfItems, +Generator)
% show/2 collects results after K steps and prints them out
% same as: show(K,Stream):-once(findnsols(K,X,X in Stream,Xs)),writeln(Xs).

show(K,Gen):-nexts_of(Gen,K,Xs),portray_clause(Xs).

%! show(+Generator)
% collects and prints 12 results of Generator
show(Stream):-show(12,Stream).


nexts_of(E,SK,[X|Xs]):-succ(K,SK),ask(E,X),!,nexts_of(E,K,Xs).
nexts_of(_,_,[]).

% CONSTRUCTORS of simple generators

%! const(+Constant, -Generator)
% Builds a constant infinite stream returning its first argument.
% the "next" step, call(=(C),X) will simply unify X and C
const(C,=(C)).

%! rand(+RandomStreamGenerator)
%
% produces a stream of random floating point numbers between 0 and 1
rand(random()).


%! gen_next(+Advancer,InitialState+,-AnswerTemplate)
%
% generic simple stream advancer
gen_next(F,State,X):-
  arg(1,State,X),
  call(F,X,Y),
  nb_linkarg(1,State,Y).

% generator step for natural numbers
nat_next(S,X):-gen_next(succ,S,X).

%! nat(-NaturalNumberStream)
%
% Natural number generator, storing the next and its initial state.
nat(nat_next(state(0))).

%! pos(-PositiveIntegerStream)
%
% stricly positive integers
pos(nat_next(state(1))).

% predecessor defined on all integers
pred(X,PX):-PX is X-1.

%! neg(-NegativeIntgerStream)
% strictly negative integers
neg(gen_next(pred,state(-1))).




%! gen_nextval(+Advancer,+InitialState,-Yield)
%
% Generic advancer, where State and Yield are distinct.
gen_nextval(Advancer,State,Yield):-
  arg(1,State,X1),
  call(Advancer,X1,X2, Yield),
  nb_linkarg(1,State,X2).

  % TODO: explain why/when nb_linkarg is ok, or 
% prove with counter examples that it is not
% convolution seems sensitive to it

% like gen_nextval, but copying X2
gen_safe_nextval(F,State,Yield):-
  arg(1,State,X1),
  call(F,X1,X2, Yield),
  nb_setarg(1,State,X2).
 
%! list(+ListOrLazyList,-Stream) 
%
% Builds stream generator from list or lazy list.  
list(Xs,gen_nextval(list_step,state(Xs))).

list_step([X|Xs],Xs,X).

% alternative, direct implementation
%list(Xs,listnext(state(Xs))).
%listnext(S,X):-arg(1,S,[X|Xs]),nb_linkarg(1,S,Xs).

%! range(+From,+To,-RangeStream)
%
% finite positive integer range generator
range(From,To,gen_next(range_step(To),state(From))).

% moves forward by incrementing state content
range_step(To,X,SX):-X<To,succ(X,SX).

%! cycle(+StreamGenerator, -CycleStreamGenerator)
%
% transforms a finite generator into an infinite cycle
% advancing over its elements repeatedly.
% Uses a circular list, unified with its own tail.
cycle(E,CycleStream):-
  findall(X,X in E,Xs,Xs), % creates circular infinite list
  list(Xs,CycleStream).



% engine-based generators

% UNWRAPPED, expendable

%! eng(+AnswerTemplate,+Goal, -Generator)
%
% Generator exposing the work of an engine as a stream of answers.  
eng(X,Goal,engine_next(E)):-engine_create(X,Goal,E).  

%! ceng(+Answertemplate,+Goal, -Generator)
%
% Clonable generator exposing the work of an engine as a stream of answers.  
% It works on a generator wrapping an engine
% such that its goal and answer template are kept.
% That makes it clonable, assuming it runs code that's side-effect free.
ceng(X,G,ask_generator(Gen)):-new_generator(X,G,Gen).



% WRAPPED with goal+template, reusable

%! ceng_clone(+Generator, -Clone)
% creates new generator from a generator's goal
ceng_clone(ask_generator(engine(_E,X,G)),NewGen):-ceng(X,G,NewGen).

%! new_generator(+AnswerTemplate,+Goal, -Generator)
%
% Creates a new generator, made of an engine and a goal for possible cloning.
new_generator(X,G,engine(E,X,G)):-engine_create(X,G,E).

%! ask_generator(Generator+, -Yield)
%
% Extracts next answer from generator wrapping an engine.
ask_generator(G,X):-arg(1,G,E),engine_next(E,X).


% stream processors

%! take(+K,+Generator, -NewGenerator)
%
% Builds generator for initial segment of length K of given generator. 
take(K,E,take_next(state(K,E))).

% advances by asking generator - not more than K times
take_next(State,X):-
  State=state(K,E),
  succ(PK,K),
  ask(E,X),
  nb_linkarg(1,State,PK).

%! drop(+K,+Generator, -NewGenerator)
%
% Roll the stream to first postion after first K items.
% Returns generator positioned K steps forward.
drop(K,E,_):-succ(PK,K),once(offset(PK,_ in E)),fail.
drop(_,E,E).

%! slice(+From,+To,+Generator, -NewGenerator)
% Builds generator for a slice of a given stream From..To (excluding To).
slice(From,To)-->{K is To-From,K>=0},drop(From),take(K).

%! setify(+Gen, -NewGen)
%
% Transforms a generator into one that produces distinct elements.
% It avoids sorting and uses the built-in distinct/2 to ensure
% that it also works on infinite streams.
setify(Gen,NewGen):-eng(X,distinct(X,X in Gen),NewGen).

% lazy functional operators  


%! map(+Closure,+Generator,-NewGenerator)
%
% Builds a generator that will apply a closure to each element of a given generator.
map(F,E,map_next(F,E)).

% advances E and applies F to result X
map_next(F,E,Y):-ask(E,X),call(F,X,Y).


%! map(+Closure,+Generator1,+Generator2, -NewGenerator)
%
% Builds a generator that combines two gnerators by creating 
% an advancer that applies a Closure to their "next" yields.
map(F,E1,E2,map_next(F,E1,E2)).

% advances bith and applies F
map_next(F,E1,E2,Z):-ask(E1,X),ask(E2,Y),call(F,X,Y,Z).


%! reduce(+Closure, +Generator, +InitialVal, -ResultGenerator)
% Builds generator that reduces given generator's yields with given closure, 
% starting with an initial value. Yields the resulting single final value.
reduce(F,InitVal,Gen,reduce_next(state(InitVal),F,Gen)).

% reduces state S while E provides "next" elements
reduce_next(S,F,E,R):-
  \+ is_done(E),
  do((
    Y in E,
    arg(1,S,X),
    call(F,X,Y,Z),
    nb_linkarg(1,S,Z)
  )),
  arg(1,S,R).

  %! do(+Goal)
%
% Bactracks over Goal for its side-effects only.  
do(G):-call(G),fail;true.


%! zipper_of(+Generator1,+Generator2, -NewGenerator)
% zipper_of/3 collects pairs of elements in matching positions
% in given two generators, finite of the same length or infinite.
zipper_of(E1,E2,E):-map(zip2,E1,E2,E).

% forms pair
zip2(X,Y,X-Y).


%! arith_sum(+Gen1,+Gen2, -NewGen)
%
% Elementwise addition of two streams.
arith_sum(E1,E2,S):-map(plus,E1,E2,S).

%! mult(+X,+Y,-P)
%
% P is the result of the multiplication of two numbers
mult(X,Y,P):-P is X*Y.

%! arith_mult(+Gen1,+Gen2, -NewGen)
%
% Elementwise multiplication of two streams.
arith_mult(E1,E2,P):-map(mult,E1,E2,P).

chain_next(F,E,Y):-ask(E,X),call(F,X,Y).

%! chain(+Closure,+Generator, -Newgenerator) 
%
% Pipes  elements of a stream through a transformer.
chain(F,E,chain_next(F,E)).

%! chains(+ListOfClosures,+Generator, -Newgenerator) 
%
% Pipes stream through a list of transformers.
chains([])-->[].
chains([F|Fs])-->chain(F),chains(Fs).

%! mplex(+Closures, +Gen, -Values)
%
% multiplexes a stream through a list of transfomers
% returns the list of values obtained by appying each 
% transformer to the next lement of the generator
mplex(Fs,E,mplex_next(state(E,Fs))).

mplex_next(state(E,Fs),Ys):-
  ask(E,X),
  maplist(revcall(X),Ys,Fs).

revcall(X,Y,F):-call(F,X,Y).

%! clause_stream(+Head, -StreamOfMatchingClauses)
%
% generates a stream of clauses matching a given goal
clause_stream(H,gen_nextval(clause_stream_step(H),state(1))).

clause_stream_step(H,I,SI,(NewH:-Bs)):-
  succ(I,SI),
  nth_clause(H,I,Ref),
  clause(NewH,Bs,Ref).



% sequence sum and product operrations  

%! sum(+Gen1,+Gen2, -NewGen)
%
% Interleaved sum merging two finite or infinite generators.
sum(E1,E2,sum_next(state(E1,E2))).

sum_next(State,X):-
  State=state(E1,E2),
  ask(E1,X),
  !,
  nb_linkarg(1,State,E2),
  nb_linkarg(2,State,E1).
sum_next(state(_,E2),X):-
  ask(E2,X).

%! cat(+GeneratorList, -ConcatenationOfGenerators)
% 
% concatenates streams of a list of generators
% Int only makes sense if all but the last one are finite.
cat(Es,cat_next(Es)).

cat_next(Es,X):-member(E,Es),ask(E,X),!.


%! prod_(+Gen1,+Gen2, -NewGen)
%
% engine-based direct product
prod_(E1,E2,E):-eng(_,prod_goal(E1,E2),E).

prod_goal(E1,E2):-
  ask(E1,A),
  prod_loop(1,A,E1-[],E2-[]).

prod_loop(Ord1,A,E1-Xs,E2-Ys):-
  flip(Ord1,Ord2,A,Y,Pair),
  forall(member(Y,Ys),engine_yield(Pair)),
  ask(E2,B),
  !,
  prod_loop(Ord2,B,E2-Ys,E1-[A|Xs]).
prod_loop(Ord1,_A,E1-_Xs,_E2-Ys):-
  flip(Ord1,_Ord2,X,Y,Pair),
  X in E1,member(Y,Ys),
  engine_yield(Pair),
  fail.
  
flip(1,2,X,Y,X-Y).
flip(2,1,X,Y,Y-X).

%! prod(+Gen1,+Gen2, -NewGen)
%
% direct product of two finite or infinite generators
prod(E1,E2,prod_next(state(0,E1,E2,A1,A2))):-
  new_array(A1),
  new_array(A2).

prod_next(S,X-Y):-
  S=state(_,E1,E2,A1,A2),
  %conv(C),
  repeat,
    ( is_done(E1),is_done(E2) -> !,fail
    ; nat_pair_next(S,I-J),
      %ask(C,I-J),
      fill_to(I,E1,A1,X),
      fill_to(J,E2,A2,Y)
    ),
    !.
  
fill_to(N,E,A,R):-
  array_size(A,L),
  do((
    between(L,N,_),
    ask(E,X),
    push_to(A,X)
  )),
  array_get(A,N,R),
  nonvar(R).
  
%! nat_pair(+PairGenerator)  
% generator of natural number pairs
% 
nat_pair(nat_pair_next(state(0))).

nat_pair_next(S,A-B):-nat_next(S,X),cantor_unpair(X,B,A).


%! cantor_pair(+Int1,+Int2, -Int)
%
% Cantor's pairing function
cantor_pair(K1,K2,P):-P is (((K1+K2)*(K1+K2+1))//2)+K2.

%! cantor_unpair(+Int, -Int1,-Int2)
%
% Inverse of Cantor's pairing function.
cantor_unpair(Z,K1,K2):-
  E is 8*Z+1,
  int_sqrt(E,R),
  I is (R-1)//2,
  K1 is ((I*(3+I))//2)-Z,
  K2 is Z-((I*(I+1))//2).

%! int_sqrt(+PosInt,-IntSqrt)
%
% computes integer square root using Newton's method
int_sqrt(0,0).
int_sqrt(N,R):-N>0,
  iterate(N,N,K),K2 is K*K,
  (K2>N->R is K-1;R=K).

% iterates until close enough   
iterate(N,X,NewR):-
  R is (X+(N//X))//2,
  A is abs(R-X),
  (A<2->NewR=R;iterate(N,R,NewR)).

conv_pairs(N,Ps):-conv_pairs(N,0,Ps).

conv_pairs(0,L,[L-0]).
conv_pairs(K,L,[L-K|Ps]):-
  succ(PK,K),succ(L,SL),
  conv_pairs(PK,SL,Ps).
 
%! conv(-Generator) 
%
% Generator for N * N self-convolution.
conv(gen_safe_nextval(conv_step,state(0-[0-0]))).

conv_step(N-[X|Xs],N-Xs,X).
conv_step(N-[],SN-Xs,X):-succ(N,SN),conv_pairs(SN,[X|Xs]).


  
  
%nobug:-conv(C),(ask(C,_),ask(C,_),ask(C,_),fail;ask(C,B)),writeln(B).


% ISO-FUNCTOR to/from lazy_lists

% data transformers

% can this be implemented directly, thus with no use of an engine?
%gen2lazy(E,Xs):-lazy_findall(X,X in E,Xs).

% YES!

%! lazy_nats(-LazyListOfNaturalNumbers)
%
% infinite lazy list of natural numbers
lazy_nats(L):-lazy_list(lazy_nats_next,0,L).

lazy_nats_next(X,SX,X):-succ(X,SX).
  
%! gen2lazy(+Generator,-LazyLIst)  
%
% Turns a generator into a lazy list  
gen2lazy(E,Ls):-lazy_list(gen2lazy_forward,E,Ls).

% E manages its state, so we just pass it on
gen2lazy_forward(E,E,X):-ask(E,X).

%! lazy2gen(+LazyList, -Generator)
%
% Turns a lazy list into a generator.
% Note that list/2 actually just  works on lazy lists!
lazy2gen(Xs,E):-list(Xs,E).

% iso functors

%! iso_fun(+Operation,+SourceType,+TargetType,+Arg1, -ResultOfSourceType)
%
% Transports a predicate of arity 2 F(+A,-B) to a domain where
% an operation can be performed and brings back the result.
iso_fun(F,From,To,A,B):-
  call(From,A,X),
  call(F,X,Y),
  call(To,Y,B).

%! iso_fun(+Operation,+SourceType,+TargetType,+Arg1,+Arg2, -ResultOfSourceType)
%
% Transports a predicate of arity 2 F(+A,+B,-C) to a domain where
% an operation can be performed and brings back the result. 
% transports F(+A,+B,-C) 
iso_fun(F,From,To,A,B,C):- % writeln(iso_fun(F,From,To,A,B,C)),
  call(From,A,X),
  call(From,B,Y),
  call(F,X,Y,Z),
  call(To,Z,C).

%! iso_fun_(+Operation,+SourceType,+TargetType,+Arg1, -Res1, -Res2)
%
% Transports a predicate of arity 2 F(+A,-B,-C) to a domain where
% an operation can be performed and brings back the results. 
% transports F(+A,+B,-C) 
iso_fun_(F,From,To,A,B,C):- 
  call(From,A,X),
  call(F,X, Y,Z), % X in, Y,Z out 
  call(To,Y,B),
  call(To,Z,C).
  
%! lazy_list_prod(+Xs,+Ys, -Zs)  
% Lazy lists borrow direct product from generators.
lazy_list_prod(Xs,Ys,Zs):-
  iso_fun(prod,lazy2gen,gen2lazy,Xs,Ys,Zs).

%! lazy_maplist(+F,+LazyXs, -LazyYs)
% Applies a predicate to a lazy list resulting in anoter lazy list
% Works with infinite list as input.
%
% Lazy lists are not plain lists, as proven by applying maplist:
% This loops!
% ?-lazy_nats(Ns),maplist(succ,Ns,Ms).
%
% lazy_maplist/3 fixes that.
lazy_maplist(F,LazyXs,LazyYs):-
  iso_fun(map(F),lazy2gen,gen2lazy,LazyXs,LazyYs).

%! lazy_maplist(+F,+LazyXs,LazyYs, -LazyYs)
%
% like maplist/4, but working on (possibly infinite) lazy lists
lazy_maplist(F,LazyXs,LazyYs,LazyZs):-
  iso_fun(map(F),lazy2gen,gen2lazy,LazyXs,LazyYs,LazyZs).  

%! split(+E, -E1, -E2)
%
% split/3 uses lazy lists to split a stream into two.
% infelicity: original stream shifted by one position ...
split(E,E1,E2):-iso_fun_(lazy_dup,gen2lazy,lazy2gen,E,E1,E2).
  
lazy_dup(Xs,Xs,Xs).

%! lazy_conv(+As,+Bs,-Ps) 
% 
% convolution of two finite or infinite lazy lists
lazy_conv(As,Bs,Ps):-
  lazy_findall(P,(
    between(0,infinite,N),
    lazy_lconv_step(N,As,Bs,P)
  ),
  Ps).

lazy_lconv_step(N,As,Bs, X-Y):-
   N1 is N-1,
   between(0,N1,L),
   K is N1-L,
   nth0(L,As,X),
   nth0(K,Bs,Y).

%! convolution(+Gen1,+Gen2,-NewGen) 
% 
% convolution of two finite or infinite lazy generators
convolution(E1,E2, E):-
  iso_fun(lazy_conv,gen2lazy,lazy2gen,E1,E2, E).
   
% evaluator  

%! eval_stream(+GeneratorExpression, -Generator)
% evaluates a generator expression to ready to use
% generator that combines their effects
eval_stream(E+F,S):- !,eval_stream(E,EE),eval_stream(F,EF),sum(EE,EF,S).
eval_stream(E*F,P):- !,eval_stream(E,EE),eval_stream(F,EF),prod(EE,EF,P).
eval_stream(E:F,R):- !,range(E,F,R).
eval_stream([X|Xs],L):-!,list([X|Xs],L).
eval_stream({E},SetGen):-!,eval_stream(E,F),setify(F,SetGen).
eval_stream(X^G,E):-!,eng(X,G,E).
eval_stream(A,C):-atomic(A),!,const(A,C).
eval_stream(E,E).

:-op(800,xfx,(in_)).

%! in_(-X, +GeneratorExpression)
%
% backtracks over elements of a generator expression
% note that in_/2 is an xfx 800 operator, used as X in_ Gen
X in_ GenExpr:-eval_stream(GenExpr,NewGen),X in NewGen.  
 
%! ask_(GeneratorExpression, -Element)
% 
% produces next element after evaluating a gnerator expression
ask_(E,X):-eval_stream(E,EE),ask(EE,X).



%! term_reader(+File,-TermGenerator)
%
% creates a generator advancing on terms read from a file

term_reader(File,next_term(Stream)):-open(File,read,Stream).

next_term(Stream,Term):-
  read(Stream,X),
  ( X\==end_of_file->Term=X
  ; close(Stream),fail
  ).

%! fact(+N,-ResultGenerator)
%
% factorial computation - use ask/2 to extract value
% used for testing
fact(N,F):-range(1,N,R),reduce(mult,N,R,F).

%! fibo(-Generator)
%
% infinite Fibonacci stream for testing
fibo(F):-fibo_pair(E),chain(arg(1),E,F).

fibo_pair(gen_next(fibo_pair_step,state(1-1))).

fibo_pair_step(X-Y,Y-Z) :- Z is X+Y.

%! prime(+Gen)
%
% simple engine-based generator for the infinite stream of prime numbers
prime(E):-eng(_,new_prime(1),E).

new_prime(N):-
  succ(N,SN),
  ( not_a_prime(SN)->true
  ; engine_yield(SN)
  ),
  new_prime(SN).
  
not_a_prime(N):-
   int_sqrt(N,M),
   between(2,M,D),
   N mod D =:=0.
   
   