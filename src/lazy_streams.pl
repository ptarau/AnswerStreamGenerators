/*
Lazy Stream Generators of finite or infinite sequences and their operations.

Their operations have  isomorphic lazy list counterparts, the main difference being the use of a generic sequence manipulation API instead the use of a concrete list syntax in the case of lazy list. This isomorphism supports delegating operations between lazy lists and lazy streams.
*/

:- module(lazy_streams,[
  ask/2, % ask for next element of a lazy stream generator
  ask_/2, % like ask, but working in a stream expression
  stop/1, % stop/1 marks a generator as done so its resources can be freed
  is_done/1, % true if generator is done,
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
  prod/3, % generator for direct product of two finite or infinite streams
  cantor_pair/3, % Cantor's pairing function
  cantor_unpair/3, % Cantor's unpairing function
  int_sqrt/2, % integer square root - with Newton's method
  sum/3, % generator for direct sum of two finite or infinite streams
  conv/1, % generator for N * N self-convolution
  map/3, % generator obtained by applying a predicate to a stream
  map/4, % % generator obtained by applying a predicate to two streams
  zipper_of/3, % forms pairs from corresponding elements of two streams
  reduce/4, % reduces/folds a stream with given predicate of arity 2
  arith_sum/3, % adds 2 streams element by element
  arith_mult/3, % multiplies 2 streams element by element
  chain/3, % generator derived from other by applying a transformer of arity 2
  chains/3, % generator piping a stream through a list of transformers of arity 2
  mplex/3, % multiplexes a stream to by applying to its output a list of generators
  lazy_nats/1, % infinite lazy list of natural numbers starting at 0
  lazy_maplist/3, % maplist working also on an infinite lazy list
  lazy_maplist/4, % maplist working also on two infinite lazy lists
  gen2lazy/2, % turns a generator into an isomorphic lazy list
  lazy2gen/2, % turns a lazy list into and isomorphic generator
  iso_fun/5, % functor transporting predicates of arity 2 between isomorphic domains
  iso_fun/6, % functor transporting predicates of arity 3 between isomorphic domains
  do/1, % runs a goal exclusively for its side effects
  fact/2, % generator for infinite stream of factorials
  fibo/1 % generator for infinite stream of Fibonacci numbers
]).

:-use_module(dynamic_arrays).
:-use_module(library(lazy_lists)).
:-use_module(library(solution_sequences)).

% the Generator Generator protocol :
% a generator step is a call to a closure that moves its state forward
% defining a generator simply stores it as a Prolog fact

% ask/2 queries a generator if it is not done
% marks generator as "done" after its first failure
% this ensures it can be garbage collected
% by making its handle unreacheable
% extracts X by calling state transformer E
ask(E,_):-is_done(E),!,fail.
ask(E,R):-call(E,X),!,R=X.
ask(E,_):-stop(E),fail.

% stop/1 marks a generator as done
% future calls to it will fail
stop(E):-nb_linkarg(1,E,done).

% checks if a generator is done
is_done(E):-arg(1,E,done).

:-op(800,xfx,(in)).

% in/2 backtracks over progressively advancing states

X in E:-ask(E,A),select_from(E,A,X).

select_from(_,A,A).
select_from(E,_,X):-X in E.

% collects results after K steps and prints them out
show(K,Stream):-once(findnsols(K,X,X in Stream,Xs)),writeln(Xs).

% collects and prints 12 results
show(Stream):-show(12,Stream).


% CONSTRUCTORS of simple generators

% constant infinite stream returning C
% the "next" step, call(=(C),X) will simply unify X and C
const(C,=(C)).

% generic simple stream advancer
gen_next(F,State,X):-
  arg(1,State,X),
  call(F,X,Y),
  nb_linkarg(1,State,Y).

% TODO: explain why nb_linkarg is ok, or 
% prove with couter example that it is not

% generator step for natural numbers
nat_next(S,X):-gen_next(succ,S,X).

% natural number generator, storing the next and its initial state
nat(nat_next(state(0))).

% stricly positive integers
pos(nat_next(state(1))).

% predecessor defined on all integers
pred(X,PX):-PX is X-1.

% strictly negative integers
neg(gen_next(pred,state(-1))).

% generic advance, where state and yield are distinct
gen_nextval(F,State,Yield):-
  arg(1,State,X1),
  call(F,X1,X2, Yield),
  nb_linkarg(1,State,X2).

% like gen_nextval, but copying X2
gen_safe_nextval(F,State,Yield):-
  arg(1,State,X1),
  call(F,X1,X2, Yield),
  nb_setarg(1,State,X2).
  
% stream from list or lazy_list   
list(Xs,gen_nextval(list_step,state(Xs))).
list_step([X|Xs],Xs,X).

% alternative, direct implementation
%list(Xs,listnext(state(Xs))).
%listnext(S,X):-arg(1,S,[X|Xs]),nb_linkarg(1,S,Xs).


% finite integer range generator
range(From,To,gen_next(range_step(To),state(From))).

% moves forward by incrementing state content
range_step(To,X,SX):-X<To,succ(X,SX).

% transforms a finite generator into an infinite cycle
% uses a circular list, unified with its own tail
cycle(E,CycleStream):-
  findall(X,X in E,Xs),
  append(Xs,Tail,Tail), % creates circular infinite list
  list(Tail,CycleStream).


% engine-based generators

% UNWRAPPED, expendable

% work of an engine exposed as a stream  
eng(X,G,engine_next(E)):-engine_create(X,G,E).  

% works on a generator wrapping an engine
% such that its goal and answer template are kept
% that makes in clonable - assuming it runs pure code
ceng(X,G,ask_generator(Gen)):-new_generator(X,G,Gen).



% WRAPPED with goal+template, reusable

% creates new generator from a generator's goal
ceng_clone(ask_generator(engine(_E,X,G)),NewGen):-ceng(X,G,NewGen).

% creates new generator = engine plus goal for possible cloning
new_generator(X,G,engine(E,X,G)):-engine_create(X,G,E).

% extracts next answer from wrapped engine
ask_generator(G,X):-arg(1,G,E),engine_next(E,X).


% stream processors

% generator for initial segment of length K of generator E 
take(K,E,take_next(state(K,E))).

% advances by asking generator - not more than K times
take_next(State,X):-
  State=state(K,E),
  succ(PK,K),
  ask(E,X),
  nb_linkarg(1,State,PK).

% roll the stream after first K items
drop(K,E,_):-succ(PK,K),once(offset(PK,_ in E)),fail.
drop(_,E,E).

% slice of a stream From.. To (excluding To)
slice(From,To)-->{K is To-From,K>=0},drop(From),take(K).


% lazy functional operators  
map(F,E,map_next(F,E)).

% advances E and applies F to result X
map_next(F,E,Y):-ask(E,X),call(F,X,Y).

% combines E1 and E2  by creating an advancer 
% that applies F to their "next" states
map(F,E1,E2,map_next(F,E1,E2)).

% advances bith and applies F
map_next(F,E1,E2,Z):-ask(E1,X),ask(E2,Y),call(F,X,Y,Z).


% reduces E with F, starting with initial value
reduce(F,InitVal,E,reduce_next(state(InitVal),F,E)).

% bactrack over G for its side-effects only  
do(G):-call(G),fail;true.

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

% collects pairs of elements in matching positions from E1 and E2
zipper_of(E1,E2,E):-map(zip2,E1,E2,E).

% forms pair
zip2(X,Y,X-Y).

% elementwise addition of two streams
arith_sum(E1,E2,S):-map(plus,E1,E2,S).

mult(X,Y,P):-P is X*Y.

% elementwise multiplication of two streams
arith_mult(E1,E2,P):-map(mult,E1,E2,P).

chain_next(F,E,Y):-ask(E,X),call(F,X,Y).
 
% pipes  elements of a stream through one transformer
chain(F,E,chain_next(F,E)).

% pipes stream through a list of transformers
chains([])-->[].
chains([F|Fs])-->chain(F),chains(Fs).

% multiplexes a stream through a list of transfomers
mplex(Fs,E,mplex_next(state(E,Fs))).

mplex_next(state(E,Fs),Ys):-
  ask(E,X),
  maplist(revcall(X),Ys,Fs).

revcall(X,Y,F):-call(F,X,Y).

% generates a stream of clauses matching a given goal
clause_stream(H,gen_nextval(clause_stream_step(H),state(1))).

clause_stream_step(H,I,SI,(NewH:-Bs)):-
  succ(I,SI),
  nth_clause(H,I,Ref),
  clause(NewH,Bs,Ref).



% sequence sum and product operrations  

% interleaved sum of two finite or infinite generators
sum(E1,E2,sum_next(state(E1,E2))).

sum_next(State,X):-
  State=state(E1,E2),
  ask(E1,X),
  !,
  nb_linkarg(1,State,E2),
  nb_linkarg(2,State,E1).
sum_next(state(_,E2),X):-
  ask(E2,X).

% cartesian product of two finite or infinite generators
prod(E1,E2,prod_next(state(0,E1,E2,A1,A2))):-
  new_array(A1),
  new_array(A2).

prod_next(S,X-Y):-
  S=state(_,E1,E2,A1,A2),
  %conv(C),
  repeat,
    ( is_done(E1),is_done(E2) -> !,fail
    ; nat_pair_next(S,I-J),
      %ask(C,I-J),ppp(I-J),
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
  
% generator of natural number pairs
nat_pair(nat_pair_next(state(0))).

nat_pair_next(S,A-B):-nat_next(S,X),cantor_unpair(X,B,A).


% cantor pairing function
cantor_pair(K1,K2,P):-P is (((K1+K2)*(K1+K2+1))//2)+K2.

% inverse of Cantor's pairing function
cantor_unpair(Z,K1,K2):-
  E is 8*Z+1,
  int_sqrt(E,R),
  I is (R-1)//2,
  K1 is ((I*(3+I))//2)-Z,
  K2 is Z-((I*(I+1))//2).

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
conv_pairs(K,L,[L-K|Ps]):-succ(PK,K),succ(L,SL),conv_pairs(PK,SL,Ps).
  
% generator for N * N self-convolution
conv(gen_safe_nextval(conv_step,state(0-[0-0]))).

conv_step(N-[X|Xs],N-Xs,X).
conv_step(N-[],SN-Xs,X):-succ(N,SN),conv_pairs(SN,[X|Xs]).
  
  
%nobug:-conv(C),(ask(C,_),ask(C,_),ask(C,_),fail;ask(C,B)),writeln(B).


% ISO-FUNCTOR to/from lazy_lists

% data transformers

% can this be implemented directly, thus with no use of an engine?
%gen2lazy(E,Xs):-lazy_findall(X,X in E,Xs).

% YES!

% infinite lazy list of natural numbers
lazy_nats(L):-lazy_list(lazy_nats_next,0,L).

lazy_nats_next(X,SX,X):-succ(X,SX).
  
% turns a generator into an isomorphic lazy list  
gen2lazy(E,Ls):-lazy_list(gen2lazy_forward,E,Ls).

% E manages its state, so we just pass it on
gen2lazy_forward(E,E,X):-ask(E,X).

% turns a lazy list into and isomorphic generator
% note that list also works on lazy lists!
lazy2gen(Xs,E):-list(Xs,E).

% iso functors - TODO: test

% transports F(A,B)
iso_fun(F,From,To,A,B):-
  call(From,A,X),
  call(F,X,Y),
  call(To,Y,B).
 
% transports F(A,B,C) 
iso_fun(F,From,To,A,B,C):-
  call(From,A,X),
  call(From,B,Y),
  call(F,X,Y,Z),
  call(To,Z,C).

% lazy lists borrow product from generators 
lazy_list_prod(Xs,Ys,Zs):-
  iso_fun(prod,lazy2gen,gen2lazy,Xs,Ys,Zs).

% lazy lists are not plain lists: this loops
% ?-lazy_nats(Ns),maplist(succ,Ns,Ms).
% lazy_maplist fixes that

lazy_maplist(F,LazyXs,LazyYs):-
  iso_fun(map(F),lazy2gen,gen2lazy,LazyXs,LazyYs).

lazy_maplist(F,LazyXs,LazyYs,LazyZs):-
iso_fun(map(F),lazy2gen,gen2lazy,LazyXs,LazyYs,LazyZs).  

% evaluator  

eeval(E+F,S):- !,eeval(E,EE),eeval(F,EF),sum(EE,EF,S).
eeval(E*F,P):- !,eeval(E,EE),eeval(F,EF),prod(EE,EF,P).
eeval(E:F,R):- !,range(E,F,R).
eeval([X|Xs],L):-!,list([X|Xs],L).
eeval(X^G,E):-!,eng(X,G,E).
eeval(A,C):-atomic(A),!,const(A,C).
eeval(E,E).

:-op(800,xfx,(in_)).
X in_ E:-eeval(E,EE),X in EE.
 
ask_(E,X):-eeval(E,EE),ask(EE,X).


% factorial for testing
fact(N,F):-range(1,N,R),reduce(mult,N,R,F).

% Fibonacci stream for testing
fibo(F):-fibo_pair(E),chain(arg(1),E,F).

fibo_pair(gen_next(fibo_pair_step,state(1-1))).

fibo_pair_step(X-Y,Y-Z) :- Z is X+Y.
