cc:-['generators.pl'].

:-ensure_loaded(arrays).

% the Generator Generator protocol
% a generator step is a call to a closure that moves its state forward
% defining a generator simply stores it as a Prolog fact

% ask_/2 queries a generator if it is not done
% marks generator as "done" after its first failure
% this ensures it can be garbage collected
% by making its handle unreacheable
% extracts X by calling state transformer E
ask_(E,_):-is_done(E),!,fail.
ask_(E,R):-call(E,X),!,R=X.
ask_(E,_):-stop_(E),fail.

% stop/1 marks a generator as done
% future calls to it will fail
stop_(E):-nb_linkarg(1,E,done).

% checks if a generator is done
is_done(E):-arg(1,E,done).

:-op(800,xfx,(in)).

% in/2 backtracks over progressively advancing states

X in E:-ask_(E,A),select_from(E,A,X).

select_from(_,A,A).
select_from(E,_,X):-X in E.

% collectsresults after K steps and prints them out
show(K,Stream):-once(findnsols(K,X,X in Stream,Xs)),writeln(Xs).

show(Stream):-show(12,Stream).


% CONSTRUCTORS of simple generators

% constant infinite stream returning C
% the "next" step, call(=(C),X) will simply unify X and C
const_(C,=(C)).

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
nat_(nat_next(state(0))).

% stricly positive integers
pos_(nat_next(state(1))).

% predecessor defined on all integers
pred(X,PX):-PX is X-1.

% strictly negative integers
neg_(gen_next(pred,state(-1))).

% generic advance, where state and yield are distinct
gen_nextval(F,State,Yield):-
  arg(1,State,X1),
  call(F,X1,X2, Yield),
  nb_linkarg(1,State,X2).

gen_safe_nextval(F,State,Yield):-
  arg(1,State,X1),
  call(F,X1,X2, Yield),
  nb_setarg(1,State,X2).
  
% stream from list or lazy_list   
list_(Xs,gen_nextval(list_step,state(Xs))).
list_step([X|Xs],Xs,X).

% alternative, direct implementation
%list_(Xs,list_next(state(Xs))).
%list_next(S,X):-arg(1,S,[X|Xs]),nb_linkarg(1,S,Xs).


% finite integer range generator
range_(From,To,gen_next(range_step(To),state(From))).

% moves forward by incrementing state content
range_step(To,X,SX):-X<To,succ(X,SX).

% transforms a finite generator into an infinite cycle
% uses a circular list, unified with its own tail
cycle_(E,CycleStream):-
  findall(X,X in E,Xs),
  append(Xs,Tail,Tail),
  list_(Tail,CycleStream).


% engine-based generators

% UNWRAPPED, expendable

% work of an engine exposed as a stream  
eng_(X,G,engine_next(E)):-engine_create(X,G,E).  

% works on a generator wrapping an engine
% such that its goal and answer template are kept
% that makes in clonable - assuming it runs pure code
ceng_(X,G,ask_generator(Gen)):-new_generator(X,G,Gen).



% WRAPPED, reusable

% creates new generator from a generator's goal
ceng_clone(ask_generator(engine(_E,X,G)),NewGen):-ceng_(X,G,NewGen).

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
  ask_(E,X),
  nb_linkarg(1,State,PK).

% roll the stream after first K items
drop(K,E,_):-succ(PK,K),once(offset(PK,_ in E)),fail.
drop(_,E,E).

% slice of a stream From.. To (excluding To)
slice(From,To)-->{K is To-From,K>=0},drop(From),take(K).


% lazy functional operators  
map_(F,E,map_next(F,E)).

% advances E and applies F to result X
map_next(F,E,Y):-ask_(E,X),call(F,X,Y).

% combines E1 and E2  by creating an advancer 
% that applies F to their "next" states
map_(F,E1,E2,map_next(F,E1,E2)).

% advances bith and applies F
map_next(F,E1,E2,Z):-ask_(E1,X),ask_(E2,Y),call(F,X,Y,Z).


% reduces E with F, starting with initial value
reduce_(F,InitVal,E,reduce_next(state(InitVal),F,E)).

% bactrack over G for its side-effects only  
do(G):-G,fail;true.

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
zipper_of(E1,E2,E):-map_(zip2,E1,E2,E).

zip2(X,Y,X-Y).

arith_sum(E1,E2,S):-map_(plus,E1,E2,S).

prod(X,Y,P):-P is X*Y.

arith_prod(E1,E2,P):-map_(prod,E1,E2,P).

fact(N,F):-range_(1,N,R),reduce_(prod,N,R,F).

chain_next(F,E,Y):-ask_(E,X),call(F,X,Y).
  
chain_(F,E,chain_next(F,E)).

chains_([])-->[].
chains_([F|Fs])-->chain_(F),chains_(Fs).


mplex_(Fs,E,mplex_next(state(E,Fs))).

mplex_next(state(E,Fs),Ys):-
  ask_(E,X),
  maplist(rcall(X),Ys,Fs).

rcall(X,Y,F):-call(F,X,Y).

fibo_pair(X-Y,Y-Z) :- Z is X+Y.

fibo_pair(gen_next(fibo_pair,state(1-1))).

fibo_(F):-fibo_pair(E),chain_(arg(1),E,F).


clause_(H,gen_nextval(clause_step(H),state(1))).

clause_step(H,I,SI,(NewH:-Bs)):-
  succ(I,SI),
  nth_clause(H,I,Ref),
  clause(NewH,Bs,Ref).



% sequence sum and product operrations  

% interleaved sum of two finite or infinite generators
sum_(E1,E2,sum_next(state(E1,E2))).

sum_next(State,X):-
  State=state(E1,E2),
  ask_(E1,X),
  !,
  nb_linkarg(1,State,E2),
  nb_linkarg(2,State,E1).
sum_next(state(_,E2),X):-
  ask_(E2,X).

% cartesian product of two finite or infinite generators
prod_(E1,E2,prod_next(state(0,E1,E2,A1,A2))):-
  new_array(A1),
  new_array(A2).

prod_next(S,X-Y):-
  S=state(_,E1,E2,A1,A2),
  %conv_(C),
  repeat,
    ( is_done(E1),is_done(E2) -> !,fail
    ; nat_pair_next(S,I-J),
      %ask_(C,I-J),ppp(I-J),
      fill_to(I,E1,A1,X),
      fill_to(J,E2,A2,Y)
    ),
    !.
  
fill_to(N,E,A,R):-
  array_size(A,L),
  do((
    between(L,N,_),
    ask_(E,X),
    push_to(A,X)
  )),
  array_get(A,N,R),
  nonvar(R).
  
% generator of natural number pairs
nat_pair_(nat_pair_next(state(0))).

nat_pair_next(S,A-B):-nat_next(S,X),cantor_unpair(X,B,A).


% cantor pairing function
cantor_pair(K1,K2,P):-P is (((K1+K2)*(K1+K2+1))//2)+K2.

% inverse of Cantor's pairing function
cantor_unpair(Z,K1,K2):-
  E is 8*Z+1,
  intSqrt(E,R),
  I is (R-1)//2,
  K1 is ((I*(3+I))//2)-Z,
  K2 is Z-((I*(I+1))//2).

% computes integer square root using Newton's method
intSqrt(0,0).
intSqrt(N,R):-N>0,
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
  

conv_(gen_safe_nextval(conv_step,state(0-[0-0]))).

conv_step(N-[X|Xs],N-Xs,X).
conv_step(N-[],SN-Xs,X):-succ(N,SN),conv_pairs(SN,[X|Xs]).
  
  
%nobug:-conv_(C),(ask_(C,_),ask_(C,_),ask_(C,_),fail;ask_(C,B)),writeln(B).


% ISO-FUNCTOR TO lazy_lists

% data transformers

% can this be implemented directly, thus with no use of an engine?
%gen2lazy(E,Xs):-lazy_findall(X,X in E,Xs).

% YES!

lazy_nats(L):-lazy_list(lazy_nats_next,0,L).

lazy_nats_next(X,SX,X):-succ(X,SX).
  
gen2lazy(E,Ls):-lazy_list(gen2lazy_forward,E,Ls).

% E manages its state, so we just pass it on
gen2lazy_forward(E,E,X):-ask_(E,X).

% list_ also works on lazy lists!
lazy2gen(Xs,E):-list_(Xs,E).

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
  iso_fun(prod_,lazy2gen,gen2lazy,Xs,Ys,Zs).

% bug: this loops
% lazy_nats(Ns),maplist(succ,Ns,Ms).

maplist_(F,LazyXs,LazyYs):-
  iso_fun(map_(F),lazy2gen,gen2lazy,LazyXs,LazyYs).

  
% evaluator  

eeval(E+F,S):- !,eeval(E,EE),eeval(F,EF),sum_(EE,EF,S).
eeval(E*F,P):- !,eeval(E,EE),eeval(F,EF),prod_(EE,EF,P).
eeval(E:F,R):- !,range_(E,F,R).
eeval([X|Xs],L):-!,list_([X|Xs],L).
eeval(X^G,E):-!,eng_(X,G,E).
eeval(A,C):-atomic(A),!,const_(A,C).
eeval(E,E).

:-op(800,xfx,(in_)).
X in_ E:-eeval(E,EE),X in EE.
 
ask__(E,X):-eeval(E,EE),ask_(EE,X).



% pipelines

% make_pipe([E|Es]):-make_pipe(Es,P),

% meta engines




% UNFINISHED - MAYBE NOT IMPORTANT
  
/*
next_unfold(S):-
  S=state([G|Gs]),
  next_clause(G,(CH:-Bs)),
*/  
  
  
/* TODO
meta_(H,meta_step(H)).

meta_step(H,B):-
  clause_(H,Hs),
  list_next(Hs,B),
  !.
meta_step(H,H).

add(0,X,X).
add(s(X),Y,s(Z)):-add(X,Y,Z).

goal(add(s(s(0)),s(s(0)),_R)).
*/

