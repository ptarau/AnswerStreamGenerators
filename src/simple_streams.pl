c:-['simple_streams.pl'].

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


:-op(800,xfx,(in)).

% in/2 backtracks over progressively advancing states

X in E:-ask_(E,A),select_from(E,A,X).

select_from(_,A,A).
select_from(E,_,X):-X in E.

% stop/1 marks a generator as done
% future calls to it will fail
stop_(E):-nb_linkarg(1,E,done).

% checks if a generator is done
is_done(E):-arg(1,E,done).

% collectsresults after K steps and prints them out
show(K,Stream):-once(findnsols(K,X,X in Stream,Xs)),writeln(Xs).

show(Stream):-show(12,Stream).


% constant infinite stream returning C
% the "next" step, call(=(C),X) will simply unify X and C
const_(C,=(C)).

% generic simple stream advancer
stream_next(F,State,X):-
  arg(1,State,X),
  call(F,X,Y),
  nb_linkarg(1,State,Y).

% generator step for natural numbers
nat_next(S,X):-stream_next(succ,S,X).

% natural number generator, storing the next and its initial state
nat_(nat_next(state(0))).

% stricly positive integers
pos_(nat_next(state(1))).

% predecessor defined on all integers
pred(X,PX):-PX is X-1.

% strictly negative integers
neg_(stream_next(pred,state(-1))).

% finite stream from list
list_(Xs,list_next(state(Xs))).

% advancer to the tail of a list
list_next(State,X):-  
  arg(1,State,[X|Xs]),
  nb_linkarg(1,State,Xs).

  % finite integer range generator
range_(From,To,range_next(state(From,To))).

% moves forward by incrementing state content
range_next(State,X):-
  State=state(X,To),
  X<To,
  succ(X,SX),
  nb_linkarg(1,State,SX).

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
gen_(X,G,ask_generator(Gen)):-new_generator(X,G,Gen).

% WRAPPED, reusable

% creates new generator from a generator's goal
gen_clone(ask_generator(engine(_E,X,G)),NewGen):-gen_(X,G,NewGen).

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

% fibo(

% sequence sum and product operrations  

% interleaved sum of two finite or infinite streams
sum_(E1,E2,sum_next(state(E1,E2))).

sum_next(State,X):-
  State=state(E1,E2),
  ask_(E1,X),
  !,
  nb_linkarg(1,State,E2),
  nb_linkarg(2,State,E1).
sum_next(state(_,E2),X):-
  ask_(E2,X).

% cartesian product of two finite or infinite streams
prod_(E1,E2,prod_next(state(0,E1,E2,A1,A2))):-
  new_array(A1),
  new_array(A2).

prod_next(S,X-Y):-
  S=state(_,E1,E2,A1,A2),
  repeat,
    ( is_done(E1),is_done(E2) -> !,fail
    ; nat_pair_next(S,I-J),
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
  
% stream of naturla number pairs
nat_pair_(nat_pair_next(state(0))).

nat_pair_next(S,A-B):-nat_next(S,X),cantor_unpair(X,A,B).


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
  
% tests

  
t1:-nat_(N),list_([10,20,30],M),map_(plus,N,M,R),show(R).
 
t2:-nat_(N),nat_(M),map_(plus,N,M,R),show(R).  

t3:-range_(1,5,E),reduce_(plus,0,E,R),show(R).

t4:-pos_(N),neg_(M),sum_(M,N,S),show(S). 

t5:-nat_(N),list_([a,b,c],M),sum_(N,M,S),show(S).

t6:-range_(1,3,N),list_([a,b,c,d,e],M),sum_(M,N,S),show(S).
  
t7:-nat_(N),slice(4,8,N,S),show(S).

t8:-neg_(A),pos_(B),prod_(A,B,P),
   take(30,P,T),show(30,T).

t9:-nat_(A),list_([a,b,c],B),prod_(A,B,P),
    take(20,P,T),forall(X in T,writeln(X)).

t10:-range_(0,5,A),list_([a,b,c],B),prod_(A,B,P),
    take(20,P,T),show(30,T).
    
t11:-nat_(A),list_([a,b,c],B),
  prod_(B,A,P),take(20,P,T),
  show(30,T).
  
t12:-const_(10,C),nat_(N),map_(plus,C,N,R),show(R).

t13:-const_(10,C),nat_(N),prod_(C,N,P),show(P).


t14:-eng_(_X,fail,E),list_([a,b],L),sum_(E,L,S),show(S).
  
t15:-eng_(X,member(X,[1,2,3]),E),list_([a,b],L),sum_(E,L,S),show(S).

t16:-eng_(X,member(X,[1,2,3]),E),list_([a,b],L),prod_(E,L,S),show(S).

t17:-eng_(X,member(X,[1,2,3]),S),(X in S,writeln(X),fail;is_done(S),writeln(S)).

t18:-(X^member(X,[1,2,3])*[a,b])=E,do((X in_ E,writeln(X))).

t19:-range_(1,5,R),cycle_(R,C),show(20,C).

t20:-range_(1,4,R),cycle_(R,C),list_([a,b,c,d,e,f],L),zipper_of(C,L,Z),show(Z).

t21:-eng_(X,member(X,[a,b,c]),G),range_(1,6,R),prod_(G,R,P),show(P).

t22:-gen_(X,member(X,[a,b,c]),G),gen_clone(G,CG),prod_(G,CG,P),show(P).

t23:-gen_(X,member(X,[a,b,c]),G),cycle_(G,C),show(C).


t24:-range_(0,10,A),range_(100,110,B),arith_sum(A,B,S),show(S).

t25:-fact(5,S),show(S).

odds(Xs) :-
  lazy_findall(X, (between(0, infinite, X0),X is 2*X0+1), Xs).

% lazy_findall leaves undead engine
t26:-odds(Xs),list_(Xs,L),nat_(N),prod_(L,N,P),show(P).



tests:-
  tell('tests.txt'),
  do((between(1,26,I),atom_concat(t,I,T),listing(T),call(T),nl)),
  do((current_engine(E),writeln(E))),
  bm,
  told.

time(G,T):-get_time(T1),once(G),get_time(T2),T is T2-T1. 
  
bm1(K):-
  nl,listing(bm1),
  N is 2^K,writeln(with_lazy_lists:N),
  lazy_findall(I,between(0,N,I),Is),
  maplist(succ,Is,Js),last(Js,X),writeln([X]).

bm2(K):-
  nl,listing(bm2),
  N is 2^K,N1 is N+1,
  writeln(with_engine_based_generators:N),
  eng_(I,between(0,N,I),R),
  map_(succ,R,SR),
  slice(N,N1,SR,S),
  show(S).
  
bm3(K):-
  nl,listing(bm3),
  N is 2^K,N1 is N+1,
  writeln(with_simple_generators:N),
  range_(0,N1,R),
  map_(succ,R,SR),
  slice(N,N1,SR,S),
  show(S).
  
bm(K):-maplist(time,[bm1(K),bm2(K),bm3(K)],Ts),nl,writeln(times=Ts).

bm:-bm(21).  
  
