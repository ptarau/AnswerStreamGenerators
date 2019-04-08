c:-['simple_streams.pl'].

:-ensure_loaded(arrays).

:-op(800,xfx,(in)).

% backtracks over progressively advancing states

X in E:-ask_(E,A),select_from(E,A,X).

select_from(_,A,A).
select_from(E,_,X):-X in E.

stop_(E):-nb_linkarg(1,E,done).

is_done(E):-arg(1,E,done).

% extracts X from state and calls transformer E
ask_(E,_):-is_done(E),!,fail.
ask_(E,R):-call(E,X),!,R=X.
ask_(E,_):-stop_(E),fail.

% constant infinite stream returning C
const_(C,=(C)).

% has_state(E):-arg(1,E,T),functor(T,state,_).

% generic simple stream advancer
stream_next(F,State,X):-
  arg(1,State,X),
  call(F,X,Y),
  nb_linkarg(1,State,Y).

nat_stream(S,X):-stream_next(succ,S,X).

% natural numbers
nat_(nat_stream(state(0))).

% stricly positive integers
pos_(nat_stream(state(1))).

% predecessor defined on all integers
pred(X,PX):-PX is X-1.

% strictly negative integers
neg_(stream_next(pred,state(-1))).

% finite stream from list
list_(Xs,list_stream(state(Xs))).

list_stream(State,X):-  
  arg(1,State,[X|Xs]),
  nb_linkarg(1,State,Xs).

eng_(X,G,engine_next(E)):-engine_create(X,G,E).  


% finite integer range
range_(From,To,range_stream(state(From,To))).

range_stream(State,X):-
  State=state(X,To),
  X<To,
  succ(X,SX),
  nb_linkarg(1,State,SX).

% initial segment of length K stream
take(K,E,taker(state(K,E))).

taker(State,X):-
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
map_(F,E,map_stream(F,E)).

map_stream(F,E,Y):-ask_(E,X),call(F,X,Y).

map_(F,E1,E2,map_stream(F,E1,E2)).

map_stream(F,E1,E2,Z):-ask_(E1,X),ask_(E2,Y),call(F,X,Y,Z).

reduce_(F,InitVal,E,reduce_stream(state(InitVal),F,E)).

reduce_stream(S,F,E,R):-
  \+ is_done(E),
  do((
    Y in E,
    arg(1,S,X),
    call(F,X,Y,Z),
    nb_linkarg(1,S,Z)
  )),
  arg(1,S,R).


% sequence sum and product operrations  

% interleaved sum of two finite or infinite streams
sum_(E1,E2,sum_stream(state(E1,E2))).

sum_stream(State,X):-
  State=state(E1,E2),
  ask_(E1,X),
  !,
  nb_linkarg(1,State,E2),
  nb_linkarg(2,State,E1).
sum_stream(state(_,E2),X):-
  ask_(E2,X).

% cartesian product of two finite or infinite streams
prod_(E1,E2,prod_stream(state(0,E1,E2,A1,A2))):-
  new_array(A1),
  new_array(A2).

prod_stream(S,X-Y):-
  S=state(_,E1,E2,A1,A2),
  repeat,
    ( is_done(E1),is_done(E2) -> !,fail
    ; nat_pair_stream(S,I-J),
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

% bactrack over G for its side-effects only  
do(G):-G,fail;true.

% stream of naturla number pairs
nat_pair_(nat_pair_stream(state(0))).

nat_pair_stream(S,A-B):-nat_stream(S,X),cantor_unpair(X,A,B).


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

  
f1(X):-nat_(N),list_([10,20,30],M),map_(plus,N,M,R),X in R.
 
f2(X):-nat_(N),nat_(M),map_(plus,N,M,R),X in R.  

f3(X):-range_(1,5,E),reduce_(plus,0,E,R),X in R.

t1(X):-pos_(N),neg_(M),sum_(M,N,S),X in S. 

t2(X):-nat_(N),list_([a,b,c],M),sum_(N,M,S),X in S.

t3(X):-range_(1,3,N),list_([a,b,c,d,e],M),sum_(M,N,S),X in S.
  
t4(X):-nat_(N),slice(4,8,N,S),X in S.

t5:-neg_(A),pos_(B),prod_(A,B,P),
   take(30,P,T),forall(X in T,writeln(X)).

t6:-nat_(A),list_([a,b,c],B),prod_(A,B,P),
    take(30,P,T),forall(X in T,writeln(X)).

t7:-range_(0,5,A),list_([a,b,c],B),prod_(A,B,P),
    take(30,P,T),forall(X in T,writeln(X)).
    
t8:-nat_(A),list_([a,b,c],B),
  prod_(B,A,P),take(30,P,T),
  forall(X in T,writeln(X)).
  
t9(X):-const_(10,C),nat_(N),map_(plus,C,N,R),X in R.

t10(X):-const_(10,C),nat_(N),prod_(C,N,P),X in P.


odds(Xs) :-
  lazy_findall(X, (between(0, infinite, X0),X is 2*X0+1), Xs).

t11(X):-odds(Xs),list_(Xs,L),nat_(N),prod_(L,N,P),X in P.
  
e1(X):-eng_(X,member(X,[1,2,3]),E),list_([a,b],L),sum_(E,L,S),X in S.

e2(X):-eng_(X,member(X,[1,2,3]),E),list_([a,b],L),prod_(E,L,S),X in S.

e3:-eng_(X,member(X,[1,2,3]),S),(X in S,writeln(X),fail;is_done(S),writeln(S)).

e4:-(X^member(X,[1,2,3])*[a,b,c,d])=E,do((X in_ E,writeln(X))).


