% tests

c:-[tests].
:-include(generators).
  
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

t22:-ceng_(X,member(X,[a,b,c]),G),ceng_clone(G,CG),prod_(G,CG,P),show(P).

t23:-ceng_(X,member(X,[a,b,c]),G),cycle_(G,C),show(C).


t24:-range_(0,10,A),range_(100,110,B),arith_sum(A,B,S),show(S).

t25:-fact(5,S),show(S).

t26:-nat_(N),chains_([succ,succ],N,N2),show(N2).

t27:-fibo_(E),show(E).

t28:-
  clause_(chains_(_,_,_),C),
  do((X in C,portray_clause(X))).

t29:-pos_(E),chains_([succ,pred],E,R),show(R).

t30:-pos_(E),mplex_([succ,pred],E,R),show(R).

t31:-lazy_nats(Ls),list_(Ls,E),show(E).


t32:-range_(1,10,N),iso_fun(maplist(succ),gen2lazy,lazy2gen,N,M),show(M).

% while maplist loops, this iso functor based map does not
t33:-lazy_nats(Ns),
  maplist_(succ,Ns,Ms),
  once(findnsols(10,I,member(I,Ms),Rs)),
  writeln(Rs).

odds(Xs) :-lazy_findall(X, (between(0, infinite, X0),X is 2*X0+1), Xs).

% lazy_findall leaves undead engine
t34:-odds(Xs),list_(Xs,L),nat_(N),prod_(L,N,P),show(P).

tests:-
  tell('tests.txt'),
  do((between(1,34,I),atom_concat(t,I,T),listing(T),call(T),nl)),
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

bm4(K):-
  Lim is 2^K,
  pos_(P),neg_(N),
  prod_(P,N,Prod),
  drop(Lim,Prod,More),
  show(50,More).

 
  
bm(K):-maplist(time,[bm1(K),bm2(K),bm3(K)],Ts),nl,writeln(times=Ts).

bm:-bm(21).  
  
ppp(X):-writeln(X).
