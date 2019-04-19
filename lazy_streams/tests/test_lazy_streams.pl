/*
:-module(test_lazy_streams,[
  mytests/0,
  bm/0
  ]
).
*/

:-use_module('../prolog/lazy_streams.pl').

c:-make.

%! TESTS AND BENCHMARKS: 
% run with: ?-mytests. and ?-bm.

t1:-nat(N),list([10,20,30],M),map(plus,N,M,R),show(R).
 
t2:-nat(N),nat(M),map(plus,N,M,R),show(R).  

t3:-range(1,5,E),reduce(plus,0,E,R),show(R).

t4:-pos(N),neg(M),sum(M,N,S),show(S). 

t5:-nat(N),list([a,b,c],M),sum(N,M,S),show(S).

t6:-range(1,3,N),list([a,b,c,d,e],M),sum(M,N,S),show(S).
  
t7:-nat(N),slice(4,8,N,S),show(S).

t8:-neg(A),pos(B),prod(A,B,P),
   take(30,P,T),show(30,T).

t9:-nat(A),list([a,b,c],B),prod(A,B,P),
    take(20,P,T),do((X in T,writeln(X))).

t10:-range(0,5,A),list([a,b,c],B),prod(A,B,P),
     take(20,P,T),show(30,T).
    
t11:-nat(A),list([a,b,c],B),
  prod(B,A,P),take(20,P,T),
  show(30,T).
  
t12:-const(10,C),nat(N),map(plus,C,N,R),show(R).

t13:-const(10,C),nat(N),prod(C,N,P),show(P).


t14:-eng(_X,fail,E),list([a,b],L),sum(E,L,S),show(S).
  
t15:-eng(X,member(X,[1,2,3]),E),list([a,b],L),sum(E,L,S),show(S).

t16:-eng(X,member(X,[1,2,3]),E),list([a,b],L),prod(E,L,S),show(S).

t17:-eng(X,member(X,[1,2,3]),S),(X in S,writeln(X),fail;is_done(S),writeln(S)).

t18:-(X^member(X,[1,2,3])*[a,b])=E,do((X in_ E,writeln(X))).

t19:-range(1,5,R),cycle(R,C),show(20,C).

t20:-range(1,4,R),cycle(R,C),list([a,b,c,d,e,f],L),zipper_of(C,L,Z),show(Z).

t21:-eng(X,member(X,[a,b,c]),G),range(1,6,R),prod(G,R,P),show(P).

t22:-ceng(X,member(X,[a,b,c]),G),ceng_clone(G,CG),prod(G,CG,P),show(P).

t23:-ceng(X,member(X,[a,b,c]),G),cycle(G,C),show(C).


t24:-range(0,10,A),range(100,110,B),arith_sum(A,B,S),show(S).

t25:-fact(5,S),show(S).

t26:-nat(N),chains([succ,succ],N,N2),show(N2).

t27:-fibo(E),show(E).

t28:-
  clause_stream(chains(_,_,_),C),
  do((X in C,portray_clause(X))).

t29:-pos(E),chains([succ,pred],E,R),show(R).

t30:-pos(E),mplex([succ,pred],E,R),show(R).

t31:-lazy_nats(Ls),list(Ls,E),show(E).


t32:-range(1,10,N),iso_fun(lazy_maplist(succ),gen2lazy,lazy2gen,N,M),show(M).

% while maplist loops, this iso functor based map does not
t33:-lazy_nats(Ns),
  lazy_maplist(succ,Ns,Ms),
  once(findnsols(10,I,member(I,Ms),Rs)),
  writeln(Rs).

% results in reusability of a lazy list 
t34:-lazy_nats(Ns),
  lazy_maplist(plus,Ns,Ns,Ms),
  once(findnsols(10,I,member(I,Ms),Rs)),
  writeln(Rs).

t35:-nat(E),
     gen2lazy(E,Xs),
     gen2lazy(E,Ys),
     lazy2gen(Ys,B),
     lazy2gen(Xs,A),
     show(A),show(B),show(E).

t36:-nat(E),split(E,E1,E2),show(E1),show(E2),show(E).

t37:-nat(E),split(E,E1,E2),ask(E,A),ask(E1,X),ask(E2,Y),writeln(A+X+Y),show(E).

t38:-list([a,b,c],L),nat(N),cat([L,N],R),show(R).

odds(Xs) :-lazy_findall(X, (between(0, infinite, X0),X is 2*X0+1), Xs).

% lazy_findall leaves undead engine
t39:-odds(Xs),list(Xs,L),nat(N),prod(L,N,P),show(P).


run_tests:-
  member(T,[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,
  t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,
  t31,t32,t33,t34,t35,t36,t37,t38,t39]),
  nl,
  listing(T),
  call(T),
  fail
; %bm,
  true.
 
 
time(G,T):-get_time(T1),once(G),get_time(T2),T is T2-T1. 
 
ll(X) :- listing(X).

bm1(K):-
  nl,ll(bm1),
  N is 2^K,writeln(with_lazy_lists:N),
  lazy_findall(I,between(0,N,I),Is),
  lazy_maplist(succ,Is,Js),last(Js,X),writeln([X]).

bm2(K):-
  nl,ll(bm2),
  N is 2^K,N1 is N+1,
  writeln(with_engine_based_generators:N),
  eng(I,between(0,N,I),R),
  map(succ,R,SR),
  slice(N,N1,SR,S),
  show(S).
  
bm3(K):-
  nl,ll(bm3),
  N is 2^K,N1 is N+1,
  writeln(with_simple_generators:N),
  range(0,N1,R),
  map(succ,R,SR),
  slice(N,N1,SR,S),
  show(S).

bm4(K):-
  Lim is 2^K,
  pos(P),neg(N),
  prod(P,N,Prod),
  drop(Lim,Prod,More),
  show(50,More).

ppp(X):-writeln(X). 
  
bm(K):-maplist(time,[bm1(K),bm2(K),bm3(K),bm4(16)],Ts),nl,writeln(times=Ts).

bm:-bm(21).  

go:-
  tell('tests.txt'),
  run_tests,
  do((current_engine(E),writeln(E))),
  %bm,
  told.
