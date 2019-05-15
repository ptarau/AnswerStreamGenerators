/*
:-module(test_lazy_streams,[
  tests/0,
  bm/0,
  bmx,
  bmp
  ]
).
*/

%:-use_module('lazy_streams.pl').

%c:-make.

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
  
  
t12:-neg(A),pos(B),prod_(A,B,P),
   take(30,P,T),show(30,T),stop(P).

t13:-nat(A),list([a,b,c],B),prod_(A,B,P),show(20,P),stop(P).

t14:-range(0,5,A),list([a,b,c],B),prod_(A,B,P),
     take(20,P,T),show(30,T),stop(P).
    
t15:-nat(A),list([a,b,c],B),
  prod_(B,A,P),take(20,P,T),
  show(30,T),
  stop(P).
  
    
t16:-const(10,C),nat(N),map(plus,C,N,R),show(R).

t17:-const(10,C),nat(N),prod(C,N,P),show(P).


t18:-eng(_X,fail,E),list([a,b],L),sum(E,L,S),show(S).
  
t19:-eng(X,member(X,[1,2,3]),E),list([a,b],L),sum(E,L,S),show(S).

t20:-eng(X,member(X,[1,2,3]),E),list([a,b],L),prod(E,L,S),show(S).

t21:-eng(X,member(X,[1,2,3]),S),(X in S,writeln(X),fail;is_done(S),writeln(S)).

t22:-(X^member(X,[1,2,3])*[a,b])=E,do((X in_ E,writeln(X))).

t23:-range(1,5,R),cycle(R,C),show(20,C).

t24:-range(1,4,R),cycle(R,C),list([a,b,c,d,e,f],L),zipper_of(C,L,Z),show(Z).

t25:-eng(X,member(X,[a,b,c]),G),range(1,6,R),prod(G,R,P),show(P).

t26:-ceng(X,member(X,[a,b,c]),G),ceng_clone(G,CG),prod(G,CG,P),show(P).

t27:-ceng(X,member(X,[a,b,c]),G),cycle(G,C),show(C).


t28:-range(0,10,A),range(100,110,B),arith_sum(A,B,S),show(S).

t29:-fact(5,S),show(S).

t30:-nat(N),chains([succ,succ],N,N2),show(N2).

t31:-fibo(E),show(E).

t32:-
  clause_stream(chains(_,_,_),C),
  do((X in C,portray_clause(X))).

t33:-pos(E),chains([succ,pred],E,R),show(R).

t34:-pos(E),mplex([succ,pred],E,R),show(R).

t35:-lazy_nats(Ls),list(Ls,E),show(E).


t36:-range(1,10,N),iso_fun(lazy_maplist(succ),gen2lazy,lazy2gen,N,M),show(M).

% while maplist loops, this iso functor based map does not
t37:-lazy_nats(Ns),
  lazy_maplist(succ,Ns,Ms),
  once(findnsols(10,I,member(I,Ms),Rs)),
  writeln(Rs).

% results in reusability of a lazy list 
t38:-lazy_nats(Ns),
  lazy_maplist(plus,Ns,Ns,Ms),
  once(findnsols(10,I,member(I,Ms),Rs)),
  writeln(Rs).

t39:-nat(E),
     gen2lazy(E,Xs),
     gen2lazy(E,Ys),
     lazy2gen(Ys,B),
     lazy2gen(Xs,A),
     show(A),show(B),show(E).

t40:-nat(E),split(E,E1,E2),show(E1),show(E2),show(E).

t41:-nat(E),split(E,E1,E2),ask(E,A),ask(E1,X),ask(E2,Y),writeln(A+X+Y),show(E),stop(E).

t42:-list([a,b,c],L),nat(N),cat([L,N],R),show(R).

odds(Xs) :-lazy_findall(X, (between(0, infinite, X0),X is 2*X0+1), Xs).

% lazy_findall leaves undead engine
t43:-odds(Xs),list(Xs,L),nat(N),prod(L,N,P),show(P).

t44:-
  lazy_nats(As),lazy_nats(Bs),lazy_conv(As,Bs,Ps),
  findall(P,(
    between(1,20,I),
    nth1(I,Ps,P)
  ),
  Qs),
  writeln(Qs).
  
t45:-neg(As),pos(Bs),
     convolution(As,Bs,Ps),show(Ps),
     stop(As),stop(Bs),stop(Ps).

t46:- do((X in_ 10:20,writeln(X))).

t47:-do((X in_ X^member(X,[a,b,c]),writeln(X))).
  
t48:-nat(A),nat(B),sum(A,B,C),setify(C,D),show(D).

t49:-eval_stream({([a,b,a])}+(1:3)*c,E),show(30,E).

t50:-term_reader('lazy_streams.pl',R),drop(9,R,S),show(2,S),stop(S).

t51:- and_nat_stream(N),show(N),stop(N).

t52:- or_nat_stream(N),show(N),stop(N).

t53:-prime(E),show(E),stop(E).

run_tests:-
  member(T,[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,
  t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,
  t31,t32,t33,t34,t35,t36,t37,t38,t39,
  t40,t41,t42,t43,t44,t45,t46,t47,t48,t49,t50,t51,t52,t53]),
  nl,
  listing(T),
  call(T),
  fail
; %bm,
  true.
 
 
time(G,T):-get_time(T1),once(G),get_time(T2),T is T2-T1. 
 
:-op(888,fx,(ll)).
 
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

%! bm
%
% benchmarking some stream operations

bm:-bm(21).  

%! tests
% runs all tests, listing their code as examples of use cases
%
% after taping 
% ?-tests.
% results are written out to file tests.txt
tests:-
  tell('tests.txt'),
  run_tests,
  do((current_engine(E),writeln(E))),
  %bm,
  told.


  

% examples from paper

and_nat_stream(E):-eng(_,nat_goal(0),E).

nat_goal(N):-
  SN is N+1,
  engine_yield(N),
  nat_goal(SN).
  
or_nat_stream(E):-eng(N, nats(0,N), E).

nats-->[].
nats-->succ,nats.

% benchmark for infinite lazy streams and lazy lists
bm_inf(N):-
 nat(Gen),
 or_nat_stream(OrEng),
 and_nat_stream(AndEng),
 lazy_nats(FastLs),
 lazy_findall(I,nats(0,I),SlowLs),
 
 % access N-the element of infinite stream or list
 time(do(nth(N,Gen,_))),
 time(do(nth(N,OrEng,_))),
 time(do(nth(N,AndEng,_))),
 time(do(nth0(N,FastLs,_))),
 time(do(nth0(N,SlowLs,_))).

% benchamrks for finite streams/ lists 
bm_last(N):-
 numlist(0,N,Ns), 
 list(Ns,Gen),
 eng(X,member(X,Ns),Eng),
 lazy_findall(X,member(X,Ns),Ls),
 succ(N,SN),test:lazy_nats_from_to(0,SN,LLs),
 range(0,N,Range),
 
 % access last element
 time(do(nth0(N,Ns,_))),
 time(do(nth(N,Gen,_))),
 time(do(nth(N,Range,_))),
 time(do(nth(N,Eng,_))),
 time(do(nth0(N,Ls,_))),
 time(do(nth0(N,LLs,_))).
 
bm_all(N):-
 numlist(0,N,Ns), 
 list(Ns,Gen),
 eng(X,member(X,Ns),Eng),
 lazy_findall(X,member(X,Ns),Ls),
 succ(N,SN),test:lazy_nats_from_to(0,SN,LLs),
 range(0,N,Range),
 % iterating over all, via backtracking
 time(do((member(_,Ns)))),
 time(do((_ in Gen))),
 time(do((_ in Range))),
 time(do((_ in Eng))),
 time(do((member(_,Ls)))),
 time(do((member(_,LLs)))).

 
%! bmx
% 
%  benchmarks comparing plain Prolog with lazy streams and lazy lists
bmx:-
  N is 2^23,
  bm_inf(N),statistics,nl,
  bm_last(N),statistics,nl,
  bm_all(N),statistics.

%! bmp
%
% benchmarking product operations
bmp:-N is 2^9,bmp(N).

bmp(N):-  
  bmp(N,prod),
  bmp(N,prod_).
  
bmp(N,Op):-
 range(1,N,L),
 range(1,N,R),
 call(Op,L,R,P),
 N2 is N*N,
 time(do(nth(N2,P,_))).
 
/*

% Run on a 128GB 18-core iMacPro, with 8 GB given to Prolog to avoid stack overflows with lazy lists.

?- bmx.
% 75,497,483 inferences, 2.999 CPU in 3.004 seconds (100% CPU, 25174563 Lips)
% 41,943,046 inferences, 8.137 CPU in 9.606 seconds (85% CPU, 5154863 Lips)
% 41,943,046 inferences, 7.579 CPU in 9.064 seconds (84% CPU, 5533910 Lips)
% 95,070,967 inferences, 11.982 CPU in 12.453 seconds (96% CPU, 7934266 Lips)
% 128,625,340 inferences, 21.734 CPU in 24.157 seconds (90% CPU, 5918102 Lips)
% Started at Wed May 15 09:19:47 2019
% 58.379 seconds cpu time for 383,867,626 inferences
% 7,453 atoms, 4,443 functors, 3,449 predicates, 69 modules, 130,610 VM-codes
% 
%                     Limit   Allocated      In use
% Local  stack:           -       20 Kb    1,824  b
% Global stack:           -    4,096 Mb    1,691 Mb
% Trail  stack:           -    2,048 Mb    3,720  b
%        Total:    8,192 Mb    6,144 Mb    1,691 Mb
% 
% 17 garbage collections gained 5,499,092,552 bytes in 13.547 seconds.
% 3 clause garbage collections gained 125 clauses in 0.000 seconds.
% Stack shifts: 2 local, 18 global, 18 trail in 0.095 seconds
% 2 threads, 0 finished threads used 0.000 seconds
% 3 engines, 2 finished engines

% 2,796,205 inferences, 0.113 CPU in 0.113 seconds (100% CPU, 24789051 Lips)
% 67,108,873 inferences, 2.908 CPU in 2.912 seconds (100% CPU, 23077131 Lips)
% 83,886,095 inferences, 3.360 CPU in 3.366 seconds (100% CPU, 24963581 Lips)
% 41,943,046 inferences, 7.779 CPU in 9.258 seconds (84% CPU, 5391779 Lips)
% 128,625,340 inferences, 14.303 CPU in 16.110 seconds (89% CPU, 8992591 Lips)
% 103,459,513 inferences, 12.810 CPU in 13.070 seconds (98% CPU, 8076431 Lips)
% Started at Wed May 15 09:19:47 2019
% 104.547 seconds cpu time for 820,076,521 inferences
% 7,456 atoms, 4,443 functors, 3,443 predicates, 69 modules, 130,610 VM-codes
% 
%                     Limit   Allocated      In use
% Local  stack:           -       24 Kb    1,824  b
% Global stack:           -    6,351 Mb    2,251 Mb
% Trail  stack:           -    1,841 Mb    3,856  b
%        Total:    8,192 Mb    8,192 Mb    2,251 Mb
% 
% 18 garbage collections gained 11,706,621,416 bytes in 20.518 seconds.
% 3 clause garbage collections gained 125 clauses in 0.000 seconds.
% Stack shifts: 3 local, 19 global, 19 trail in 0.197 seconds
% 2 threads, 0 finished threads used 0.000 seconds
% 3 engines, 4 finished engines

% 8,388,611 inferences, 0.495 CPU in 0.496 seconds (100% CPU, 16949771 Lips)
% 75,497,495 inferences, 3.248 CPU in 3.258 seconds (100% CPU, 23245090 Lips)
% 92,274,703 inferences, 3.738 CPU in 3.739 seconds (100% CPU, 24683495 Lips)
% 50,331,668 inferences, 8.057 CPU in 9.571 seconds (84% CPU, 6246626 Lips)
% 134,217,757 inferences, 14.664 CPU in 16.237 seconds (90% CPU, 9153069 Lips)
% 109,051,925 inferences, 12.945 CPU in 12.980 seconds (100% CPU, 8424321 Lips)
% Started at Wed May 15 09:19:47 2019
% 152.156 seconds cpu time for 1,298,228,497 inferences
% 7,458 atoms, 4,443 functors, 3,443 predicates, 69 modules, 130,610 VM-codes
% 
%                     Limit   Allocated      In use
% Local  stack:           -       24 Kb    1,752  b
% Global stack:           -    6,351 Mb    2,626 Mb
% Trail  stack:           -    1,841 Mb    3,856  b
%        Total:    8,192 Mb    8,192 Mb    2,626 Mb
% 
% 19 garbage collections gained 18,003,118,520 bytes in 27.139 seconds.
% 3 clause garbage collections gained 125 clauses in 0.000 seconds.
% Stack shifts: 3 local, 19 global, 19 trail in 0.197 seconds
% 2 threads, 0 finished threads used 0.000 seconds
% 3 engines, 6 finished engines
true.

*/

test:lazy_nats_from_to(N,M,Ns) :- put_attr(Ns,test,state(N,M,_)).

test:attr_unify_hook(State,Value) :-
  State=state(N,M,Read),
  N<M,
  ( var(Read) ->
      succ(N,SN),
      test:lazy_nats_from_to(SN,M,Tail),
      nb_setarg(3,State,[N|Tail]),
      arg(3,State,Value)
  ;
      Value = Read
  ).
  