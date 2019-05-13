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
 range(0,N,Range),
 
 % access last element
 time(do(nth0(N,Ns,_))),
 time(do(nth(N,Gen,_))),
 time(do(nth(N,Range,_))),
 time(do(nth(N,Eng,_))),
 time(do(nth0(N,Ls,_))).
 
bm_all(N):-
 numlist(0,N,Ns), 
 list(Ns,Gen),
 eng(X,member(X,Ns),Eng),
 lazy_findall(X,member(X,Ns),Ls),
 range(0,N,Range),
 % iterating over all, via backtracking
 time(do((member(_,Ns)))),
 time(do((_ in Gen))),
 time(do((_ in Range))),
 time(do((_ in Eng))),
 time(do((member(_,Ls)))).

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

?- lazy_streams:bmx.
% 75,497,485 inferences, 3.004 CPU in 3.008 seconds (100% CPU, 25133297 Lips)
% 41,943,046 inferences, 13.957 CPU in 16.543 seconds (84% CPU, 3005098 Lips)
% 41,943,046 inferences, 13.098 CPU in 15.602 seconds (84% CPU, 3202354 Lips)
% 120,236,795 inferences, 18.022 CPU in 18.823 seconds (96% CPU, 6671800 Lips)
% 128,625,342 inferences, 25.965 CPU in 28.634 seconds (91% CPU, 4953776 Lips)
% Started at Fri May 10 13:32:54 2019
% 82.709 seconds cpu time for 409,024,763 inferences
% 7,436 atoms, 4,428 functors, 3,471 predicates, 68 modules, 130,130 VM-codes
% 
%                     Limit   Allocated      In use
% Local  stack:           -       20 Kb    1,824  b
% Global stack:           -    2,048 Mb    2,013 Mb
% Trail  stack:           -    1,024 Mb    3,792  b
%        Total:    8,192 Mb    3,072 Mb    2,013 Mb
% 
% 20 garbage collections gained 5,842,469,976 bytes in 14.964 seconds.
% 2 clause garbage collections gained 109 clauses in 0.000 seconds.
% Stack shifts: 2 local, 17 global, 17 trail in 0.014 seconds
% 2 threads, 0 finished threads used 0.000 seconds
% 3 engines, 0 finished engines

% 2,796,205 inferences, 0.114 CPU in 0.114 seconds (100% CPU, 24576621 Lips)
% 67,108,873 inferences, 2.791 CPU in 2.793 seconds (100% CPU, 24045947 Lips)
% 83,886,096 inferences, 3.334 CPU in 3.340 seconds (100% CPU, 25163627 Lips)
% 41,943,046 inferences, 13.420 CPU in 15.971 seconds (84% CPU, 3125467 Lips)
% 128,625,340 inferences, 26.703 CPU in 30.097 seconds (89% CPU, 4816896 Lips)
% Started at Fri May 10 13:32:54 2019
% 136.603 seconds cpu time for 741,774,100 inferences
% 7,439 atoms, 4,428 functors, 3,461 predicates, 68 modules, 130,148 VM-codes
% 
%                     Limit   Allocated      In use
% Local  stack:           -       20 Kb    1,824  b
% Global stack:           -    4,096 Mb    2,181 Mb
% Trail  stack:           -    2,048 Mb    4,784  b
%        Total:    8,192 Mb    6,144 Mb    2,181 Mb
% 
% 21 garbage collections gained 9,608,162,632 bytes in 21.059 seconds.
% 2 clause garbage collections gained 109 clauses in 0.000 seconds.
% Stack shifts: 2 local, 18 global, 18 trail in 0.014 seconds
% 2 threads, 0 finished threads used 0.000 seconds
% 3 engines, 2 finished engines

% 8,388,611 inferences, 0.483 CPU in 0.484 seconds (100% CPU, 17361722 Lips)
% 75,497,495 inferences, 3.093 CPU in 3.100 seconds (100% CPU, 24410884 Lips)
% 92,274,703 inferences, 3.689 CPU in 3.696 seconds (100% CPU, 25012260 Lips)
% 50,331,669 inferences, 13.322 CPU in 15.876 seconds (84% CPU, 3778062 Lips)
% 134,217,757 inferences, 25.539 CPU in 28.115 seconds (91% CPU, 5255455 Lips)
% Started at Fri May 10 13:32:54 2019
% 189.281 seconds cpu time for 1,110,874,042 inferences
% 7,441 atoms, 4,428 functors, 3,460 predicates, 68 modules, 130,148 VM-codes
% 
%                     Limit   Allocated      In use
% Local  stack:           -       20 Kb    1,752  b
% Global stack:           -    4,096 Mb    2,276 Mb
% Trail  stack:           -    2,048 Mb    4,784  b
%        Total:    8,192 Mb    6,144 Mb    2,276 Mb
% 
% 22 garbage collections gained 13,440,569,688 bytes in 26.112 seconds.
% 2 clause garbage collections gained 109 clauses in 0.000 seconds.
% Stack shifts: 2 local, 18 global, 18 trail in 0.014 seconds
% 2 threads, 0 finished threads used 0.000 seconds
% 3 engines, 4 finished engines
true.

?- lazy_streams:bmp.
% 1,309,715 inferences, 0.423 CPU in 0.502 seconds (84% CPU, 3099866 Lips)
% 241,963,257 inferences, 8.841 CPU in 8.851 seconds (100% CPU, 27367598 Lips)

*/
