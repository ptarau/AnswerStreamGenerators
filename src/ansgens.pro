new_generator(X,G,engine(E,X,G)):-engine_create(X,G,E).

clone_generator(engine(_,X,G),engine(E,X,G)):-engine_create(X,G,E).

stop_generator(E,R):-is_done_generator(E),!,R=E.
stop_generator(engine(E,X,G),engine(done,X,G)):-engine_destroy(E).

stop_generator(E):-stop_generator(E,_).

stop_generator:-engine_self(E),engine_destroy(E).

ask_generator(engine(done,_,_),_):-!,fail.
ask_generator(engine(E,_,_),X):-engine_next(E,A),!,X=A.
ask_generator(Estate,_):-nb_setarg(1,Estate,done),fail.  

generate_answer(X):-engine_yield(X).

is_done_generator(engine(done,_,_)).

% shows engine handle or "done" as well as the generator's stored goal
show_generator(engine(E,X,G)):-writeln(X:G=E).

nat(E):-new_generator(_,nat_goal(0),E).

nat_goal(N):-
  SN is N+1,
  generate_answer(N),
  nat_goal(SN).

nat_(E):-new_generator(N, between(0,inf,N), E).

:-op(800,xfx,(in)).

X in E:-ask_generator(E,A),select_from(E,A,X).

select_from(_,A,A).
select_from(E,_,X):-X in E.

range(From,To,E):-Max is To-1,new_generator(I,between(From,Max,I),E).

range(To,E):-range(0,To,E).

list2generator(Xs,engine(E,X,G)):-G=member(X,Xs),engine_create(X,G,E).

finite_generator2list(E,Xs):-findall(X,X in E,Xs).

generator2list(E,Xs):-lazy_findall(X,X in E,Xs).

lazy_nats(Xs):-lazy_findall(X,between(0,infinite,X),Xs).

generator2list_(engine(E,_,_),Xs):-lazy_list(lazy_engine_next(E, 1), Xs).

lend_operation_to_lazy_lists(Op,Xs,Ys,Zs):-
  list2generator(Xs,E1),
  list2generator(Ys,E2),
  call(Op,E1,E2,E3),
  generator2list(E3,Zs).

lazy_list_sum(Xs,Ys,Zs):-lend_operation_to_lazy_lists(dir_sum,Xs,Ys,Zs).

lazy_list_prod(Xs,Ys,Zs):-lend_operation_to_lazy_lists(cart_prod,Xs,Ys,Zs).

fin_dir_sum(E1,E2,E):-new_generator(R, (R in E1 ; R in E2), E).

fin_cart_prod(E1,E2,E):-new_generator(R, fin_cart_prod_goal(E1,E2,R), E).

fin_cart_prod_goal(E1,E2,X-Y):-
  X in E1,
  clone_generator(E2,Clone),
  Y in Clone.  

dir_sum(E1,E2,engine(E,X,G)):-
  G=dir_sum_goal(E1,E2,X),
  engine_create(X,G,E).
  
dir_sum_goal(E1,E2,X):-
  ( ask_generator(E1,X)
  ; ask_generator(E2,X)
  ; \+ (is_done_generator(E1),is_done_generator(E2)),
    dir_sum_goal(E1,E2,X)
  ).  

cart_prod(E1,E2,engine(E,X,G)):-
  G=cart_prod_goal(E1,E2),
  engine_create(X,G,E).

cart_prod_goal(E1,E2):-
  ask_generator(E1,A),
  cart_prod_loop(1,A,E1-[],E2-[]).

cart_prod_loop(Ord1,A,E1-Xs,E2-Ys):-
  flip(Ord1,Ord2,A,Y,Pair),
  forall(member(Y,Ys),generate_answer(Pair)),
  ask_generator(E2,B),
  !,
  cart_prod_loop(Ord2,B,E2-Ys,E1-[A|Xs]).
cart_prod_loop(Ord1,_A,E1-_Xs,_E2-Ys):-
  flip(Ord1,_Ord2,X,Y,Pair),
  X in E1,member(Y,Ys),
  generate_answer(Pair),
  fail.

flip(1,2,X,Y,X-Y).
flip(2,1,X,Y,Y-X).


conv(E1,E2,E):-
  new_generator(_,conv_loop(E1,E2,0,[],[]),E).

conv_loop(E1,E2,N,Xs,Ys):-
  succ(N,SN),
  ask_generator(E1,X),XXs=[X|Xs],
  ask_generator(E2,Y),YYs=[Y|Ys],
  ( nth0(I,XXs,A),
    K is N-I,
    nth0(K,YYs,B),
    generate_answer(A-B),
    fail
  ; true  
  ),
  conv_loop(E1,E2,SN,XXs,YYs).

setify(E,SE):-new_generator(X,distinct(X,X in E),SE).

:-op(100,fx,(!)).

eeval(engine(E,X,G),engine(E,X,G)).
eeval(!E,ClonedE):-clone_generator(E,ClonedE).
eeval(E+F,S):-eeval(E,EE),eeval(F,EF),dir_sum(EE,EF,S).
eeval(E*F,P):-eeval(E,EE),eeval(F,EF),cart_prod(EE,EF,P).
eeval({E},SetGen):-eeval(E,F),setify(F,SetGen).
eeval([X|Xs],E):-list2generator([X|Xs],E).

:-op(800,xfx,(in_)).
X in_ E:-eeval(E,EE),X in EE.     



deep_clone(engine(_,X,G),engine(CE,X,G)):-engine_create(X,G,CE).
deep_clone(E+F,CE+CF):-deep_clone(E,CE),deep_clone(F,CF).
deep_clone(E*F,CE*CF):-deep_clone(E,CE),deep_clone(F,CF).
deep_clone(!E,CE):-deep_clone(E,CE).
deep_clone([X|Xs],[X|Xs]).

map_generator(F,E,NewE):-new_generator(Y,map_goal(F,E,Y),NewE).

map_goal(F,E,Y):-X in E,call(F,X,Y).

map_generator(F,E1,E2,NewE):-new_generator(_,map_goal2(F,E1,E2),NewE).

map_goal2(F,E1,E2):-
  ( ask_generator(E1,X1)->Ok1=true;Ok1=fail),
  ( ask_generator(E2,X2)->Ok2=true;Ok2=fail),
  ( Ok1,Ok2->call(F,X1,X2,R),
    generate_answer(R),
    map_goal2(F,E1,E2)
  ; \+Ok1,Ok2->stop_generator(E2),fail
  ; Ok1,\+Ok2->stop_generator(E1),fail
  ).

zipper_of(E1,E2,E):-map_generator(zip2,E1,E2,E).

zip2(X,Y,X-Y).

reduce_with(Op,E,R):-
  ask_generator(E,First),
  Res=result(First),
  ( Y in E,
    arg(1,Res,X),
    call(Op,X,Y,R),
    nb_setarg(1,Res,R),
    fail
  ; arg(1,Res,R)
  ).

reducer(Op,E,NewE):-new_generator(R,reduce_with(Op,E,R),NewE).

slice(E,From,To,NewE):-new_generator(X,in_slice_of(E,From,To,X),NewE).

in_slice_of(E,From,To, X):-
  From>=0,From<To,
  Ctr=c(0),
  X in E,
  arg(1,Ctr,K),K1 is K+1,nb_setarg(1,Ctr,K1),
  (
    K<From->fail
  ; K>=To->stop_generator(E),!,fail
  ; true
  ).
  
take(K,E,NewE):-slice(E,0,K,NewE).

drop(K,E,NewE):-slice(E,K,inf,NewE).

pythagoras(Triplets):-
  nat(M),
  nat(N),
  cart_prod(M,N,Pairs),
  map_generator(mn2xyz,Pairs,Triplets).

mn2xyz(M-N,X^2+Y^2=:=Z^2):-N>0,M>N,
  X is M^2-N^2,
  Y is 2*M*N,
  Z is M^2+N^2.

pythagorean_slice:-
  pythagoras(P),
  slice(P,100001,100005,S),
  forall(R in S,(assertion(R),writeln(R))).

t1:-
  range(0,3,E1),
  %range(10,14,E2),
  list2generator([a,b,c],E2),
  range(100,102,E3),
  eeval(E1+E3+E2+!E3, E),
  forall(X in E,writeln(X)).
    
t2:-
  range(0,2,E1),
  list2generator([a,b,c],E2),
  eeval(E1*E2, E),
  forall(X in E,writeln(X)).

  
t3:-
  range(0,3,E1),
  %range(10,14,E2),
  list2generator([a,b,c],E2),
  range(100,102,E3),
  eeval(E1*E3+E2*!E3, E),
  forall(X in E,writeln(X)).

t4:- 
  range(3,E),
  eeval((!E * !E)*(!E + !E)*E,NewE),
  forall(X in NewE,ppp(X)). 
  
t5:- 
 eeval([the]*[cat,dog,robot]*[walks,runs],E),eeval(!E,EE),
  stop_generator(E,_),
  forall(X in EE,ppp(X)).

t6:-nat(E),take(4,E,F),forall(X in F,ppp(X)).

t7:-range(20,E),drop(15,E,F),forall(X in F,writeln(X)).

t8:-range(5,E),map_generator(succ,E,NewE),forall(X in NewE,writeln(X)).


t9:-range(0,10,E1),range(100,110,E2),
    dir_sum(E1,E2,E),
    forall(R in E,writeln(R)).
    
t10:-range(0,10,E1),range(100,110,E2),
    map_generator(plus,E1,E2,E),
    forall(R in E,writeln(R)).

t11:-range(0,10,E1),
    list2generator([a,b,c,d],E2),
    zipper_of(E1,E2,E),
    forall(R in E,writeln(R)).


t12:-range(10,E),reduce_with(plus,E,Sum),writeln(Sum).

c1:-
  nat(N),nat(M),conv(M,N,C),ppg(30,C).

c2:-
  nat(N),list2generator([a,b,c,d],M),conv(M,N,C),ppg(30,C).

c3:-
  nat(N),list2generator([a,b,c,d],M),conv(N,M,C),ppg(30,C).

c4:-
  range(5,N),range(5,M),conv(M,N,C),ppg(30,C).

c5:-
  range(4,N),list2generator([a,b,c,d],M),conv(M,N,C),ppg(30,C).
  
      
go:-
   member(G,[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12]),
   ( nl,writeln(G),G,fail
   ; current_engine(E),ppp(E),fail 
     % tests that all stays clean
   ).

cgo:-member(G,[c1,c2,c3,c4,c5]),
   ( nl,writeln(G),G,fail
   ; current_engine(E),ppp(E),fail 
     % tests that all stays clean
   ).
      
% helpers
ppp(X):-portray_clause(X).

% adaptor for lazy_list syntax

% finite printer for generators
ppg(K,E):-take(K,E,F),forall(X in F,ppp(X)).

% finite printer for lazy lists
ppl(K,Xs):-findnsols(K,X,member(X,Xs),Rs),!,ppp(Rs).

% positive integers
pos(Xs) :-lazy_findall(X, between(1, infinite, X), Xs).

% negative integers
neg(Xs) :-lazy_findall(X, (between(1, infinite, X0), X is -X0), Xs).
  
lazy_take(K,Xs,Ys):-
   list2generator(Xs,E1),
   take(K,E1,E2),
   generator2list(E2,Ys).

% etc. 
% this can be generalized with higher order mapping predicates
% like one would do in  Haskell
   
l1:-pos(Ns),neg(Ms),lazy_list_sum(Ns,Ms,Xs),ppl(20,Xs).
l2:-pos(Ns),neg(Ms),lazy_list_prod(Ns,Ms,Xs),ppl(20,Xs).  
  

