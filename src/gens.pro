ask(E,_):-is_done(E),!,fail.
ask(E,R):-call(E,X),!,R=X.
ask(E,_):-stop(E),fail.

is_done(E):-arg(1,E,done).
stop(E):-nb_setarg(1,E,done).

:-op(800,xfx,(in)).

X in Gen:-ask(Gen,A),select_from(Gen,A,X).

select_from(_,X,X).
select_from(Gen,_,X):-X in Gen.

const(C,=(C)).

rand(random()).

gen_next(F,State,X):-arg(1,State,X),call(F,X,Y),nb_setarg(1,State,Y).

nat(gen_next(succ,state(0))).

gen_nextval(Advancer,State,Yield):-
  arg(1,State,X1),
  call(Advancer,X1,X2,Yield),
  nb_setarg(1,State,X2).

list(Xs, gen_nextval(list_step,state(Xs))).

list_step([X|Xs],Xs,X).

eng(X,Goal,engine_next(Engine)):-engine_create(X,Goal,Engine). 

and_nat_stream(Gen):-eng(_,nat_goal(0),Gen).

nat_goal(N):-succ(N,SN),engine_yield(N),nat_goal(SN).

or_nat_stream(Gen):-eng(N, nat_from(0,N), Gen).

nat_from(From,To):- From=To ; succ(From,Next),nat_from(Next,To).

sum(E1,E2,sum_next(state(E1,E2))).

sum_next(State,X):-State=state(E1,E2),ask(E1,X),!,
  nb_setarg(1,State,E2),
  nb_setarg(2,State,E1).
sum_next(state(_,E2),X):-ask(E2,X).

prod(E1,E2,E):-eng(_,prod_goal(E1,E2),E).

prod_goal(E1,E2):-ask(E1,A),prod_loop(1,A,E1-[],E2-[]).

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

eval_stream(E+F,S):- !,eval_stream(E,EE),eval_stream(F,EF),sum(EE,EF,S).
eval_stream(E*F,P):- !,eval_stream(E,EE),eval_stream(F,EF),prod(EE,EF,P).
eval_stream(E:F,R):- !,range(E,F,R).
eval_stream([],L):-!,list([],L).
eval_stream([X|Xs],L):-!,list([X|Xs],L).
eval_stream({E},SetGen):-!,eval_stream(E,F),setify(F,SetGen).
eval_stream(X^G,E):-!,eng(X,G,E).
eval_stream(A,C):-atomic(A),!,const(A,C).
eval_stream(E,E).

setify(E,SE):-eng(X,distinct(X,X in E),SE).

:-op(800,xfx,(in_)).

X in_ GenExpr:-eval_stream(GenExpr,NewGen),X in NewGen.     

map(F,Gen,map_next(F,Gen)).

map_next(F,Gen,Y):-ask(Gen,X),call(F,X,Y).

reduce(F,InitVal,Gen, reduce_next(state(InitVal),F,Gen)).

reduce_next(S,F,E,R):- \+ is_done(E),
  do((
    Y in E, arg(1,S,X),
    call(F,X,Y,Z),
    nb_setarg(1,S,Z)
  )),
  arg(1,S,R).

do(G):-call(G),fail;true.

scan(F,InitVal,Gen,scan_next(state(InitVal),F,Gen)).

scan_next(S,F,Gen,R) :- arg(1,S,X),
  ask(Gen,Y),
  call(F,X,Y,R),
  nb_setarg(1,S,R).

term_reader(File,next_term(Stream)):-open(File,read,Stream).

next_term(Stream,Term):-read(Stream,X),
  ( X\==end_of_file->Term=X
  ; close(Stream),fail
  ).

simple:lazy_nats(List):-simple:lazy_nats_from(0,List). 
simple:lazy_nats_from(N,List):-put_attr(List,simple,N).

simple:attr_unify_hook(N,Value):-succ(N,M),
  simple:lazy_nats_from(M,Tail),Value = [N|Tail].

lazy_nats_from(N,L) :- put_attr(L,lazy_streams,state(N,_)).

attr_unify_hook(State,Value) :-State=state(N,Read),
  ( var(Read) ->
      succ(N,M),
      nats(M,Tail),
      nb_setarg(2,State,[N|Tail]),
      arg(2,State,Value)
  ;
      Value = Read
  ).

gen2lazy(Gen,Ls):-lazy_list(gen2lazy_forward,Gen,Ls).

gen2lazy_forward(E,E,X):-ask(E,X).

lazy2gen(Xs, Gen):-list(Xs, Gen).

iso_fun(F,From,To,A,B):-call(From,A,X),call(F,X,Y),call(To,Y,B).

%
% Transports a predicate of arity 2 F(+A,+B,-C) to a domain where
% an operation can be performed and brings back the result. 
% transports F(+A,+B,-C) 
iso_fun(F,From,To,A,B,C):-
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

lazy_maplist(F,LazyXs,LazyYs):-iso_fun(map(F),lazy2gen,gen2lazy,LazyXs,LazyYs).

sum_(E1,E2, E):-iso_fun(lazy_sum,gen2lazy,lazy2gen,E1,E2, E).

lazy_sum(Xs,Ys,Zs):-lazy_list(lazy_sum_next,Xs-Ys,Zs).
  
lazy_sum_next([X|Xs]-Ys,Ys-Xs,X).
lazy_sum_next(Xs-[Y|Ys],Ys-Xs,Y).

