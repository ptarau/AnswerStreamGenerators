ask(E,_):-is_done(E),!,fail.
ask(E,R):-call(E,X),!,R=X.
ask(E,_):-stop(E),fail.

:-op(800,xfx,(in)).

X in Gen:-ask(Gen,A),select_from(Gen,A,X).

select_from(_,X,X).
select_from(Gen,_,X):-X in Gen.

rand(random()).

gen_next(F,State,X):-
  arg(1,State,X),
  call(F,X,Y),
  nb_setarg(1,State,Y).

nat(nat_next(state(0))).

nat_next(S,X):-gen_next(succ,S,X).

gen_nextval(Advancer,State,Yield):-
  arg(1,State,X1),
  call(Advancer,X1,X2, Yield),
  nb_setarg(1,State,X2).

list(Xs, gen_nextval(list_step,state(Xs))).

list_step([X|Xs],Xs,X).

eng(X,Goal,engine_next(Engine)):-engine_create(X,Goal,Engine). 

and_nat_stream(Gen):-eng(_,nat_goal(0),Gen).

nat_goal(N):-
  SN is N+1,
  engine_yield(N),
  nat_goal(SN).

or_nat_stream(Gen):-eng(N, between(0,infinite,N), Gen).


%! gen2lazy(+Generator,-LazyLIst)  
%
% Turns a generator into a lazy list  
gen2lazy(E,Ls):-lazy_list(gen2lazy_forward,E,Ls).

% E manages its state, so we just pass it on
gen2lazy_forward(E,E,X):-ask(E,X).

%! lazy2gen(+LazyList, -Generator)
%
% Turns a lazy list into a generator.
% Note that list/2 actually just  works on lazy lists!
lazy2gen(Xs,E):-list(Xs,E).


% iso functors

%! iso_fun(+Operation,+SourceType,+TargetType,+Arg1, -ResultOfSourceType)
%
% Transports a predicate of arity 2 F(+A,-B) to a domain where
% an operation can be performed and brings back the result.
iso_fun(F,From,To,A,B):-
  call(From,A,X),
  call(F,X,Y),
  call(To,Y,B).

%! iso_fun(+Operation,+SourceType,+TargetType,+Arg1,+Arg2, -ResultOfSourceType)
%
% Transports a predicate of arity 2 F(+A,+B,-C) to a domain where
% an operation can be performed and brings back the result. 
% transports F(+A,+B,-C) 
iso_fun(F,From,To,A,B,C):- % writeln(iso_fun(F,From,To,A,B,C)),
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

%TODO

prod(E1,E2,engine(E,X,G)):-
  G=prod_goal(E1,E2),
  engine_create(X,G,E).

prod_goal(E1,E2):-
  ask_generator(E1,A),
  prod_loop(1,A,E1-[],E2-[]).

prod_loop(Ord1,A,E1-Xs,E2-Ys):-
  flip(Ord1,Ord2,A,Y,Pair),
  forall(member(Y,Ys),generate_answer(Pair)),
  ask_generator(E2,B),
  !,
  cart_prod_loop(Ord2,B,E2-Ys,E1-[A|Xs]).
prod_loop(Ord1,_A,E1-_Xs,_E2-Ys):-
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

%! eval_stream(+GeneratorExpression, -Generator)
% evaluates a generator expressioin to ready to use
% generator that combines their effects
eval_stream(E+F,S):- !,eval_stream(E,EE),eval_stream(F,EF),sum(EE,EF,S).
eval_stream(E*F,P):- !,eval_stream(E,EE),eval_stream(F,EF),prod(EE,EF,P).
eval_stream(E:F,R):- !,range(E,F,R).
eval_stream([X|Xs],L):-!,list([X|Xs],L).
eval_stream({E},SetGen):-!,eval_stream(E,F),setify(F,SetGen).
eval_stream(X^G,E):-!,eng(X,G,E).
eval_stream(A,C):-atomic(A),!,const(A,C).
eval_stream(E,E).

:-op(800,xfx,(in_)).
X in_ E:-eeval(E,EE),X in EE.     


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

