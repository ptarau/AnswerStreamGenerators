:-module(dynamic_arrays,[
  new_array/2,
  new_array/1,
  push_to/2,
  pop_from/2,
  array_get/3,
  array_set/3,
  array_length/2,
  array_size/2
]).

/** 
* Dynamic stateful arrays in Prolog - extend and shrink as needed.
* Updates persist on backtracking.
*/


%! new_array(+InitSize, -Array)
% 
% for use as a stack, initially filled with unbound variables
% and  Top=0
% 
new_array(InitSize,'a'(0,A)):-InitSize>0,functor(A,'x',InitSize).


%! new_full_array(+InitSize, -Array)
% 
% for use as an array, initialized with variables marking empty slots
new_full_array(InitSize,'a'(Top,A)):-InitSize>0,Top is InitSize-1,functor(A,'x',InitSize).


%! new_array(-Array)
% 
% for use as a stack, initially filled with unbound variables
% and  Top=0
% 
% small array of default size 1
new_array(A):-new_array(1,A).

%! push_to(+Array, +Element)
%
% pushes Elelment ot to the array seen as a stack
push_to(A,X):-push_to_if_it_fits(A,X),!.
push_to(A,X):-resize_up(A),push_to_if_it_fits(A,X).

%! pop_from(+Array, -Element)
% pops last element pushed to the array
pop_from(A,X):-pop_from_and_maybe_trim(A,X).

%! array_get(+Array, +Index, -Value)
%
% gets value at given index of the array, starting at 0
% fails if out of range
array_get(A,I,X):-integer(I),!,
  arg(1,A,Top),I=<Top,succ(I,I1),arg(2,A,T),arg(I1,T,X).
array_get(A,I,X):- % var I, for use as an interator, via nondet arg/3
  arg(1,A,Top),arg(2,A,T),arg(I1,T,X),I<Top,succ(I,I1). 

%! array_set(+Array, +Index, +Value)
%
% sets value at given index of the array, starting at 0
% fails if out of range
% assumes array grown big enough already for this - assigs destructively
% works also with stack uses - for indices below Top
array_set(A,I,X):-integer(I),
  arg(1,A,Top),I=<Top,succ(I,I1),arg(2,A,T),nb_setarg(I1,T,X).

%! array_length(+Array, -Length)
% size of the container array, not number of elements
array_length(A,Size):-arg(2,A,T),functor(T,_,Size).


%! array_size(+Array, -Size)
%
% number of elements, how mamy elements it has up to Top
array_size(A,Top):-arg(1,A,Top).

% implementation

push_to_if_it_fits(A,X):-
   A=a(Top,T),functor(T,_,N),
   Top<N,
   NewTop is Top+1,
   nb_setarg(1,A,NewTop),
   nb_setarg(NewTop,T,X).
   
pop_from_and_maybe_trim(A,X):-
   A=a(Top,T),Top>0,
   NewTop is Top-1,
   nb_setarg(1,A,NewTop),
   arg(Top,T,X),
   nb_setarg(Top,T,_),
   (resize_down(A)->true;true).
      
copy_args(0,_,_):-!.
copy_args(I,T,NewT):-I>0,I1 is I-1,arg(I,T,X),arg(I,NewT,X),copy_args(I1,T,NewT).

resize_up(A):-arg(2,A,T),functor(T,_,N),
  N2 is N<<1, % expand
  new_array(N2,B),arg(2,B,NewT),
  copy_args(N,T,NewT),
  nb_setarg(2,A,NewT).

resize_down(A):-A=a(Top,T),functor(T,_,N),
  Top<<2=<N,N2 is N>>1, % shrink
  new_array(N2,B),arg(2,B,NewT),
  copy_args(N2,T,NewT),
  nb_setarg(2,A,NewT).
  
  