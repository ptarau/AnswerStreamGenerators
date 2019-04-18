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

% dynamic arrays in Prolog - extend and shrink as needed
%
% for use as a stack, initially fill with unbound variables, Top=0
new_array(InitSize,'a'(0,A)):-InitSize>0,functor(A,'x',InitSize).

% for use as an array, initialized with variables marking empty slots
new_full_array(InitSize,'a'(Top,A)):-InitSize>0,Top is InitSize-1,functor(A,'x',InitSize).

% small array of default size 1
new_array(A):-new_array(1,A).

% pushes to the stack
push_to(A,X):-push_to_if_it_fits(A,X),!.
push_to(A,X):-resize_up(A),push_to_if_it_fits(A,X).

% pops last e;ement pushed
pop_from(A,X):-pop_from_and_maybe_trim(A,X).

% fails if out of range
array_get(A,I,X):-integer(I),!,
  arg(1,A,Top),I=<Top,succ(I,I1),arg(2,A,T),arg(I1,T,X).
array_get(A,I,X):- % var I, for use as an interator, via nondet arg/3
  arg(1,A,Top),arg(2,A,T),arg(I1,T,X),I<Top,succ(I,I1). 

% assumes array big enough already for this - assigs destructively
array_set(A,I,X):-integer(I),
  arg(1,A,Top),I=<Top,succ(I,I1),arg(2,A,T),nb_setarg(I1,T,X).

% size of the container array, not number of elements
array_length(A,Size):-arg(2,A,T),functor(T,_,Size).

% how namy elements it has up to Top
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
  
  