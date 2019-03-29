# direct sum and convolution of generators in Python

def dir_sum(g1,g2) :
  ok=True
  while(ok) :
    x=next(g1,None)
    y=next(g2,None)
    if x!=None : yield x
    if y!=None : yield y
    ok=x or y
 
# convolution, also ok for finite sequences    
def conv(g1,g2) :
  n=0
  xs=[]
  ys=[]
  while(True) :
    x=next(g1,None)
    y=next(g2,None)
    ok1=x!=None
    ok2=y!=None
    if not (ok1 or ok2) : break
    if ok1 : xs.append(x)
    if ok2 : ys.append(y)
    l1 = len(xs)-1
    l2 = len(ys)-1 
    for i in range(n+1) :
      k=n-i
      yield(xs[min(l1,i)],ys[min(l2,k)])
    n+=1

    
def pos() :
  n=1
  while(True) :
    yield n
    n+=1
    
def neg() :
  n= -1
  while(True) :
    yield n
    n-=1

def take(k,g) :
  i=0
  while(i<k) :
    x=next(g,None)
    if x==None : break
    yield x
    i+=1
 
# tests

def s1() :
  a=iter(range(6))
  b=iter(['a','b','c'])
  c=dir_sum(a,b)
  for p in c:
    print(p)
    
def s2():
  a=iter(range(6))
  b=iter(['a','b','c'])
  c=dir_sum(b,a)
  for p in c:
     print(p)

def s3():
  a=iter(range(6))
  b=iter(['a','b','c'])
  c=dir_sum(b,a)
  for p in c:
    print(p)
    
    
    
def c1() :
  g=conv(pos(),neg())
  xs=take(20,g)
  print(list(xs))
    
def c2() :
  g=conv(neg(),iter(range(1,5)))
  xs=take(50,g)
  print(list(xs))    
  
def c3() :
  g=conv(iter(range(1,5)),neg())
  xs=take(50,g)
  print(list(xs))      
    
def c4() :
  g=conv(iter(range(1,5)),iter("abcd"))
  xs=take(50,g)
  print(list(xs))   
  
    
    
    
    
    
    
    
    