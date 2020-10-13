import multiprocessing
import time

def f(x):
   print(multiprocessing.current_process().name)
   #time.sleep(1)
   return x * x

def b():
   p = multiprocessing.Pool()
   for i in range(8):
      p.apply_async(f, args=(i,))
   p.close()
   p.join()
print("Começou o código")
start = time.time()

b()

print("Final do código")
end = time.time()
print("Tempo total:", (end - start))

