CFLAGS=-c -fPIC -O3

libmge1d_mpfit.so : mpfit.o MGE1D.o
	$(CC) -shared MGE1D.o mpfit.o -o libmge1d_mpfit.so  -lm -L.

MGE1D.o : MGE1D.c nnls.c
	$(CC) $(CFLAGS)  MGE1D.c

mpfit.o : mpfit.c mpfit.h
	$(CC) $(CFLAGS)  mpfit.c



clean:
	rm *.o 
