CC        = gcc
CFLAGS   += -lm -lgsl -lgslcblas -O4

all: est-sfs 
       
est-sfs : est-sfs.c routine-library.c
	$(CC) -o $@ $^ $(CFLAGS)

