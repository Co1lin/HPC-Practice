CXXFLAGS := -O3
CC := g++ -fopenmp -march=broadwell -Wall

.PHONY: clean

all: mem

mem: mem.cpp
	${CC} $^ ${CXXFLAGS} -o $@

clean:
	rm mem
