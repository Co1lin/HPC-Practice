CXXFLAGS := -O3

.PHONY: clean

all: trap_p2p trap_col naive_reduce mpi_reduce

trap_p2p: trap_p2p.cpp
	mpicxx $^ ${CXXFLAGS} -o $@

trap_col: trap_col.cpp
	mpicxx $^ ${CXXFLAGS} -o $@

naive_reduce: naive_reduce.cpp
	mpicxx $^ ${CXXFLAGS} -fopenmp -o $@

mpi_reduce: mpi_reduce.cpp
	mpicxx $^ ${CXXFLAGS} -fopenmp -o $@

clean:
	rm trap_p2p trap_col naive_reduce mpi_reduce
