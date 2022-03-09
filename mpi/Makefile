CXXFLAGS := -O3

.PHONY: clean

all: mpi_sync mpi_async

trap_p2p: trap_p2p.cpp
	mpicxx $^ ${CXXFLAGS} -o $@

trap_col: trap_col.cpp
	mpicxx $^ ${CXXFLAGS} -o $@

clean:
	rm trap_p2p trap_col
