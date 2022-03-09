CXXFLAGS := -O3

.PHONY: clean

all: trap_p2p trap_col

trap_p2p: trap_p2p.cpp
	mpicxx $^ ${CXXFLAGS} -o $@

trap_col: trap_col.cpp
	mpicxx $^ ${CXXFLAGS} -o $@

clean:
	rm trap_p2p trap_col
