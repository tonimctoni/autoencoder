all:
	g++ test06.cpp -Wall -Wextra -std=c++0x -O3 -funroll-loops -ftree-loop-distribution -march=native
run: all
	./a.out