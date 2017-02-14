SRC = $(wildcard src/*.cpp)

OUT = bin/matrixprod

# compiler
CCC = g++

# library paths
LIBS = -lpapi

# options
COMPILE_AND_LINK_COMMON_OPTIONS = -fopenmp

.SUFFIXES: .cpp

default: release

release: CCFLAGS = -O2 -Wall
release: $(OUT)

$(OUT):
	mkdir -p bin
	$(CCC) $(CCFLAGS) $(SRC) $(LIBS) $(COMPILE_AND_LINK_COMMON_OPTIONS) -o $(OUT)

clean:
	rm -f $(OBJ) $(OUT)

test:
	echo $(SRC)
	echo $(OBJ)
