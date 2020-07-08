CXX = g++
CC = gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC
BIN = libodm-train libodm-predict
OBJ = odm.o tron.o
LIB = blas/blas.a
SO = libodm.so
OS = $(shell uname)
ifeq ($(OS), Darwin)
	SHARED_LIB_FLAG = -dynamiclib -Wl,-install_name,$(SO)
else
	SHARED_LIB_FLAG = -shared -Wl,-soname,$(SO)
endif

all: $(BIN)

$(BIN): libodm-%: %.c $(OBJ) $(LIB)
	$(CXX) $(CFLAGS) -o $@ $< $(OBJ) $(LIB)

$(OBJ): %.o: %.cpp
	$(CXX) $(CFLAGS) -o $@ -c $< 

$(LIB): blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

lib: $(OBJ) $(LIB)
	$(CXX) ${SHARED_LIB_FLAG} $(OBJ) $(LIB) -o $(SO)

clean:
	make -C blas clean
	rm -f *~ $(OBJ) $(BIN) $(SO)