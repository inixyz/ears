SRC := src/core/*.cu src/*.cpp
PYBIND11 := $(shell python -m pybind11 --includes)
TARGET := build/ears$(shell python3-config --extension-suffix)

CC := nvcc 
CCFLAGS := -Xcompiler "-Wall, -fPIC" -shared
COMPILE_COMMAND = bear -- $(CC) $(CCFLAGS) $(SRC) $(PYBIND11) -o $(TARGET)

.PHONY: all clean

all:
	mkdir -p build
	$(COMPILE_COMMAND)

clean:
	test -d build && rm -r build || true
	test -d .cache && rm -r .cache || true
	test -f compile_commands.json && rm compile_commands.json || true
