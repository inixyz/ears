SRC := src/*.cpp src/*.cu
TARGET := app.out

CC := nvcc 
CCFLAGS := -Wall -Werror
COMPILE_COMMAND = bear -- $(CC) $(SRC) -o $(TARGET)

.PHONY: all debug clean

all:
	$(COMPILE_COMMAND)

debug:
	$(COMPILE_COMMAND) -g

clean:
	rm $(TARGET) compile_commands.json
