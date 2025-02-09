SRC := src/*.cpp src/*.cu
TARGET := app.out

CC := nvcc 
CCFLAGS := -Wall -Werror
COMPILE_COMMAND = $(CC) $(SRC) -o $(TARGET)

.PHONY: all debug compilation_database clean

all: compilation_database
	$(COMPILE_COMMAND)

debug: compilation_database
	$(COMPILE_COMMAND) -g

compilation_database:
	bear -- $(COMPILE_COMMAND)

clean:
	rm $(TARGET) compile_commands.json
