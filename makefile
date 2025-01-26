SRC := src/*.cu
TARGET := app.out

CC := nvcc 
CCFLAGS := -Wall -Werror
LDLIBS := -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
COMPILE_COMMAND = $(CC) $(SRC) $(LDLIBS) -o $(TARGET)

.PHONY: all debug compilation_database clean

all: compilation_database
	$(COMPILE_COMMAND)

debug: compilation_database
	$(COMPILE_COMMAND) -g

compilation_database:
	bear -- $(COMPILE_COMMAND)

clean:
	rm $(TARGET) compile_commands.json
