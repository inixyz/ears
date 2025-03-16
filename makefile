SRC := src/*.cpp src/*.cu
TARGET := app.out

CC := nvcc 
CCFLAGS := -std=c++20
LDLIBS := -lraylib -lGL -lm -lX11 -lsndfile
COMPILE_COMMAND = bear -- $(CC) $(SRC) -o $(TARGET) $(LDLIBS) $(CCFLAGS) 

.PHONY: all debug clean

all:
	$(COMPILE_COMMAND)

debug:
	$(COMPILE_COMMAND) -g

clean:
	rm $(TARGET) compile_commands.json
