SRC := src/*.cpp src/*.cu
TARGET := app.out

IMGUI_SRC = src/imgui/*.cpp
IMGUI_INC = src/imgui/

CC := nvcc 
CCFLAGS := -std=c++23 -Wall -Werror
LDLIBS := -lraylib -lGL -lm -lX11 -lsndfile
COMPILE_COMMAND = bear -- $(CC) $(SRC) -o $(TARGET) $(IMGUI_SRC) -I $(IMGUI_INC) $(LDLIBS)

.PHONY: all debug clean

all:
	$(COMPILE_COMMAND)

debug:
	$(COMPILE_COMMAND) -g

clean:
	rm $(TARGET) compile_commands.json
