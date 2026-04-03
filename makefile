CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
LDFLAGS  := -framework OpenGL -framework OpenCL
INCLUDES := $(shell sdl2-config --cflags) -I/opt/homebrew/include
LIBS     := $(shell sdl2-config --libs)
TARGET   := atom
ARGS     := 0 0 0
SRCS     := main.cpp
 
.PHONY: all clean run
 
all: $(TARGET)
 
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $(TARGET) $(LDFLAGS) $(LIBS)
 
run: all
	./$(TARGET) $(ARGS)

clean:
	rm -f $(TARGET) output.ppm