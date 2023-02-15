CC = "g++"
PROJECT = display
SRC = vidDisplay.cpp filter.cpp

LIBS = `pkg-config opencv4 --cflags --libs`

$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(PROJECT) $(LIBS)