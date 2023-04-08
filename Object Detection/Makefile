CC = "g++"
PROJECT = display
SRC = objRecognition.cpp filter.cpp csv_util.cpp

LIBS = `pkg-config opencv4 --cflags --libs`

$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(PROJECT) $(LIBS)