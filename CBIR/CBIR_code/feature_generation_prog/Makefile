CC = g++ -g -Wall
PROJECT = feature
SRC = main.cpp csv_util.cpp filter.cpp feature_generator.cpp

LIBS = `pkg-config opencv4 --cflags --libs`

$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(PROJECT) $(LIBS)