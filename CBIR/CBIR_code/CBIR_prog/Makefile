CC = g++ -g -Wall
PROJECT = feature
SRC = CBIR.cpp csv_util.cpp distanceMetric.cpp filter.cpp feature_generator.cpp

LIBS = `pkg-config opencv4 --cflags --libs`

$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(PROJECT) $(LIBS)