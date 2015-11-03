
CLFFTROOT=/home/lifxu/github/clFFT/src
AMDAPPSDKROOT=/opt/AMDAPP/AMDAPPSDK-2.9-1
AMD_LIB=${AMDAPPSDKROOT}/lib/x86_64
CC=g++

SOURCE_FILE=main.c
EXECUTABLE=main.exe

all:
	$(CC) -I$(AMDAPPSDKROOT)/include -I./ -L${CLFFTROOT}/library -L$(AMD_LIB) $(SOURCE_FILE) -o $(EXECUTABLE) -lOpenCL -lclFFT

test:
	./$(EXECUTABLE) 4 3 4 4 2 1 2 


clean:
	rm -rf *~ $(EXECUTABLE) *.linkinfo  
