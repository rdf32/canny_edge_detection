Sequential

Compile

g++ sequential_canny.cpp -o sequential_canny `pkg-config --cflags --libs opencv4`

Execute

./sequential_canny castle.png

For my implementation I used the castle.png image. For any other image you would pass its location in place of that argument.

Expected Outputs

In the project directory there is an outputs folder. The edge detection output will be stored in that outputs folder and named "sequentialcanny.png". Additionally, the execution time will be printed to the terminal.

Pthread

Compile

g++ pthreads_canny.cpp -o pthreads_canny `pkg-config --cflags --libs opencv4` -pthread

Execute

./pthreads_canny castle.png <num threads>

For my implementation I used the castle.png image. For any other image you would pass its location in place of that argument. Additionally, the second argument is the number of threads you wish to execute the program with.

Expected Outputs

In the project directory there is an outputs folder. The edge detection output will be stored in that outputs folder and named "pthreadcanny.png". Additionally, the execution time will be printed to the terminal. The work distribution across the threads should also be printed to the terminal showing the load balancing approach.

OpenMP

Compile

g++ -o openmp_canny openmp_canny.cpp `pkg-config --cflags --libs opencv4` -fopenmp

Execute

./openmp_canny castle.png <num threads>

For my implementation I used the castle.png image. For any other image you would pass its location in place of that argument. Additionally, the second argument is the number of threads you wish to execute the program with.

Expected Outputs

In the project directory there is an outputs folder. The edge detection output will be stored in that outputs folder and named "openmpcanny.png". Additionally, the execution time will be printed to the terminal.
