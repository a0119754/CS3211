#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// arguments: number of strips (size), number of threads
int main(int argc, char* argv[])
{
        int processes;
        int processID;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &processes);
        MPI_Comm_rank(MPI_COMM_WORLD, &processID);

        int size = (argc > 1) ? atoi(argv[1]) : 1;
        int threads;

        if (argc > 2) {
                omp_set_num_threads(atoi(argv[2]));
        }

        #pragma omp parallel
        {
                threads = omp_get_num_threads();
        }
        printf("Process %d here: Reporting number of threads as %d\n", processID, threads);

        if (processID == 0) { // Master 
                printf("Master process, ready to start! Processes = %d\n", processes);
                double area = 0;
                double temp;
                int slave;
                int slaveTag;
                MPI_Status status;
                printf("Master process, started! Processes = %d\n", processes);
                #pragma omp parallel for shared(processes) private(slave, temp, slaveTag, status) reduction(+:area)
                for (slave = 1; slave < processes; slave++) {
                        printf("Master process, waiting for data from slave %d\n", slave);
                        MPI_Recv(&temp, 1, MPI_DOUBLE, slave, slaveTag, MPI_COMM_WORLD, &status);
                        printf("Master process, received temp %.6f from slave %d\n", temp, slave);
                        area += temp;
                        printf("Master process, added temp slave %d\n", temp, slave);
                }
                printf("Using %d processes (%d threads each) and %d strips, integral is %.6f\n", processes, threads, size, area);
        }
        else { // Slave
                double communicationTime;
                double computationTime;
                double before, after;

                double area = 0;
                double temp;
                int i;
                int start = (processID - 1) * (processes - 1)  / size;
                int end = (processID == processes - 1) ? (size) : (processID * (processes - 1) / size);
                printf("Slave %d out of %d processes: for-loop from %d to %d\n", processID, processes, start, end);
                before = omp_get_wtime();
                #pragma omp parallel for shared(size) private(start, end) reduction(+:area)
                for (i = start; i < end; i++) {
                        temp = ((1.0 / size) * (4.0 / (1 + ((double)i / size) * ((double)i / size))));
                        area += temp;
                        printf("Slave %d: inside for-loop at %d: temp = %.6f\n", processID, i, temp);
                }
                after = omp_get_wtime();
                computationTime = after - before;

                before = omp_get_wtime();
                MPI_Send(&area, 1, MPI_DOUBLE, 0, processID, MPI_COMM_WORLD);
                after = omp_get_wtime();
                communicationTime = after - before;

                printf("Slave %d out of %d processes: area = %.6f, computation time = %.6f, communication time = %.6f\n", processID, processes, area, computationTime, communicationTime);
        }

        MPI_Finalize();
        return 0;
}
