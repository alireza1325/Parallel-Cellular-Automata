#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <sstream>
#include <string>
#include <mpi.h>

#define CELL_START_TAG = 10;
#define CELL_END_TAG = 10;

void parallelRange(int globalStart, int globalStop, int irank, int nproc, int& localStart, int& localStop, int& localCount) {
    int nrows = globalStop - globalStart;
    int divisor = nrows / nproc;
    int remainder = nrows % nproc;
    int offset;
    if (irank < remainder) offset = irank;
    else offset = remainder;

    localStart = irank * divisor + globalStart + offset;
    localStop = localStart + divisor - 1;
    if (remainder > irank) localStop += 1;
    localCount = localStop - localStart + 1;
}

using namespace std;

void ECA_serial(int nx, int initial_index, int rule, int maxiter,int &state_counts, cv::Mat& population)
{
    //---------------------------------
    // Generate the CA population
    //---------------------------------
    

    assert(initial_index >= 0 && initial_index < nx);

    //for easier printing we will make the zero state white
    for (unsigned int ix = 0; ix < nx; ix++)
    {
        population.at<uchar>(0, ix) = 255;
    }

    //and the one state black
    population.at<uchar>(0, initial_index) = 0;

    int neighbour_states[3];
    
    for (int iter = 0; iter < maxiter-1; iter++)
    {
   /*    // Uncomment if you want to see the generation 
        cv::namedWindow("Population", cv::WINDOW_AUTOSIZE);
        cv::imshow("Population", population);
        cv::waitKey(10);

        std::cout << "Iteration # " << iter << " of " << maxiter << std::endl;*/

        for (int ix = 0; ix < nx; ix++)
        {
            //This could be implemented far more efficiently on a sliding rule
            if (ix == 0)
            {
                neighbour_states[0] = 0;
                neighbour_states[1] = (population.at<uchar>(iter, ix) == 0);
                neighbour_states[2] = (population.at<uchar>(iter, ix + 1) == 0);
            }
            else if (ix == nx - 1)
            {
                neighbour_states[0] = (population.at<uchar>(iter, ix - 1) == 0);
                neighbour_states[1] = (population.at<uchar>(iter, ix) == 0);
                neighbour_states[2] = 0;
            }
            else
            {
                neighbour_states[0] = (population.at<uchar>(iter, ix - 1) == 0);
                neighbour_states[1] = (population.at<uchar>(iter, ix) == 0);
                neighbour_states[2] = (population.at<uchar>(iter, ix + 1) == 0);
            }

            //convert the neighbour states to an integer
            int neighbour_pattern_index = 0;
            neighbour_pattern_index |= (neighbour_states[0] << 2);
            neighbour_pattern_index |= (neighbour_states[1] << 1);
            neighbour_pattern_index |= (neighbour_states[2] << 0);

            //the next state is the "neighbour pattern index"th bit of the rule.
            int new_state = (rule & (1 << (neighbour_pattern_index))) != 0;

            //Uncomment if you want to see state conversion
            //std::cout << "Neighbour states = " << neighbour_states[0] << " " << neighbour_states[1] << " " << neighbour_states[2] << " = " << neighbour_pattern_index << std::endl;
            //std::cout << "For Rule " << rule << " this gives the new state = " << new_state << std::endl;
            state_counts++;
            assert(new_state == 0 || new_state == 1);
            population.at<uchar>(iter + 1, ix) = 255 * (1 - new_state);
        }
    }
}


void ECA(int cellStart,int cellStop,int maxiter, cv::Mat& population,int rule)
{
    int rank;
    int nproc;

    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dest;
    int source;
    if (rank == 0)
    {
        dest = 1;
        source = nproc - 1;

    }
    else if (rank == nproc - 1)
    {
        dest = 0;
        source = nproc -2;
    }
    else
    {
        dest = rank + 1;
        source = rank - 1;
    }
    MPI_Status status;
    int neighbour_state_send1 = 0;
    int neighbour_state_recv1 = 0;
    int neighbour_state_send2 = 0;
    int neighbour_state_recv2 = 0;

    int neighbour_states[3];
    int state_counts = 0;
    for (int iter = 0; iter < maxiter - 1; iter++)
    {   // if you want to visualize the generation for specific rank, you can uncomment the below section
        /*if (rank == 1)
        {
            cv::namedWindow("Population", cv::WINDOW_AUTOSIZE);
            cv::imshow("Population", population);
            cv::waitKey(10);
            std::cout << "Iteration # " << iter << " of " << maxiter << std::endl;
        }*/
        neighbour_state_send1 = (population.at<uchar>(iter, cellStop ) == 0);
        neighbour_state_recv1 = 0;

        neighbour_state_send2 = (population.at<uchar>(iter, cellStart) == 0);
        neighbour_state_recv2 = 0;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Sendrecv(&neighbour_state_send1, 1, MPI_INT, dest, 98, &neighbour_state_recv1, 1, MPI_INT, source, 98, MPI_COMM_WORLD, &status);

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Sendrecv(&neighbour_state_send2, 1, MPI_INT, source, 99, &neighbour_state_recv2, 1, MPI_INT, dest, 99, MPI_COMM_WORLD, &status);

        for (int ix = cellStart; ix <= cellStop; ix++)
        {
            //This could be implemented far more efficiently on a sliding rule
            if (ix == cellStart)
            {
                neighbour_states[0] = neighbour_state_recv1;
                neighbour_states[1] = (population.at<uchar>(iter, ix) == 0);
                neighbour_states[2] = (population.at<uchar>(iter, ix + 1) == 0);

                
            }
            else if (ix == cellStop )
            {
                neighbour_states[0] = (population.at<uchar>(iter, ix - 1) == 0);
                neighbour_states[1] = (population.at<uchar>(iter, ix) == 0);
                neighbour_states[2] = neighbour_state_recv2; 
                
            }
            else
            {
                neighbour_states[0] = (population.at<uchar>(iter, ix - 1) == 0);
                neighbour_states[1] = (population.at<uchar>(iter, ix) == 0);
                neighbour_states[2] = (population.at<uchar>(iter, ix + 1) == 0);
            }

            //convert the neighbour states to an integer
            int neighbour_pattern_index = 0;
            neighbour_pattern_index |= (neighbour_states[0] << 2);
            neighbour_pattern_index |= (neighbour_states[1] << 1);
            neighbour_pattern_index |= (neighbour_states[2] << 0);

            //the next state is the "neighbour pattern index"th bit of the rule.
            int new_state = (rule & (1 << (neighbour_pattern_index))) != 0;

            //Uncomment if you want to see state conversion
            //std::cout << "Neighbour states = " << neighbour_states[0] << " " << neighbour_states[1] << " " << neighbour_states[2] << " = " << neighbour_pattern_index << std::endl;
            //std::cout << "For Rule " << rule << " this gives the new state = " << new_state << std::endl;
            state_counts++;
            assert(new_state == 0 || new_state == 1);
            population.at<uchar>(iter + 1, ix) = 255 * (1 - new_state);
        }
    }
    
     //std::cout << "state counts " << "Rank " << rank << " = " << state_counts << std::endl;
}

int main(int argc, char** argv)
{
    

    int rank, nproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 5)
    {
        std::cout << "You need to input information" << std::endl;
        assert(argc == 5);
    }
    //-----------------------
    // Convert Command Line
    //-----------------------

    int nx = atoi(argv[1]);
    int initial_index = atoi(argv[2]);
    int rule = atoi(argv[3]);
    int maxiter = atoi(argv[4]);

    //---------------------------------
    // Generate the CA population
    //---------------------------------
    double parallel_time_start;
    double parallel_time_stop ;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
         parallel_time_start = MPI_Wtime();
    }
    

    cv::Mat population(maxiter, nx, CV_8UC1);

    assert(initial_index >= 0 && initial_index < nx);

    //for easier printing we will make the zero state white
    for (unsigned int ix = 0; ix < nx; ix++)
    {
        population.at<uchar>(0, ix) = 255;
    }
    //and the one state black
    population.at<uchar>(0, initial_index) = 0;

    int cellStart, cellStop, cellCount;
    
    parallelRange(0, nx - 1, rank, nproc, cellStart, cellStop, cellCount);

    ECA(cellStart, cellStop, maxiter, population, rule);

    MPI_Datatype cell_cols;
    MPI_Type_vector(maxiter, cellCount, nx, MPI_UNSIGNED_CHAR, &cell_cols);
    MPI_Type_commit(&cell_cols);

    
    //std::cout << "processor " << rank << " cellstart " << cellStart << " cellstop " << cellStop << " cellcount " << cellCount << std::endl;
    
    if (rank != 0)
    {
        MPI_Send(&population.data[cellStart], 1, cell_cols, 0, 99, MPI_COMM_WORLD);
        //std::cout << "processor " << rank << " sent " << cellCount << " cols to processor 0" << std::endl;
    }
    else 
    {
        std::vector<int> localstarts(nproc), localstops(nproc), localcounts(nproc);

        cv::Mat All_population(maxiter, nx, CV_8UC1);
        
        for (int irank = 1; irank < nproc; irank++)
        {
            parallelRange(0, nx - 1, irank, nproc, localstarts[irank], localstops[irank], localcounts[irank]);
            MPI_Type_vector(maxiter, localcounts[irank], nx, MPI_UNSIGNED_CHAR, &cell_cols);
            MPI_Type_commit(&cell_cols);
            MPI_Recv(&population.data[localstarts[irank]], 1, cell_cols, irank, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        parallel_time_stop = MPI_Wtime();
        std::cout << "Job completed" << std::endl;
        int serial_state_counts=0;

        cv::Mat population_serial(maxiter, nx, CV_8UC1);

        double serial_time_start = MPI_Wtime();
        ECA_serial(nx, initial_index, rule, maxiter, serial_state_counts, population_serial);
        double serial_time_stop = MPI_Wtime();


        double serial_time = serial_time_stop - serial_time_start;
        double parallel_time = parallel_time_stop - parallel_time_start;

        std::cout << "Serial Time = " << serial_time << " seconds." << std::endl;
        std::cout << "Parallel Time = " << parallel_time << " seconds." << std::endl;
        std::cout << "Speedup = " << serial_time / parallel_time << std::endl;
        std::cout << "Serial state counts = " << serial_state_counts << std::endl;
       
        ostringstream converter;
        converter << "Parallel_ElementaryCA_" << nx << "_x_" << maxiter << "_Rule" << rule << ".jpg";
        imwrite(converter.str(), population);

        ostringstream converters;
        converters << "Serial_ElementaryCA_" << nx << "_x_" << maxiter << "_Rule" << rule << ".jpg";
        imwrite(converters.str(), population_serial);

        cv::namedWindow("Parallel ElementaryCA", cv::WINDOW_AUTOSIZE);
        cv::imshow("Parallel ElementaryCA", population);

        cv::namedWindow("Serial ElementaryCA", cv::WINDOW_AUTOSIZE);
        cv::imshow("Serial ElementaryCA", population_serial);

        cv::waitKey(3000);	//wait 3 seconds before closing image (or a keypress to close)
        
    }
    
    if (rank !=0)
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    

    MPI_Finalize();
    return 0;
}
