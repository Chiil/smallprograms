#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>

void communicate_vector_string(std::vector<std::string>& vs, int mpiid)
{
    // Communicate the vector size
    int vector_size = vs.size();
    MPI_Bcast(&vector_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::cout << "(" << mpiid << ") vector size = " << vector_size << std::endl;

    vs.resize(vector_size);

    for (int i=0; i<vector_size; ++i)
    {
        int string_size = vs[i].size();
        MPI_Bcast(&string_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::cout << "(" << mpiid << "), " << i << ", string size = " << string_size << std::endl;

        vs[i].resize(string_size);

        // This is a nasty one. the .c_str() gives you a const char which Bcast does not like.
        MPI_Bcast(&vs[i][0], vs[i].size(), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
}

int main()
{
    MPI_Init(NULL, NULL);

    int mpiid;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiid);

    std::cout << "mpiid = " << mpiid << std::endl;

    std::vector<std::string> vs;

    if (mpiid == 0)
    {
        vs.push_back("Make");
        vs.push_back("America");
        vs.push_back("Great");
        vs.push_back("Again");
    }

    communicate_vector_string(vs, mpiid);

    for (auto& s : vs)
    {
        std::cout << "(" << mpiid << ") " << s << std::endl;
    }

    MPI_Finalize();
}
