#include <iostream>
#include <thread>
#include <chrono>
#include <omp.h>
#include <vector>

void stats(std::vector<double>& v, const int iter)
{
    
    #pragma omp critical
    std::cout << "Stats thread: " << omp_get_thread_num() << " of iter = " << iter << " is starting!" << std::endl;

    #pragma omp parallel
    {
        #pragma omp for
        for (int i=0; i<v.size(); ++i)
        {
            #pragma omp critical
            std::cout << "Stats: for loop, i = " << i << ", thread = " << omp_get_thread_num() << std::endl;
            v[i] *= 0.5;
        }
    }

    std::this_thread::sleep_for(std::chrono::seconds(5));
    #pragma omp critical
    std::cout << "Stats thread: " << omp_get_thread_num() << " of iter = " << iter << " is finished!" << std::endl;
}

int main()
{
    std::vector<double> v(4);
    for (int i=0; i<v.size(); ++i)
        v[i] = i+1;

    omp_set_nested(1);

    #pragma omp parallel num_threads(2)
    {
        #pragma omp master
        {
            std::cout << "Parallel region with " << omp_get_num_threads() << " threads" << std::endl;

            int iter = 0;
            while (true)
            {
                #pragma omp critical
                std::cout << "Iter = " << iter << ", Main thread: " << omp_get_thread_num() << std::endl;

                if (iter % 20 == 0)
                {
                    #pragma omp taskwait
                    std::vector<double> v_copy(v);

                    #pragma omp task
                    stats(v_copy, iter);
                }
                ++iter;

                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }
    }
    return 0;
}

