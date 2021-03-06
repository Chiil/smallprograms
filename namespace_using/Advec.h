#ifndef ADVEC
#define ADVEC

#include <exception>
#include "Advec_disabled.h"
#include "Advec_2.h"


namespace Advec
{
    std::string swadvec;

    void init(const std::string& swadvecin)
    {
        swadvec = swadvecin;
    }

    void exec()
    {
        if (swadvec == "disabled")
            Advec_disabled::exec();
        else if (swadvec == "2")
            Advec_2::exec();
        else
        {
            std::string error = "Illegal option for swadvec: " + swadvec;
            throw std::runtime_error(error);
        }
    }
}
#endif
