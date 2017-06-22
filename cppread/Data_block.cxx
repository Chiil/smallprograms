#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include "Data_block.h"

namespace
{
    std::vector<std::string> split_string(std::string& line, const std::string splitter)
    {
        // Strip of the comments.
        boost::trim(line);
        std::vector<std::string> strings;
        boost::split(strings, line, boost::is_any_of("#"));

        // Keep part that is not comment.
        if (strings.size() >= 2)
            line = strings[0];

        // Strip of all the whitespace.
        boost::trim(line);

        // Split string on the given splitter.
        strings.clear();
        boost::split(strings, line, boost::is_any_of(splitter));

        return strings;
    }
}

Data_block::Data_block(const std::string& file_name)
{
    std::string blockname;

    // Read file and throw exception on error.
    std::ifstream infile;

    infile.open(file_name);
    if (!infile.good())
        throw std::runtime_error("Illegal file name");

    std::string line;

    while (std::getline(infile, line))
    {
        std::vector<std::string> strings = split_string(line, " ");

        std::cout << "Number of items: " << strings.size() << std::endl;
        for (const std::string& s : strings)
            std::cout << s << ", ";

        std::cout << std::endl;
    }
}
