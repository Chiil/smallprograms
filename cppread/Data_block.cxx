#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include "Data_block.h"

namespace
{
    std::map< std::string, std::vector<std::string> > data_series;

    std::vector<std::string> split_line(std::string& line, const std::string splitter)
    {
        std::vector<std::string> strings;

        // Strip of the comments.
        boost::trim(line);
        boost::split(strings, line, boost::is_any_of("#"));

        // Keep part that is not comment and clear strings.
        if (strings.size() >= 2)
            line = strings[0];
        strings.clear();

        // Return if line is empty.
        if (line.empty())
            return strings;

        // Strip of all the whitespace.
        boost::trim(line);


        // Split string on the given splitter.
        boost::split(strings, line, boost::is_any_of(splitter));

        return strings;
    }
}

Data_block::Data_block(const std::string& file_name)
{
    // Read file and throw exception on error.
    std::ifstream infile;

    infile.open(file_name);
    if (!infile.good())
        throw std::runtime_error("Illegal file name");

    std::string line;

    // First, read the header.
    int number_of_vectors;
    int line_number = 0;
    while (std::getline(infile, line))
    {
        ++line_number;
        std::vector<std::string> strings = split_line(line, " \t");
        if (strings.size() == 0)
            continue;

        number_of_vectors = strings.size();

        for (const std::string& s : strings)
        {
            auto it = data_series.find(s);
            if (it != data_series.end())
                throw std::runtime_error("Duplicate name in header");
            else
                data_series[s] = std::vector<std::string>();
        }
        break;
    }

    // Second, read the data.
    while (std::getline(infile, line))
    {
        ++line_number;
        std::vector<std::string> strings = split_line(line, " \t");
        if (strings.size() == 0)
            continue;

        if (strings.size() != number_of_vectors)
        {
            std::string error_string = "Illegal number of items on line ";
            error_string += std::to_string(line_number);
            throw std::runtime_error(error_string);
        }

        auto it_s = strings.begin();
        for (auto& v : data_series)
        {
            v.second.push_back(*it_s);
            ++it_s;
        }
    }

    for (const auto& v : data_series)
    {
        std::cout << "label: " << v.first << std::endl;
        for (const auto& s : v.second)
            std::cout << s << std::endl;
    }
}
