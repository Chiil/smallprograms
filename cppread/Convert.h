#ifndef CONVERT
namespace Convert
{
    template<typename T>
    void check_item(const T& t) {}
    
    template<>
    void check_item(const std::string& s)
    {
        // Check whether string is empty or whether the first character is not alpha.
        if (s.empty())
            throw std::runtime_error("Illegal string");
        else if (!isalpha(s[0]))
            throw std::runtime_error("Illegal string: " + s);
    
        // Return string if all characters are alphanumeric.
        if (find_if(s.begin(), s.end(), [](const char c) { return !std::isalnum(c); }) == s.end())
            return;
        else
            throw std::runtime_error("Illegal string: " + s);
    }

    template<typename T>
    T get_item_from_stream(std::istringstream& ss)
    {
        // Read the item from the stringstream, operator >> trims automatically.
        T item;
        if (!(ss >> item))
            throw std::runtime_error("Item does not match type");
    
        // Check whether stringstream is empty, if not type is incorrect.
        std::string dummy;
        if (ss >> dummy)
            throw std::runtime_error("Partial item does not match type");
    
        return item;
    }
}
#endif
