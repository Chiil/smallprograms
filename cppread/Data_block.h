#ifndef DATA_BLOCK
// #include <map>

class Data_block
{
    public:
        Data_block(const std::string&);
        template<typename T> std::vector<T> get_vector(const std::string&, const int);
};
#endif

