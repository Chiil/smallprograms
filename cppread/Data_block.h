#ifndef DATA_BLOCK
// #include <map>

class Data_block
{
    public:
        Data_block(const std::string&);
        template<typename T> std::vector<T> get_vector(const std::string&,
                                                       const int,
                                                       const int start_index=0);
        template<typename T> void get_vector_range(std::vector<T>&,
                                                   const std::string&,
                                                   const int,
                                                   const int,
                                                   const int);
};
#endif

