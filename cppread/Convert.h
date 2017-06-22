#ifndef CONVERT
namespace Convert
{
    template<typename T>
    void check_item(const T&);
    
    template<typename T>
    T get_item_from_stream(std::istringstream&);
}
#endif
