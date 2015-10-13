#include <string>

extern void print_message(const std::string&);

int main()
{
    std::string message = "Hello World!";
    print_message(message);
    return 0;
}
