#include <filesystem>
#include <iostream>

int main()
{
    std::filesystem::path cross_path("./cross");
    std::cout << "Checking for cross directory... ";
    if (std::filesystem::exists(cross_path))
        std::cout << "exists" << std::endl;
    else
    {
        std::filesystem::create_directory(cross_path);
        std::cout << "created" << std::endl;
    }

    return 0;
}
