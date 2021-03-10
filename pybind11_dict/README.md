# Compilation

```
g++-10 -O3 -Wall -shared -std=c++14 -undefined dynamic_lookup $(python3 -m pybind11 --includes) test_dict.cpp -o test_dict$(python3-config --extension-suffix)
```
