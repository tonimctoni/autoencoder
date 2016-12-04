#ifndef __MYSTUFF335503__IAMTONI__
#define __MYSTUFF335503__IAMTONI__
#include <exception>
#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <array>
#include <vector>

//Exception to be thrown on assertion fails.
class AssertionException : public std::exception
{
private:
    std::string error;
public:
    AssertionException(const char *file, const char *func, int line, const char* assertion)
    {
        error.reserve(512);
        error.append("\n");
        error.append("File:       ");
        error.append(file);
        error.append("\n");
        error.append("Function:   ");
        error.append(func);
        error.append("\n");
        error.append("Line:       ");
        error.append(std::to_string(line));
        error.append("\n");
        error.append("Assertion:  ");
        error.append(assertion);
        error.append("\n");
    }

    AssertionException(const char *message, const char *file, const char *func, int line, const char* assertion)
    {
        error.reserve(1024);
        error.append("\n");
        error.append("Message:    ");
        error.append(message);
        error.append("\n");
        error.append("File:       ");
        error.append(file);
        error.append("\n");
        error.append("Function:   ");
        error.append(func);
        error.append("\n");
        error.append("Line:       ");
        error.append(std::to_string(line));
        error.append("\n");
        error.append("Assertion:  ");
        error.append(assertion);
        error.append("\n");
    }

    virtual const char* what() const noexcept override
    {
        return error.c_str();
    }

    ~AssertionException()noexcept override{}
};

//Asserts "c", throws "AssertionException" on fail.
#define assert(c){\
    if(c){}else throw AssertionException(__FILE__, __func__, __LINE__, #c);\
}

#define assertm(c,m){\
    if(c){}else throw AssertionException(m, __FILE__, __func__, __LINE__, #c);\
}

inline void print()
{
    std::cout << std::endl;
}

// template<typename T>
// inline void print(T &v)
// {
//     std::cout << v << std::endl;
// }

// template<typename T, typename... Args>
// inline void print(T &v, Args... args)
// {
//     std::cout << v << " ";
//     print(args...);
// }

template<typename T>
inline void print(T v)
{
    std::cout << v << std::endl;
}

template<typename T, typename... Args>
inline void print(T v, Args... args)
{
    std::cout << v << " ";
    print(args...);
}

template<unsigned long N>
inline size_t get_weighted_random_index(const std::array<double, N> &arr)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());

    double sum=0.0;
    for(const auto e:arr)
        sum+=e;
    assert(sum<1.0625);
    std::uniform_real_distribution<double> dst(0.0,sum);

    sum=0.0;
    double r=dst(gen);
    for(size_t i=0;i<N;i++)
    {
        sum+=arr[i];
        if(r<=sum) return i;
    }

    assert(0);
}

template<unsigned long N>
inline size_t get_max_index(const std::array<double, N> &arr)
{
    size_t max_index=0;
    for(size_t i=1;i<N;i++)
    {
        if(arr[i]>arr[max_index])max_index=i;
    }
    return max_index;
}

inline void read_file_to_string(const char *filename, std::string &out_str)
{
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    assert(in.good());
    in.seekg(0, std::ios::end);
    out_str.resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(&out_str[0], out_str.size());
    in.close();
}

inline void read_file_to_string(const char *filename, std::string &out_str, size_t max_size)
{
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    assert(in.good());
    in.seekg(0, std::ios::end);
    out_str.resize(size_t(in.tellg())>max_size?max_size:size_t(in.tellg()));
    in.seekg(0, std::ios::beg);
    in.read(&out_str[0], size_t(in.tellg())>max_size?max_size:size_t(in.tellg()));
    in.close();
}

std::vector<std::string> split_string(const std::string &str, const std::string &sep)
{
    assert(str.size()>0);
    assert(sep.size()>0);
    assert(str.size()>sep.size());
    std::vector<std::string> ret;
    size_t index=0;
    for(;;)
    {
        if(index>=str.size()) break;
        size_t sep_index=str.find(sep, index);
        if(sep_index==std::string::npos) sep_index=str.size();

        ret.emplace_back(str, index, sep_index-index);

        index=sep_index+sep.size();
    }


    return ret;
}
#endif