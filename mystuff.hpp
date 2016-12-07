#ifndef __MYSTUFF335503__IAMTONI__
#define __MYSTUFF335503__IAMTONI__
#include <exception>
#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <array>
#include <vector>
#include <chrono>

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
        auto current_time=std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        error.append(std::ctime(&current_time));
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
        auto current_time=std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        error.append(std::ctime(&current_time));
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

static_assert(sizeof(char)==1, "sizeof(char)!=1");
template<unsigned long height, unsigned long width>
class Image: public std::array<std::array<std::array<unsigned char,3>,width>,height>
{
private:
public:
    void to_bmp_file(const char *filename) const
    {
        static constexpr size_t raw_bitmap_data_size=(width*3+(4-width*3%4)%4)*height;
        static constexpr size_t header_size=54;
        static constexpr size_t file_size=header_size+raw_bitmap_data_size;
        static constexpr size_t pad_size=((4-width*3%4)%4);
        const size_t thousand=1000;

        std::array<char, 54> header_data;
        header_data.fill(0);
        header_data[ 0]=(char)66;
        header_data[ 1]=(char)77;
        header_data[ 2]=(char)((file_size>> 0)&0xFF);
        header_data[ 3]=(char)((file_size>> 8)&0xFF);
        header_data[ 4]=(char)((file_size>>16)&0xFF);
        header_data[ 5]=(char)((file_size>>24)&0xFF);
        // 4 reserved bytes
        header_data[10]=(char)54;

        //bmpinfoheader:[u8;40]
        header_data[14]=(char)40;
        // 3 more bytes for the length of this header (not needed here)
        header_data[18]=(char)((width>> 0)&0xFF);
        header_data[19]=(char)((width>> 8)&0xFF);
        header_data[20]=(char)((width>>16)&0xFF);
        header_data[21]=(char)((width>>24)&0xFF);
        header_data[22]=(char)((height>> 0)&0xFF);
        header_data[23]=(char)((height>> 8)&0xFF);
        header_data[24]=(char)((height>>16)&0xFF);
        header_data[25]=(char)((height>>24)&0xFF);
        header_data[26]=(char)1;
        header_data[28]=(char)24;
        header_data[34]=(char)((raw_bitmap_data_size>> 0)&0xFF);
        header_data[35]=(char)((raw_bitmap_data_size>> 8)&0xFF);
        header_data[36]=(char)((raw_bitmap_data_size>>16)&0xFF);
        header_data[37]=(char)((raw_bitmap_data_size>>24)&0xFF);
        header_data[38]=(char)((thousand>> 0)&0xFF);
        header_data[39]=(char)((thousand>> 8)&0xFF);
        header_data[40]=(char)((thousand>>16)&0xFF);
        header_data[41]=(char)((thousand>>24)&0xFF);
        header_data[42]=(char)((thousand>> 0)&0xFF);
        header_data[43]=(char)((thousand>> 8)&0xFF);
        header_data[44]=(char)((thousand>>16)&0xFF);
        header_data[45]=(char)((thousand>>24)&0xFF);


        std::array<char, pad_size> pad;pad.fill(0);
        std::ofstream out(filename,std::ios_base::trunc|std::ios::binary);
        assert(out.good());
        out.write(header_data.data(), header_data.size());
        for(size_t i=(*this).size()-1;;i--)
        {
            for(const auto &pixel:(*this)[i])
            {
                out.write((const char *) pixel.data()+2, 1);
                out.write((const char *) pixel.data()+1, 1);
                out.write((const char *) pixel.data()+0, 1);
            }
            out.write(pad.data(), pad.size());
            if(i==0) break;
        }
        out.close(); //redundant
    }
};

template<unsigned long height, unsigned long width>
class GrayscaleImage: public std::array<std::array<unsigned char,width>,height>
{
private:
public:
    void to_bmp_file(const char *filename) const
    {
        static constexpr size_t raw_bitmap_data_size=(width*3+(4-width*3%4)%4)*height;
        static constexpr size_t header_size=54;
        static constexpr size_t file_size=header_size+raw_bitmap_data_size;
        static constexpr size_t pad_size=((4-width*3%4)%4);
        const size_t thousand=1000;

        std::array<char, 54> header_data;
        header_data.fill(0);
        header_data[ 0]=(char)66;
        header_data[ 1]=(char)77;
        header_data[ 2]=(char)((file_size>> 0)&0xFF);
        header_data[ 3]=(char)((file_size>> 8)&0xFF);
        header_data[ 4]=(char)((file_size>>16)&0xFF);
        header_data[ 5]=(char)((file_size>>24)&0xFF);
        // 4 reserved bytes
        header_data[10]=(char)54;

        //bmpinfoheader:[u8;40]
        header_data[14]=(char)40;
        // 3 more bytes for the length of this header (not needed here)
        header_data[18]=(char)((width>> 0)&0xFF);
        header_data[19]=(char)((width>> 8)&0xFF);
        header_data[20]=(char)((width>>16)&0xFF);
        header_data[21]=(char)((width>>24)&0xFF);
        header_data[22]=(char)((height>> 0)&0xFF);
        header_data[23]=(char)((height>> 8)&0xFF);
        header_data[24]=(char)((height>>16)&0xFF);
        header_data[25]=(char)((height>>24)&0xFF);
        header_data[26]=(char)1;
        header_data[28]=(char)24;
        header_data[34]=(char)((raw_bitmap_data_size>> 0)&0xFF);
        header_data[35]=(char)((raw_bitmap_data_size>> 8)&0xFF);
        header_data[36]=(char)((raw_bitmap_data_size>>16)&0xFF);
        header_data[37]=(char)((raw_bitmap_data_size>>24)&0xFF);
        header_data[38]=(char)((thousand>> 0)&0xFF);
        header_data[39]=(char)((thousand>> 8)&0xFF);
        header_data[40]=(char)((thousand>>16)&0xFF);
        header_data[41]=(char)((thousand>>24)&0xFF);
        header_data[42]=(char)((thousand>> 0)&0xFF);
        header_data[43]=(char)((thousand>> 8)&0xFF);
        header_data[44]=(char)((thousand>>16)&0xFF);
        header_data[45]=(char)((thousand>>24)&0xFF);


        std::array<char, pad_size> pad;pad.fill(0);
        std::ofstream out(filename,std::ios_base::trunc|std::ios::binary);
        assert(out.good());
        out.write(header_data.data(), header_data.size());
        for(size_t i=(*this).size()-1;;i--)
        {
            for(const auto &p:(*this)[i])
            {
                std::array<char,3> pixel;
                pixel.fill(p);
                out.write(pixel.data(), pixel.size());
                // out.write((const char *) &p, 1);
                // out.write((const char *) &p, 1);
                // out.write((const char *) &p, 1);
            }
            out.write(pad.data(), pad.size());
            if(i==0) break;
        }
        out.close(); //redundant
    }
};
#endif