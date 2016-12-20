#include "matrix.hpp"
#include "mystuff.hpp"
using namespace std;






union bytewise_int32_t
{
    int32_t val;
    std::array<int8_t, 4> bytewise_val;
};

int main()
{
    static constexpr size_t test_image_num=10000;
    static constexpr size_t num_outs=100;
    // static constexpr size_t image_size=28*28;
    array<size_t, test_image_num> image_afiliation;
    {
        static constexpr size_t prototype_num=20;
        static constexpr size_t code_size=10;
        static constexpr size_t num_iterations=100;
        static constexpr size_t num_samples_for_initializing_prototypes=10;

        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<size_t> dst(0,test_image_num-1);
        // uniform_real_distribution<double> dst(0,1);

        array<Matrix<1,code_size>,test_image_num> test_image_codes;
        {
            ifstream in("test_code_matrices.tcm", std::ios::binary);
            assert(in.good());
            for(auto &test_image_code:test_image_codes)
            {
                test_image_code.from_bin_file(in);
            }
        }

        array<Matrix<1,code_size>, prototype_num> prototypes;
        for(auto &prototype:prototypes)
        {
            for(size_t i=0;i<num_samples_for_initializing_prototypes;i++)prototype.add(test_image_codes[dst(gen)]);
            prototype.div((double)num_samples_for_initializing_prototypes);
        }
        //     for(auto &element:prototype[0])
        //         element=dst(gen);


        for(size_t iteration=0;iteration<num_iterations;iteration++)
        {
            double added_distances=0.0;
            for(size_t i=0;i<test_image_num;i++)
            {
                size_t shortest_distance_index=0;
                double shortest_distance=e_distance2(test_image_codes[i], prototypes[shortest_distance_index]);
                for(size_t j=1;j<prototype_num;j++)
                {
                    double dist=e_distance2(test_image_codes[i], prototypes[j]);
                    if(dist<shortest_distance)
                    {
                        shortest_distance=dist;
                        shortest_distance_index=j;
                    }
                }
                image_afiliation[i]=shortest_distance_index;
                added_distances+=shortest_distance;
            }
            print(added_distances);
            array<int, prototype_num> num_afiliated_to_prototype;
            for(auto &natp:num_afiliated_to_prototype) natp=0;
            for(auto &prototype:prototypes)prototype.set(0.0);
            for(size_t i=0;i<test_image_num;i++)
            {
                prototypes[image_afiliation[i]].add(test_image_codes[i]);
                num_afiliated_to_prototype[image_afiliation[i]]++;
            }
            for(size_t j=0;j<prototype_num;j++)
            {
                prototypes[j].div((double)num_afiliated_to_prototype[j]);
            }
            // for(const auto natp:num_afiliated_to_prototype) cout << natp << " ";cout << endl;
            if(iteration==num_iterations-1){for(const auto natp:num_afiliated_to_prototype) cout << natp << " ";cout << endl;}
        }
    }

    {
        ifstream in("../mnist/t10k-images.idx3-ubyte", std::ios::binary);
        assert(in.good());
        bytewise_int32_t a;
        in.read((char*)&a.bytewise_val[3], 1);
        in.read((char*)&a.bytewise_val[2], 1);
        in.read((char*)&a.bytewise_val[1], 1);
        in.read((char*)&a.bytewise_val[0], 1);
        assert(a.val==2051);
        in.read((char*)&a.bytewise_val[3], 1);
        in.read((char*)&a.bytewise_val[2], 1);
        in.read((char*)&a.bytewise_val[1], 1);
        in.read((char*)&a.bytewise_val[0], 1);
        assert(a.val==test_image_num);
        in.read((char*)&a.bytewise_val[3], 1);
        in.read((char*)&a.bytewise_val[2], 1);
        in.read((char*)&a.bytewise_val[1], 1);
        in.read((char*)&a.bytewise_val[0], 1);
        assert(a.val==28);
        in.read((char*)&a.bytewise_val[3], 1);
        in.read((char*)&a.bytewise_val[2], 1);
        in.read((char*)&a.bytewise_val[1], 1);
        in.read((char*)&a.bytewise_val[0], 1);
        assert(a.val==28);

        // std::array<unsigned char,image_size> buffer;// for(size_t j=0;j<image_size;j++)images[i][j]=buffer[j]/255.;
        GrayscaleImage<28,28> img;
        for(size_t i=0;i<num_outs;i++)
        {
            in.read((char*)img.data(), 28*28);
            assert(in.good());

            char buffer[256];
            sprintf(buffer, "out/%02lu_%04lu.bmp", image_afiliation[i]+1, i);
            // print(buffer);
            img.to_bmp_file(buffer);
        }
    }

    return 0;
}