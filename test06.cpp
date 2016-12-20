#include "matrix.hpp"
#include "mystuff.hpp"
#include <random>
#include <sys/resource.h>
using namespace std;

template<unsigned long input_size, unsigned long hidden_size, unsigned long batch_size=1>
class AutoencoderLayer
{
private:
    //Parameters
    Matrix<input_size,hidden_size> syn;
    Matrix<1,hidden_size> bias_hidden;
    Matrix<1,input_size> bias_out;
    //State
    Matrix<batch_size,hidden_size> layer1;
    Matrix<batch_size,input_size> layer2;
    Matrix<batch_size,hidden_size> delta1;
    Matrix<batch_size,input_size> delta2;
    //Optimizer stuff
    Matrix<input_size,hidden_size> syn_rms;
    Matrix<1,hidden_size> bias_hidden_rms;
    Matrix<1,input_size> bias_out_rms;
public:
    AutoencoderLayer(): syn_rms(1.0), bias_hidden_rms(1.0), bias_out_rms(1.0)
    {
        syn.randomize_for_autoencoder();
        bias_hidden.set(0.0);
        bias_out.set(0.0);
    }

    void reset_rms() noexcept
    {
        syn_rms.set(1.0);
        bias_hidden_rms.set(1.0);
        bias_out_rms.set(1.0);
    }

    template<unsigned long L>
    void set_parameters(const AutoencoderLayer<input_size, hidden_size, L> &ael) noexcept
    {
        syn.set(ael.syn);
        bias_hidden.set(ael.bias_hidden);
        bias_out.set(ael.bias_out);
    }

    const Matrix<batch_size,hidden_size>& calc1(const Matrix<batch_size,input_size>& X) noexcept
    {
        layer1.equals_a_dot_b(X,syn);
        layer1.add_as_rows(bias_hidden);
        layer1.apply_sigmoid();
        return layer1;
    }

    const Matrix<batch_size,hidden_size>& calc1_corrupted(const Matrix<batch_size,input_size>& X, std::mt19937 &gen, std::bernoulli_distribution &dst) noexcept
    {
        Matrix<batch_size,input_size> corrupted_X(X, gen, dst);
        layer1.equals_a_dot_b(corrupted_X,syn);
        layer1.add_as_rows(bias_hidden);
        layer1.apply_sigmoid();
        return layer1;
    }

    const Matrix<batch_size,input_size>& calc2(const Matrix<batch_size,hidden_size>& X) noexcept
    {
        layer2.equals_a_dot_bt(X, syn);
        layer2.add_as_rows(bias_out);
        layer2.apply_sigmoid();
        return layer2;
    }

    inline Matrix<batch_size,hidden_size>& get_delta1() noexcept
    {
        return delta1;
    }

    inline Matrix<batch_size,input_size>& get_delta2() noexcept
    {
        return delta2;
    }

    inline const Matrix<batch_size,hidden_size>& get_output1() noexcept
    {
        return layer1;
    }

    inline const Matrix<batch_size,input_size>& get_output2() noexcept
    {
        return layer2;
    }

    double set_delta2(const Matrix<batch_size,input_size>& Y) noexcept
    {
        delta2.equals_a_sub_b(Y,layer2);
        return delta2.sum_of_squares();
    }

    void propagate_delta2(Matrix<batch_size,hidden_size>& delta_next) noexcept
    {
        delta2.mult_after_func01(layer2);
        delta_next.equals_a_dot_b(delta2, syn);
    }

    void propagate_delta1() noexcept
    {
        delta1.mult_after_func01(layer1);
    }

    void propagate_delta1(Matrix<batch_size,input_size>& delta_next) noexcept
    {
        propagate_delta1();
        delta_next.equals_a_dot_bt(delta1, syn);
    }

    void learn(const double learning_rate, const Matrix<batch_size,input_size>& X1, const Matrix<batch_size,hidden_size>& X2, const double decay=.9) noexcept
    {
        syn_rms.mul(decay);
        for(size_t i=0;i<input_size;i++)
            for(size_t j=0;j<hidden_size;j++)
            {
                double gradient=0.0;
                for(size_t k=0;k<batch_size;k++)gradient+=X1[k][i]*delta1[k][j]+delta2[k][i]*X2[k][j];
                syn_rms[i][j]+=gradient*gradient*(1-decay);
                syn[i][j]+=(gradient*learning_rate)/sqrt(syn_rms[i][j]+1e-8);
            }

        bias_hidden_rms.mul(decay);
        for(size_t j=0;j<hidden_size;j++)
        {
            double gradient=0.0;
            for(size_t k=0;k<batch_size;k++)gradient+=delta1[k][j];
            bias_hidden_rms[0][j]+=gradient*gradient*(1-decay);
            bias_hidden[0][j]+=(gradient*learning_rate)/sqrt(bias_hidden_rms[0][j]+1e-8);
        }

        bias_out_rms.mul(decay);
        for(size_t i=0;i<input_size;i++)
        {
            double gradient=0.0;
            for(size_t k=0;k<batch_size;k++)gradient+=delta2[k][i];
            bias_out_rms[0][i]+=gradient*gradient*(1-decay);
            bias_out[0][i]+=(gradient*learning_rate)/sqrt(bias_out_rms[0][i]+1e-8);
        }
    }
};

union bytewise_int32_t
{
    int32_t val;
    std::array<int8_t, 4> bytewise_val;
};

int moin()
{
    static constexpr size_t image_size=28*28;
    static constexpr size_t image_num=60000;
    static constexpr size_t hidden_size=4;
    static constexpr size_t batch_size=20;
    static constexpr double learning_rate=.005;
    static constexpr size_t num_iterations=80000/batch_size;
    static constexpr size_t num_pre_iterations=20000/batch_size;
    static constexpr size_t iterations_in_between_prints=1000;
    static constexpr double corruption=.25;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dst(0,image_num-1);
    std::bernoulli_distribution bern_dst(1-corruption);

    AutoencoderLayer<image_size, 512, batch_size> ael1;
    AutoencoderLayer<512, 256, batch_size> ael2;
    AutoencoderLayer<256, 128, batch_size> ael3;//to 32 after
    AutoencoderLayer<128, hidden_size, batch_size> ael4;
    // AutoencoderLayer<200, 100, batch_size> ael5;
    // AutoencoderLayer<100, hidden_size, batch_size> ael6;

    std::vector<std::array<double,image_size>> images(image_num);
    // X.reserve(image_num);
    {
        std::ifstream in("../mnist/train-images.idx3-ubyte", std::ios::binary);
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
        assert(a.val==image_num);
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

        std::array<unsigned char,image_size> buffer;
        for(size_t i=0;i<image_num;i++)
        {
            in.read((char*)buffer.data(), buffer.size());
            for(size_t j=0;j<image_size;j++)images[i][j]=buffer[j]/255.;
            assert(in.good());
        }
    }

    Matrix<batch_size,image_size> X;
    double error=0.0;
    auto set_X=[&X, &gen, &dst, &images]()
    {
        for(size_t b=0;b<batch_size;b++)
        {
            size_t rpos=dst(gen);
            for(size_t i=0;i<image_size;i++)X[b][i]=images[rpos][i];
        }
    };

    print("pretrain1...");
    for(size_t iteration=0;iteration<num_pre_iterations;iteration++)
    {
        set_X();
        ael1.calc1_corrupted(X, gen, bern_dst);
        ael1.calc2(ael1.get_output1());
        error+=ael1.set_delta2(X);
        ael1.propagate_delta2(ael1.get_delta1());
        ael1.propagate_delta1();
        ael1.learn(learning_rate, X, ael1.get_output1());
        if((iteration+1)%(iterations_in_between_prints/batch_size)==0)
        {print(error/iterations_in_between_prints);error=0.0;}
    }

    print("pretrain2...");
    for(size_t iteration=0;iteration<num_pre_iterations;iteration++)
    {
        set_X();
        ael1.calc1_corrupted(X, gen, bern_dst);
        ael2.calc1(ael1.get_output1());
        ael2.calc2(ael2.get_output1());
        error+=ael2.set_delta2(ael1.get_output1());
        ael2.propagate_delta2(ael2.get_delta1());
        ael2.propagate_delta1();
        ael2.learn(learning_rate, ael1.get_output1(), ael2.get_output1());
        if((iteration+1)%(iterations_in_between_prints/batch_size)==0)
        {print(error/iterations_in_between_prints);error=0.0;}
    }

    print("pretrain3...");
    for(size_t iteration=0;iteration<num_pre_iterations;iteration++)
    {
        set_X();
        ael1.calc1_corrupted(X, gen, bern_dst);
        ael2.calc1(ael1.get_output1());
        ael3.calc1(ael2.get_output1());
        ael3.calc2(ael3.get_output1());
        error+=ael3.set_delta2(ael2.get_output1());
        ael3.propagate_delta2(ael3.get_delta1());
        ael3.propagate_delta1();
        ael3.learn(learning_rate, ael2.get_output1(), ael3.get_output1());
        if((iteration+1)%(iterations_in_between_prints/batch_size)==0)
        {print(error/iterations_in_between_prints);error=0.0;}
    }

    print("pretrain4...");
    for(size_t iteration=0;iteration<num_pre_iterations;iteration++)
    {
        set_X();
        ael1.calc1_corrupted(X, gen, bern_dst);
        ael2.calc1(ael1.get_output1());
        ael3.calc1(ael2.get_output1());
        ael4.calc1(ael3.get_output1());
        ael4.calc2(ael4.get_output1());
        error+=ael4.set_delta2(ael3.get_output1());
        ael4.propagate_delta2(ael4.get_delta1());
        ael4.propagate_delta1();
        ael4.learn(learning_rate, ael3.get_output1(), ael4.get_output1());
        if((iteration+1)%(iterations_in_between_prints/batch_size)==0)
        {print(error/iterations_in_between_prints);error=0.0;}
    }

    // print("pretrain5...");
    // for(size_t iteration=0;iteration<num_pre_iterations;iteration++)
    // {
    //     set_X();
    //     ael1.calc1_corrupted(X, gen, bern_dst);
    //     ael2.calc1(ael1.get_output1());
    //     ael3.calc1(ael2.get_output1());
    //     ael4.calc1(ael3.get_output1());
    //     ael5.calc1(ael4.get_output1());
    //     ael5.calc2(ael5.get_output1());
    //     error+=ael5.set_delta2(ael4.get_output1());
    //     ael5.propagate_delta2(ael5.get_delta1());
    //     ael5.propagate_delta1();
    //     ael5.learn(learning_rate, ael4.get_output1(), ael5.get_output1());
    //     if((iteration+1)%(iterations_in_between_prints/batch_size)==0)
    //     {print(error/iterations_in_between_prints);error=0.0;}
    // }

    // print("pretrain6...");
    // for(size_t iteration=0;iteration<num_pre_iterations;iteration++)
    // {
    //     set_X();
    //     ael1.calc1_corrupted(X, gen, bern_dst);
    //     ael2.calc1(ael1.get_output1());
    //     ael3.calc1(ael2.get_output1());
    //     ael4.calc1(ael3.get_output1());
    //     ael5.calc1(ael4.get_output1());
    //     ael6.calc1(ael5.get_output1());
    //     ael6.calc2(ael6.get_output1());
    //     error+=ael6.set_delta2(ael5.get_output1());
    //     ael6.propagate_delta2(ael6.get_delta1());
    //     ael6.propagate_delta1();
    //     ael6.learn(learning_rate, ael5.get_output1(), ael6.get_output1());
    //     if((iteration+1)%(iterations_in_between_prints/batch_size)==0)
    //     {print(error/iterations_in_between_prints);error=0.0;}
    // }


    auto calculate=[&X, &ael1, &ael2, &ael3, &ael4]()
    {
        ael1.calc1(X);
        ael2.calc1(ael1.get_output1());
        ael3.calc1(ael2.get_output1());
        ael4.calc1(ael3.get_output1());
        // ael5.calc1(ael4.get_output1());
        // ael6.calc1(ael5.get_output1());
        // ael6.calc2(ael6.get_output1());
        // ael5.calc2(ael6.get_output2());
        ael4.calc2(ael4.get_output1());
        ael3.calc2(ael4.get_output2());
        ael2.calc2(ael3.get_output2());
        ael1.calc2(ael2.get_output2());
    };
    auto set_deltas=[&X, &ael1, &ael2, &ael3, &ael4]()
    {
        double error=ael1.set_delta2(X);
        ael1.propagate_delta2(ael2.get_delta2());
        ael2.propagate_delta2(ael3.get_delta2());
        ael3.propagate_delta2(ael4.get_delta2());
        ael4.propagate_delta2(ael4.get_delta1());
        // ael5.propagate_delta2(ael6.get_delta2());
        // ael6.propagate_delta2(ael6.get_delta1());
        // ael6.propagate_delta1(ael5.get_delta1());
        // ael5.propagate_delta1(ael4.get_delta1());
        ael4.propagate_delta1(ael3.get_delta1());
        ael3.propagate_delta1(ael2.get_delta1());
        ael2.propagate_delta1(ael1.get_delta1());
        ael1.propagate_delta1();
        return error;
    };
    auto learn=[&X, &ael1, &ael2, &ael3, &ael4]()
    {
        ael1.learn(learning_rate, X, ael2.get_output2());
        ael2.learn(learning_rate, ael1.get_output1(), ael3.get_output2());
        ael3.learn(learning_rate, ael2.get_output1(), ael4.get_output2());
        ael4.learn(learning_rate, ael3.get_output1(), ael4.get_output1());
        // ael5.learn(learning_rate, ael4.get_output1(), ael6.get_output2());
        // ael6.learn(learning_rate, ael5.get_output1(), ael6.get_output1());
    };
    ael1.reset_rms();
    ael2.reset_rms();
    ael3.reset_rms();
    ael4.reset_rms();
    // ael5.reset_rms();
    // ael6.reset_rms();
    error=0.0;
    print("training...");
    for(size_t iteration=0;iteration<num_iterations;iteration++)
    {
        set_X();

        ael1.calc1_corrupted(X, gen, bern_dst);
        ael2.calc1(ael1.get_output1());
        ael3.calc1(ael2.get_output1());
        ael4.calc1(ael3.get_output1());
        ael4.calc2(ael4.get_output1());
        ael3.calc2(ael4.get_output2());
        ael2.calc2(ael3.get_output2());
        ael1.calc2(ael2.get_output2());
        // calculate();
        error+=set_deltas();
        learn();

        if((iteration+1)%(iterations_in_between_prints/batch_size)==0)
        {print(error/iterations_in_between_prints);error=0.0;}
    }
    print("training... part 2...");
    for(size_t iteration=0;iteration<num_iterations;iteration++)
    {
        // double nlr=learning_rate/(1.+(iteration*batch_size)/(4000.0));
        double nlr=learning_rate*pow(0.9440608762859234, (iteration*batch_size/1000.));
        // learning_rate=0.01*pow(0.8912509381337456, (iteration/1000));// gets divided by 10 every 20k steps
        set_X();

        ael1.calc1_corrupted(X, gen, bern_dst);
        ael2.calc1(ael1.get_output1());
        ael3.calc1(ael2.get_output1());
        ael4.calc1(ael3.get_output1());
        ael4.calc2(ael4.get_output1());
        ael3.calc2(ael4.get_output2());
        ael2.calc2(ael3.get_output2());
        ael1.calc2(ael2.get_output2());
        // calculate();
        error+=set_deltas();
        ael1.learn(nlr, X, ael2.get_output2());
        ael2.learn(nlr, ael1.get_output1(), ael3.get_output2());
        ael3.learn(nlr, ael2.get_output1(), ael4.get_output2());
        ael4.learn(nlr, ael3.get_output1(), ael4.get_output1());

        if((iteration+1)%(iterations_in_between_prints/batch_size)==0)
        {print(error/iterations_in_between_prints);error=0.0;}
    }

    error=0.0;
    print("calculating error...");
    for(size_t iteration=0;iteration<4*iterations_in_between_prints/batch_size;iteration++)
    {
        set_X();
        calculate();
        error+=ael1.set_delta2(X);
    }
    print("Error:", error/(iterations_in_between_prints*4));
    print("Iterations:", num_iterations);
    print("Pre-iterations:", num_pre_iterations);
    print("Batch size:", batch_size);
    print("Corruption:", corruption);
    print("Hidden size:", hidden_size);
    // num_iterations, num_pre_iterations, batch_size, corruption, hidden_size

    GrayscaleImage<28*2+14+28*2,28*batch_size> img;
    for(auto &row:img)row.fill(0);
    for(size_t iteration=0;iteration<batch_size;iteration++)
    {
        size_t rpos=dst(gen);
        for(size_t i=0;i<image_size;i++)X[iteration][i]=images[rpos][i];
        for(size_t i=0;i<28;i++)
        {
            for(volatile size_t j=0;j<28;j++)
            {
                img[i][j+iteration*28]=((images[rpos][i*28+j])*255);
                // print(i*28+j, i*28+j<image_size);
                // std::cout << i*28+j << " "<< (i*28+j<image_size) << std::endl;
            }
        }
    }
    calculate();
    for(size_t iteration=0;iteration<batch_size;iteration++)
    {
        for(size_t i=0;i<28;i++)
        {
            for(volatile size_t j=0;j<28;j++)
            {
                img[i+28][j+iteration*28]=(ael1.get_output2()[iteration][i*28+j]*255);
            }
        }
    }
    for(size_t iteration=0;iteration<batch_size;iteration++)
    {
        size_t rpos=dst(gen);
        for(size_t i=0;i<image_size;i++)X[iteration][i]=images[rpos][i];
        for(size_t i=0;i<28;i++)
        {
            for(volatile size_t j=0;j<28;j++)
            {
                img[i+70][j+iteration*28]=((images[rpos][i*28+j])*255);
            }
        }
    }
    calculate();
    for(size_t iteration=0;iteration<batch_size;iteration++)
    {
        for(size_t i=0;i<28;i++)
        {
            for(volatile size_t j=0;j<28;j++)
            {
                img[i+28+70][j+iteration*28]=(ael1.get_output2()[iteration][i*28+j]*255);
            }
        }
    }

    {
        char buffer[256];
        sprintf(buffer, "bmpfile_%lu_iters_%lu_train_iters_%lu_batches_%lf_corr_%lu_btn.bmp", num_iterations, num_pre_iterations, batch_size, corruption, hidden_size);
        img.to_bmp_file(buffer);
    }

    AutoencoderLayer<image_size, 512, 1> t_ael1;
    AutoencoderLayer<512, 256, 1> t_ael2;
    AutoencoderLayer<256, 128, 1> t_ael3;
    AutoencoderLayer<128, hidden_size, 1> t_ael4;

    std::vector<Matrix<1,image_size>> test_images(10000);
    // Matrix<10000, image_size> test_images;
    {
        std::ifstream in("../mnist/t10k-images.idx3-ubyte", std::ios::binary);
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
        assert(a.val==(int32_t)test_images.size());
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

        std::array<unsigned char,image_size> buffer;
        for(size_t i=0;i<test_images.size();i++)
        {
            in.read((char*)buffer.data(), buffer.size());
            for(size_t j=0;j<image_size;j++)test_images[i][0][j]=buffer[j]/255.;
            assert(in.good());
        }
    }

    // {
    //     ofstream out("test_code_matrices.tcm",std::ios_base::trunc|std::ios::binary);
    //     assert(out.good());
    //     //maybe use training images im k-means in the future too, and use only their labels to decide the clusters label
    //     for(const auto &test_image:test_images)
    //     {
    //         t_ael1.calc1(test_image);
    //         t_ael2.calc1(t_ael1.get_output1());
    //         t_ael3.calc1(t_ael2.get_output1());
    //         t_ael4.calc1(t_ael3.get_output1());
    //         // t_ael4.calc2(t_ael4.get_output1());
    //         // t_ael3.calc2(t_ael4.get_output2());
    //         // t_ael2.calc2(t_ael3.get_output2());
    //         // t_ael1.calc2(t_ael2.get_output2());
    //         t_ael4.get_output1().to_bin_file(out);
    //     }
    // }

    return 0;
}
int main()
{
    struct rlimit rl;
    // print("getrlimit:", getrlimit(RLIMIT_STACK, &rl));
    assert(getrlimit(RLIMIT_STACK, &rl)==0);
    print("Current stack size:", rl.rlim_cur);
    print("Max stack size:", rl.rlim_max);
    rl.rlim_cur=0x10000000;
    assert(setrlimit(RLIMIT_STACK, &rl)==0);
    // print("setrlimit:", setrlimit(RLIMIT_STACK, &rl));
    print("Upgrading stack...");
    print("Current stack size:", rl.rlim_cur);
    return moin();
}