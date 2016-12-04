// #include "perceptron_timeseries_class.hpp"
// #include "LSTM_class.hpp"
// #include "softmax_timeseries_class.hpp"
// #include <string>
// #include <vector>
// #include <array>
// #include <unordered_map>
// #include <memory>
// #include <ctime>
#include "matrix.hpp"
#include "mystuff.hpp"
#include <chrono>
#include <random>
using namespace std;

template<unsigned long input_size, unsigned long hidden_size>
class AutoencoderLayer
{
private:
    //Parameters
    Matrix<input_size,hidden_size> syn;
    Matrix<1,hidden_size> bias_hidden;
    Matrix<1,input_size> bias_out;
    //State
    Matrix<1,hidden_size> layer1;
    Matrix<1,input_size> layer2;
    Matrix<1,hidden_size> delta1;
    Matrix<1,input_size> delta2;
public:
    AutoencoderLayer()
    {
        syn.randomize_for_autoencoder();
        bias_hidden.set(0.0);
        bias_out.set(0.0);
    }

    const Matrix<1,hidden_size>& calc1(const Matrix<1,input_size>& X) noexcept
    {
        layer1.equals_a_dot_b(X,syn);
        layer1.add(bias_hidden);
        layer1.apply_sigmoid();
        return layer1;
    }

    const Matrix<1,hidden_size>& calc1_corrupted(const Matrix<1,input_size>& X, std::mt19937 &gen, std::bernoulli_distribution &dst) noexcept
    {
        Matrix<1,input_size> corrupted_X(X, gen, dst);
        layer1.equals_a_dot_b(corrupted_X,syn);
        layer1.add(bias_hidden);
        layer1.apply_sigmoid();
        return layer1;
    }

    const Matrix<1,input_size>& calc2(const Matrix<1,hidden_size>& X) noexcept
    {
        layer2.equals_a_dot_bt(X, syn);
        layer2.add(bias_out);
        layer2.apply_sigmoid();
        return layer2;
    }

    inline Matrix<1,hidden_size>& get_delta1() noexcept
    {
        return delta1;
    }

    inline Matrix<1,input_size>& get_delta2() noexcept
    {
        return delta2;
    }

    inline const Matrix<1,hidden_size>& get_output1() noexcept
    {
        return layer1;
    }

    inline const Matrix<1,input_size>& get_output2() noexcept
    {
        return layer2;
    }

    double set_delta2(const Matrix<1,input_size>& Y) noexcept
    {
        delta2.equals_a_sub_b(Y,layer2);
        return delta2.sum_of_squares();
    }

    void propagate_delta2(Matrix<1,hidden_size>& delta_next) noexcept
    {
        delta2.mult_after_func01(layer2);
        delta_next.equals_a_dot_b(delta2, syn);
    }

    void propagate_delta1() noexcept
    {
        delta1.mult_after_func01(layer1);
    }

    void propagate_delta1(Matrix<1,input_size>& delta_next) noexcept
    {
        propagate_delta1();
        delta_next.equals_a_dot_bt(delta1, syn);
    }

    void learn(const double learning_rate, const Matrix<1,input_size>& X1, const Matrix<1,hidden_size>& X2) noexcept
    {
        delta1.mul(learning_rate);
        delta2.mul(learning_rate);

        syn.add_at_dot_b(X1,delta1);
        syn.add_at_dot_b(delta2,X2);
        bias_hidden.add(delta1);
        bias_out.add(delta2);
    }
};

int main()
{
    static constexpr size_t num_visible=54;
    // static constexpr size_t num_hidden=3;
    static constexpr double learning_rate=0.1;
    static constexpr size_t num_iterations=200000;
    std::random_device rd;
    std::mt19937 gen(rd());
    // std::uniform_real_distribution<double> dst(0,1);
    std::uniform_int_distribution<size_t> dst(0,num_visible-1);

    AutoencoderLayer<num_visible, 20> ael1;
    AutoencoderLayer<20, 4> ael2;

    OneHot<num_visible> X;
    double error=0.0;
    for(size_t i=0;i<num_iterations;i++)
    {
        X.set(dst(gen));
        // ael1.calc2(ael2.calc2(ael2.calc1(ael1.calc1(X.get()))));
        ael1.calc1(X.get());
        ael2.calc1(ael1.get_output1());
        ael2.calc2(ael2.get_output1());
        ael1.calc2(ael2.get_output2());

        error+=ael1.set_delta2(X.get()/*Y*/);
        ael1.propagate_delta2(ael2.get_delta2());
        ael2.propagate_delta2(ael2.get_delta1());
        ael2.propagate_delta1(ael1.get_delta1());
        ael1.propagate_delta1();

        ael1.learn(learning_rate, X.get(), ael2.get_output2());
        ael2.learn(learning_rate, ael1.get_output1(), ael2.get_output1());

        if((i+1)%10000==0)
        {
            print(error/10000);
            error=0.0;
        }
    }

    // print(ael1.bias_out);
    // print(ael1.bias_hidden);
    // print(ael2.bias_out);
    // print(ael2.bias_hidden);

    // OneHot<num_visible> X;
    // for(size_t i=0;i<num_iterations;i++)
    // {
    //     X.set(dst(gen));
    //     ael.calc2(ael.calc1(X.get()));

    //     double error=ael.set_delta2(X.get());
    //     print(error);
    //     ael.propagate_delta2(ael.get_delta1());
    //     ael.propagate_delta1();

    //     ael.learn(learning_rate, X.get(), ael.get_output1());
    // }

    // // Matrix<1,num_visible> X;
    // Matrix<num_visible,num_hidden> syn0;
    // Matrix<1,num_hidden> b_h;
    // Matrix<1,num_visible> b_v;
    // syn0.randomize_for_nn(num_visible+1);
    // b_h.randomize_for_nn(num_visible+1);
    // b_v.randomize_for_nn(num_visible+1);
    // Matrix<1,num_hidden> l_h;
    // Matrix<1,num_visible> l_v;
    // Matrix<1,num_hidden> d_h;
    // Matrix<1,num_visible> d_v;

    // OneHot<num_visible> X;
    // double error=0.0;
    // for(size_t i=0;i<num_iterations;i++)
    // {
    //     // X.set(dst(gen));
    //     X.set(dst(gen));
    //     l_h.equals_a_dot_b(X.get(),syn0);
    //     l_h.add(b_h);
    //     l_h.apply_sigmoid();
    //     l_v.equals_a_dot_bt(l_h, syn0);
    //     l_v.add(b_v);
    //     l_v.apply_sigmoid();

    //     d_v.equals_a_sub_b(X.get(),l_v);
    //     if((i+1)%(num_iterations/10)==0)
    //     {
    //         print(error/(num_iterations/10));
    //         error=0.0;
    //     }
    //     error+=d_v.sum_of_squares();
    //     d_v.mult_after_func01(l_v);
    //     d_h.equals_a_dot_b(d_v, syn0);
    //     d_h.mult_after_func01(l_h);

        // d_v.mul(learning_rate);
        // d_h.mul(learning_rate);

        // syn0.add_at_dot_b(X.get(),d_h);
        // syn0.add_at_dot_b(d_v,l_h);
        // b_h.add(d_h);
        // b_v.add(d_v);
    // }
    // print(b_h);
    // print(b_v);
    return 0;
}