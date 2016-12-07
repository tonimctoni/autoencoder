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
public:
    AutoencoderLayer()
    {
        syn.randomize_for_autoencoder();
        bias_hidden.set(0.0);
        bias_out.set(0.0);
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

    void learn(const double learning_rate, const Matrix<batch_size,input_size>& X1, const Matrix<batch_size,hidden_size>& X2) noexcept
    {
        delta1.mul(learning_rate);
        delta2.mul(learning_rate);

        syn.add_at_dot_b(X1,delta1);
        syn.add_at_dot_b(delta2,X2);
        bias_hidden.add_all_rows(delta1);
        bias_out.add_all_rows(delta2);
    }
};

//check if old stuff works
int main()
{
    static constexpr size_t num_visible=54;
    // static constexpr size_t num_hidden=3;
    static constexpr double learning_rate=0.05;
    static constexpr size_t batch_size=10;
    static constexpr size_t num_iterations=2000000/batch_size;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dst(0,num_visible-1);

    AutoencoderLayer<num_visible, 20, batch_size> ael1;
    AutoencoderLayer<20, 4, batch_size> ael2;

    RandomOneHots<batch_size, num_visible> X;
    // OneHot<num_visible> X;
    double error=0.0;
    for(size_t i=0;i<num_iterations;i++)
    {
        X.set_random();
        // X.set(dst(gen));
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

        if((i+1)%(100000/batch_size)==0)
        {
            print(error/10000);
            error=0.0;
        }
    }

    return 0;
}