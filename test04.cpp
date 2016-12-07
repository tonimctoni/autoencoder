#include "matrix.hpp"
#include "mystuff.hpp"
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

//check if old stuff works
int main()
{
    static constexpr size_t num_visible=5;
    // static constexpr size_t num_hidden=3;
    static constexpr double learning_rate=0.1;
    static constexpr size_t batch_size=20;
    static constexpr size_t num_iterations=2000000/batch_size;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dst(0,num_visible-1);

    AutoencoderLayer<num_visible, 40, batch_size> ael1;
    AutoencoderLayer<40, 1, batch_size> ael2;

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