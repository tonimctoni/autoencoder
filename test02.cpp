// #include "perceptron_timeseries_class.hpp"
// #include "LSTM_class.hpp"
// #include "softmax_timeseries_class.hpp"
// #include <string>
// #include <vector>
// #include <array>
// #include <unordered_map>
// #include <memory>
// #include <ctime>
// #include "matrix.hpp"
#include "mystuff.hpp"
#include <chrono>
#include <random>
#include <armadillo>
using namespace std;
using namespace arma;

template<unsigned long input_size, unsigned long hidden_size>
class AutoencoderLayer
{
private:
    //Parameters
    mat::fixed<input_size,hidden_size> syn;
    rowvec::fixed<hidden_size> bias_hidden;
    rowvec::fixed<input_size> bias_out;
    //State
    rowvec::fixed<hidden_size> layer1;
    rowvec::fixed<input_size> layer2;
    rowvec::fixed<hidden_size> delta1;
    rowvec::fixed<input_size> delta2;
public:
    AutoencoderLayer()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        double a=4.*sqrt(6./(input_size+hidden_size));
        std::uniform_real_distribution<double> dst(-a,a);
        for(auto &element:syn) element=dst(gen);
        // syn.transform([&dst,&gen](double x){return dst(gen);});

        bias_hidden.fill(0.0);
        bias_out.fill(0.0);
    }

    const rowvec::fixed<hidden_size>& calc1(const rowvec::fixed<input_size>& X) noexcept
    {
        layer1=X*syn;
        layer1+=bias_hidden;
        layer1.transform([](double x){return 1.0/(1.0+std::exp(-x));});
        // layer1.equals_a_dot_b(X,syn);
        // layer1.add(bias_hidden);
        // layer1.apply_sigmoid();
        return layer1;
    }

    const rowvec::fixed<input_size>& calc2(const rowvec::fixed<hidden_size>& X) noexcept
    {
        layer2=X*syn.t();
        layer2+=bias_out;
        layer2.transform([](double x){return 1.0/(1.0+std::exp(-x));});
        // layer2.equals_a_dot_bt(X, syn);
        // layer2.add(bias_out);
        // layer2.apply_sigmoid();
        return layer2;
    }

    inline rowvec::fixed<hidden_size>& get_delta1() noexcept
    {
        return delta1;
    }

    inline rowvec::fixed<input_size>& get_delta2() noexcept
    {
        return delta2;
    }

    inline const rowvec::fixed<hidden_size>& get_output1() noexcept
    {
        return layer1;
    }

    inline const rowvec::fixed<input_size>& get_output2() noexcept
    {
        return layer2;
    }

    double set_delta2(const rowvec::fixed<input_size>& Y) noexcept
    {
        delta2=Y-layer2;
        // delta2.equals_a_sub_b(Y,layer2);
        double sum=0.0;
        for(const auto &element:delta2) sum+=element*element;
        return sum;
    }

    void propagate_delta2(rowvec::fixed<hidden_size>& delta_next) noexcept
    {
        delta2%=layer2%(1.0-layer2);
        delta_next=delta2*syn;
        // delta2.mult_after_func01(layer2);
        // delta_next.equals_a_dot_b(delta2, syn);
    }

    void propagate_delta1(rowvec::fixed<input_size>& delta_next) noexcept
    {
        delta1%=layer1%(1.0-layer1);
        delta_next=delta1*syn.t();
        // delta1.mult_after_func01(layer1);
        // delta_next.equals_a_dot_bt(delta1, syn);
    }

    void propagate_delta1() noexcept
    {
        delta1%=layer1%(1.0-layer1);
        // delta1.mult_after_func01(layer1);
    }

    void learn(const double learning_rate, const rowvec::fixed<input_size>& X1, const rowvec::fixed<hidden_size>& X2) noexcept
    {
        delta1*=learning_rate;
        delta2*=learning_rate;
        syn+=X1.t()*delta1;
        syn+=delta2.t()*X2;
        bias_hidden+=delta1;
        bias_out+=delta2;
        // delta1.mul(learning_rate);
        // delta2.mul(learning_rate);
        // syn.add_at_dot_b(X1,delta1);
        // syn.add_at_dot_b(delta2,X2);
        // bias_hidden.add(delta1);
        // bias_out.add(delta2);
    }
};

template<unsigned long mat_size>
class OneHot
{
private:
    size_t hot_index;
    rowvec::fixed<mat_size> X;
public:
    OneHot()noexcept:hot_index(0)
    {
        X.fill(0.0);
    }

    inline void set(size_t index)
    {
        assert(index<mat_size);
        X/*[0]*/[hot_index]=0.0;
        hot_index=index;
        X/*[0]*/[hot_index]=1.0;
    }

    inline void reset() noexcept
    {
        X/*[0]*/[hot_index]=0.0;
    }

    inline const rowvec::fixed<mat_size>& get() const noexcept
    {
        return X;
    }
};

int main()
{
    static constexpr size_t num_visible=48;
    // static constexpr size_t num_hidden=3;
    static constexpr double learning_rate=0.1;
    static constexpr size_t num_iterations=100000;
    std::random_device rd;
    std::mt19937 gen(rd());
    // std::uniform_real_distribution<double> dst(0,1);
    std::uniform_int_distribution<size_t> dst(0,num_visible-1);

    AutoencoderLayer<num_visible, 200> ael1;
    AutoencoderLayer<200, 8/*3 later*/> ael2;

    OneHot<num_visible> X;
    double error=0.0;
    for(size_t i=0;i<num_iterations;i++)
    {
        X.set(dst(gen));

        ael1.calc1(X.get());
        ael2.calc1(ael1.get_output1());
        ael2.calc2(ael2.get_output1());
        ael1.calc2(ael2.get_output2());

        error+=ael1.set_delta2(X.get());
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

    return 0;
}