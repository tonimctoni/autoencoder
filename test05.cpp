#include "matrix.hpp"
#include "mystuff.hpp"
// #include <chrono>
#include <random>
#include <unordered_map>
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

//.5
int main()
{
    static constexpr size_t max_word_size=15;
    static constexpr size_t allowed_char_amount=26;
    static constexpr size_t num_visible=max_word_size*allowed_char_amount;
    // static constexpr double learning_rate=0.001;
    static constexpr size_t batch_size=100;
    static constexpr size_t num_iterations=2000000/batch_size;
    // static constexpr size_t num_pre_iterations=200000/batch_size;
    static const string index_to_char="abcdefghijklmnopqrstuvwxyz";
    unordered_map<char, size_t> char_to_index;for(size_t i=0;i<index_to_char.size();i++) char_to_index[index_to_char[i]]=i;
    assert(index_to_char.size()==allowed_char_amount and char_to_index.size()==allowed_char_amount);

    auto words=[]()
    {
        string str;
        read_file_to_string("../asoiaf/asoiaf_words.txt", str);
        return split_string(str, "\n");
    }();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dst(0,words.size()-1);
    // std::bernoulli_distribution bern_dst(.3);

    // AutoencoderLayer<num_visible, 20, batch_size> ael;
    AutoencoderLayer<num_visible, 120, batch_size> ael1;
    AutoencoderLayer<120, 20, batch_size> ael2;
    // AutoencoderLayer<100, 40, batch_size> ael3;
    Matrix<batch_size,num_visible> X;

    // //Pretraining
    // double error=0.0;
    // for(size_t iteration=0;iteration<num_pre_iterations;iteration++)
    // {
    //     double learning_rate=0.01*pow(0.9772372209558107, ((iteration*batch_size)/10000.));
    //     X.set(0.0);
    //     for(size_t k=0;k<batch_size;k++)
    //     {
    //         const auto& word=words[dst(gen)];
    //         for(size_t i=0;i<word.size();i++)
    //         {
    //             X[k][i*allowed_char_amount+char_to_index[word[i]]]=1.0;
    //         }
    //     }

    //     ael1.calc1(X);
    //     ael1.calc2(ael1.get_output1());

    //     error+=ael1.set_delta2(X);
    //     ael1.propagate_delta2(ael1.get_delta1());
    //     ael1.propagate_delta1();

    //     ael1.learn(learning_rate, X, ael1.get_output1());

    //     if((iteration+1)%(10000/batch_size)==0)
    //     {
    //         print("Pretraining1", error/10000, learning_rate);
    //         error=0.0;
    //     }
    // }

    // error=0.0;
    // for(size_t iteration=0;iteration<num_pre_iterations;iteration++)
    // {
    //     double learning_rate=0.01*pow(0.9772372209558107, ((iteration*batch_size)/10000.));
    //     X.set(0.0);
    //     for(size_t k=0;k<batch_size;k++)
    //     {
    //         const auto& word=words[dst(gen)];
    //         for(size_t i=0;i<word.size();i++)
    //         {
    //             X[k][i*allowed_char_amount+char_to_index[word[i]]]=1.0;
    //         }
    //     }

    //     ael1.calc1(X);
    //     ael2.calc1(ael1.get_output1());
    //     ael2.calc2(ael2.get_output1());

    //     error+=ael2.set_delta2(ael1.get_output1());
    //     ael2.propagate_delta2(ael2.get_delta1());
    //     ael2.propagate_delta1();

    //     ael2.learn(learning_rate, ael1.get_output1(), ael2.get_output1());

    //     if((iteration+1)%(10000/batch_size)==0)
    //     {
    //         print("Pretraining2", error/10000, learning_rate);
    //         error=0.0;
    //     }
    // }

    double error=0.0;
    for(size_t iteration=0;iteration<num_iterations;iteration++)
    {
        double learning_rate=0.02*pow(0.9772372209558107, ((iteration*batch_size)/10000.));
        X.set(0.0);
        for(size_t k=0;k<batch_size;k++)
        {
            const auto& word=words[dst(gen)];
            for(size_t i=0;i<word.size();i++)
            {
                X[k][i*allowed_char_amount+char_to_index[word[i]]]=1.0;
            }
        }

        // ael1.calc1(X);
        // ael2.calc1(ael1.get_output1());
        // ael3.calc1(ael2.get_output1());
        // ael3.calc2(ael3.get_output1());
        // ael2.calc2(ael3.get_output2());
        // ael1.calc2(ael2.get_output2());

        // error+=ael1.set_delta2(X);
        // ael1.propagate_delta2(ael2.get_delta2());
        // ael2.propagate_delta2(ael3.get_delta2());
        // ael3.propagate_delta2(ael3.get_delta1());
        // ael3.propagate_delta1(ael2.get_delta1());
        // ael2.propagate_delta1(ael1.get_delta1());
        // ael1.propagate_delta1();

        // ael1.learn(learning_rate, X, ael2.get_output2());
        // ael2.learn(learning_rate, ael1.get_output1(), ael3.get_output2());
        // ael3.learn(learning_rate, ael2.get_output1(), ael3.get_output1());

        ael1.calc1(X);
        ael2.calc1(ael1.get_output1());
        ael2.calc2(ael2.get_output1());
        ael1.calc2(ael2.get_output2());

        error+=ael1.set_delta2(X);
        ael1.propagate_delta2(ael2.get_delta2());
        ael2.propagate_delta2(ael2.get_delta1());
        ael2.propagate_delta1(ael1.get_delta1());
        ael1.propagate_delta1();

        ael1.learn(learning_rate, X, ael2.get_output2());
        ael2.learn(learning_rate, ael1.get_output1(), ael2.get_output1());

        // ael.calc1(X);
        // ael.calc2(ael.get_output1());

        // error+=ael.set_delta2(X);
        // ael.propagate_delta2(ael.get_delta1());
        // ael.propagate_delta1();

        // ael.learn(learning_rate, X, ael.get_output1());

        if((iteration+1)%(10000/batch_size)==0)
        {
            print(error/10000, learning_rate);
            error=0.0;
        }
    }

    return 0;
}