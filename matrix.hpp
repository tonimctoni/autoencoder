#ifndef __MYMATRIX78256__IAMTONI__
#define __MYMATRIX78256__IAMTONI__
#include <array>
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include "mystuff.hpp"

template<unsigned long M, unsigned long N>
class Matrix: public std::array<std::array<double,N>,M>
{
private:
public:
    /// Constructors
    /// ################################################################################################
    Matrix()=default;

    Matrix(std::initializer_list<std::initializer_list<double>> init)
    {
        assert(init.size()==M);

        int i=0;
        for(auto &row:init)
        {
            assert(row.size()==N);
            int j=0;
            for(auto e:row)
            {
                (*this)[i][j]=e;
                j++;
            }
            i++;
        }
    }

    Matrix(double s) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=s;
    }

    Matrix(const Matrix &other_arr, std::mt19937 &gen, std::bernoulli_distribution &dst) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                if(dst(gen))(*this)[i][j]=other_arr[i][j];
                else        (*this)[i][j]=0.0;
    }

    Matrix(const Matrix &other_arr) noexcept
    {
        std::cout << "()COPY ALERT !!!" << std::endl;
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=other_arr[i][j];
    }

    inline Matrix &operator=(const Matrix &other_arr) noexcept
    {
        std::cout << "=COPY ALERT !!!" << std::endl;
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=other_arr[i][j];

        return *this;
    }

    Matrix(Matrix &&other_arr) noexcept
    {
        std::cout << "()MOVE ALERT !!!" << std::endl;
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=other_arr[i][j];
    }

    inline Matrix &operator=(Matrix &&other_arr) noexcept
    {
        std::cout << "=MOVE ALERT !!!" << std::endl;
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=other_arr[i][j];

        return *this;
    }

    /// Operator Overload
    /// ################################################################################################
    inline bool operator==(const Matrix& rhs) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                if((*this)[i][j]!=rhs[i][j]) return false;

        return true;
    }

    inline bool operator!=(const Matrix& rhs) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                if((*this)[i][j]!=rhs[i][j]) return true;

        return false;
    }

    inline friend std::ostream& operator<< (std::ostream &out, const Matrix &matrix) noexcept
    {
        for(const auto &row:matrix)
        {
            for(const auto &element:row)
            {
                out << element << ", ";
            }
            out << std::endl;
        }
        return out;
    }

    inline Matrix& operator+=(const Matrix& rhs) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]+=rhs[i][j];

        return *this;
    }

    inline Matrix& operator-=(const Matrix& rhs) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]-=rhs[i][j];

        return *this;
    }

    inline Matrix& operator*=(const Matrix& rhs) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]*=rhs[i][j];

        return *this;
    }

    inline Matrix& operator/=(const Matrix& rhs) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]/=rhs[i][j];

        return *this;
    }

    inline Matrix& operator+=(const double rhs) noexcept
    {
        for(auto &row:*this)
            for(auto &element:row)
                element+=rhs;

        return *this;
    }

    inline Matrix& operator-=(const double rhs) noexcept
    {
        for(auto &row:*this)
            for(auto &element:row)
                element-=rhs;

        return *this;
    }

    inline Matrix& operator*=(const double rhs) noexcept
    {
        for(auto &row:*this)
            for(auto &element:row)
                element*=rhs;

        return *this;
    }

    inline Matrix& operator/=(const double rhs) noexcept
    {
        for(auto &row:*this)
            for(auto &element:row)
                element/=rhs;

        return *this;
    }

    /// Same as operator overload but with normal methods (also no return) (also set, which is like operator=)
    /// ################################################################################################
    inline void set(const Matrix &rhs) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=rhs[i][j];
    }

    inline void add(const Matrix& rhs) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]+=rhs[i][j];
    }

    inline void sub(const Matrix& rhs) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]-=rhs[i][j];
    }

    inline void mul(const Matrix& rhs) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]*=rhs[i][j];
    }

    inline void div(const Matrix& rhs) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]/=rhs[i][j];
    }

    inline void set(const double &rhs) noexcept
    {
        for(auto &row:*this)
            for(auto &element:row)
                element=rhs;
    }

    inline void add(const double rhs) noexcept
    {
        for(auto &row:*this)
            for(auto &element:row)
                element+=rhs;
    }

    inline void sub(const double rhs) noexcept
    {
        for(auto &row:*this)
            for(auto &element:row)
                element-=rhs;
    }

    inline void mul(const double rhs) noexcept
    {
        for(auto &row:*this)
            for(auto &element:row)
                element*=rhs;
    }

    inline void div(const double rhs) noexcept
    {
        for(auto &row:*this)
            for(auto &element:row)
                element/=rhs;
    }

    /// Set matrix to dot product of
    /// ################################################################################################
    template<unsigned long L>//user should make sure that this!=&a and this!=&b
    inline void equals_a_dot_b(const Matrix<M,L>& a, const Matrix<L,N>& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
            {
                double acc=0.0;
                for(size_t k=0;k<L;k++) acc+=a[i][k]*b[k][j];
                (*this)[i][j]=acc;
            }
    }

    template<unsigned long L>//user should make sure that this!=&a and this!=&b
    inline void equals_a_dot_bt(const Matrix<M,L>& a, const Matrix<N,L>& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
            {
                double acc=0.0;
                for(size_t k=0;k<L;k++) acc+=a[i][k]*b[j][k];
                (*this)[i][j]=acc;
            }
    }

    template<unsigned long L>//user should make sure that this!=&a and this!=&b
    inline void equals_at_dot_b(const Matrix<L,M>& a, const Matrix<L,N>& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
            {
                double acc=0.0;
                for(size_t k=0;k<L;k++) acc+=a[k][i]*b[k][j];
                (*this)[i][j]=acc;
            }
    }

    /// Add to matrix the dot product of
    /// ################################################################################################
    template<unsigned long L>//user should make sure that this!=&a and this!=&b
    inline void add_a_dot_b(const Matrix<M,L>& a, const Matrix<L,N>& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
            {
                double acc=0.0;
                for(size_t k=0;k<L;k++) acc+=a[i][k]*b[k][j];
                (*this)[i][j]+=acc;
            }
    }

    template<unsigned long L>//user should make sure that this!=&a and this!=&b
    inline void add_a_dot_bt(const Matrix<M,L>& a, const Matrix<N,L>& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
            {
                double acc=0.0;
                for(size_t k=0;k<L;k++) acc+=a[i][k]*b[j][k];
                (*this)[i][j]+=acc;
            }
    }

    template<unsigned long L>//user should make sure that this!=&a and this!=&b
    inline void add_at_dot_b(const Matrix<L,M>& a, const Matrix<L,N>& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
            {
                double acc=0.0;
                for(size_t k=0;k<L;k++) acc+=a[k][i]*b[k][j];
                (*this)[i][j]+=acc;
            }
    }

    /// Subtract from matrix the dot product of
    /// ################################################################################################
    template<unsigned long L>//user should make sure that this!=&a and this!=&b
    inline void sub_a_dot_b(const Matrix<M,L>& a, const Matrix<L,N>& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
            {
                double acc=0.0;
                for(size_t k=0;k<L;k++) acc+=a[i][k]*b[k][j];
                (*this)[i][j]-=acc;
            }
    }

    template<unsigned long L>//user should make sure that this!=&a and this!=&b
    inline void sub_a_dot_bt(const Matrix<M,L>& a, const Matrix<N,L>& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
            {
                double acc=0.0;
                for(size_t k=0;k<L;k++) acc+=a[i][k]*b[j][k];
                (*this)[i][j]-=acc;
            }
    }

    template<unsigned long L>//user should make sure that this!=&a and this!=&b
    inline void sub_at_dot_b(const Matrix<L,M>& a, const Matrix<L,N>& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
            {
                double acc=0.0;
                for(size_t k=0;k<L;k++) acc+=a[k][i]*b[k][j];
                (*this)[i][j]-=acc;
            }
    }
    /// Set to, add, or subtract the result of a+b or a-b
    /// ################################################################################################
    inline void equals_a_add_b(const Matrix& a, const Matrix& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=a[i][j]+b[i][j];
    }

    inline void add_a_add_b(const Matrix& a, const Matrix& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]+=a[i][j]+b[i][j];
    }

    inline void sub_a_add_b(const Matrix& a, const Matrix& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]-=a[i][j]+b[i][j];
    }

    inline void equals_a_sub_b(const Matrix& a, const Matrix& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=a[i][j]-b[i][j];
    }

    inline void add_a_sub_b(const Matrix& a, const Matrix& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]+=a[i][j]-b[i][j];
    }

    inline void sub_a_sub_b(const Matrix& a, const Matrix& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]-=a[i][j]-b[i][j];
    }

    /// Set matrix to dot product of (...) using only one row of the first matrix
    /// ################################################################################################
    template<unsigned long L, unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void equals_row_of_a_dot_b(const Matrix<A,L>& a, const size_t row_of_a, const Matrix<L,N>& b) noexcept
    {
        static_assert(M==1,"M is not 1");
        for(size_t j=0;j<N;j++)
        {
            double acc=0.0;
            for(size_t k=0;k<L;k++) acc+=a[row_of_a][k]*b[k][j];
            (*this)[0][j]=acc;
        }
    }

    template<unsigned long L, unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void equals_row_of_at_dot_b(const Matrix<L,A>& a, const size_t row_of_a, const Matrix<L,N>& b) noexcept
    {
        static_assert(M==1,"M is not 1");
        for(size_t j=0;j<N;j++)
        {
            double acc=0.0;
            for(size_t k=0;k<L;k++) acc+=a[k][row_of_a]*b[k][j];
            (*this)[0][j]=acc;
        }
    }

    template<unsigned long L, unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void equals_row_of_a_dot_bt(const Matrix<A,L>& a, const size_t row_of_a, const Matrix<N,L>& b) noexcept
    {
        static_assert(M==1,"M is not 1");
        for(size_t j=0;j<N;j++)
        {
            double acc=0.0;
            for(size_t k=0;k<L;k++) acc+=a[row_of_a][k]*b[j][k];
            (*this)[0][j]=acc;
        }
    }

    /// Add to matrix dot product of (...) using only one row of the first matrix
    /// ################################################################################################
    template<unsigned long L, unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void add_row_of_a_dot_b(const Matrix<A,L>& a, const size_t row_of_a, const Matrix<L,N>& b) noexcept
    {
        static_assert(M==1,"M is not 1");
        for(size_t j=0;j<N;j++)
        {
            double acc=0.0;
            for(size_t k=0;k<L;k++) acc+=a[row_of_a][k]*b[k][j];
            (*this)[0][j]+=acc;
        }
    }

    template<unsigned long L, unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void add_row_of_at_dot_b(const Matrix<L,A>& a, const size_t row_of_a, const Matrix<L,N>& b) noexcept
    {
        static_assert(M==1,"M is not 1");
        for(size_t j=0;j<N;j++)
        {
            double acc=0.0;
            for(size_t k=0;k<L;k++) acc+=a[k][row_of_a]*b[k][j];
            (*this)[0][j]+=acc;
        }
    }

    template<unsigned long L, unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void add_row_of_a_dot_bt(const Matrix<A,L>& a, const size_t row_of_a, const Matrix<N,L>& b) noexcept
    {
        static_assert(M==1,"M is not 1");
        for(size_t j=0;j<N;j++)
        {
            double acc=0.0;
            for(size_t k=0;k<L;k++) acc+=a[row_of_a][k]*b[j][k];
            (*this)[0][j]+=acc;
        }
    }

    /// Subtract matrix dot product of (...) using only one row of the first matrix
    /// ################################################################################################
    template<unsigned long L, unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void sub_row_of_a_dot_b(const Matrix<A,L>& a, const size_t row_of_a, const Matrix<L,N>& b) noexcept
    {
        static_assert(M==1,"M is not 1");
        for(size_t j=0;j<N;j++)
        {
            double acc=0.0;
            for(size_t k=0;k<L;k++) acc+=a[row_of_a][k]*b[k][j];
            (*this)[0][j]-=acc;
        }
    }

    template<unsigned long L, unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void sub_row_of_at_dot_b(const Matrix<L,A>& a, const size_t row_of_a, const Matrix<L,N>& b) noexcept
    {
        static_assert(M==1,"M is not 1");
        for(size_t j=0;j<N;j++)
        {
            double acc=0.0;
            for(size_t k=0;k<L;k++) acc+=a[k][row_of_a]*b[k][j];
            (*this)[0][j]-=acc;
        }
    }

    template<unsigned long L, unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void sub_row_of_a_dot_bt(const Matrix<A,L>& a, const size_t row_of_a, const Matrix<N,L>& b) noexcept
    {
        static_assert(M==1,"M is not 1");
        for(size_t j=0;j<N;j++)
        {
            double acc=0.0;
            for(size_t k=0;k<L;k++) acc+=a[row_of_a][k]*b[j][k];
            (*this)[0][j]-=acc;
        }
    }

    /// Set to, add to, or subtract from matrix dot product of (...) using only one row of the first matrix after it has been transposed
    /// ################################################################################################
    template<unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void equals_row_of_a_t_dot_b(const Matrix<A,M>& a, const size_t row_of_a, const Matrix<1,N>& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=a[row_of_a][i]*b[0][j];
    }

    template<unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void add_row_of_a_t_dot_b(const Matrix<A,M>& a, const size_t row_of_a, const Matrix<1,N>& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]+=a[row_of_a][i]*b[0][j];
    }

    template<unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void sub_row_of_a_t_dot_b(const Matrix<A,M>& a, const size_t row_of_a, const Matrix<1,N>& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]-=a[row_of_a][i]*b[0][j];
    }

    /// Set to, add to, or subtract from matrix the addition or difference of (...) using only one row of the first matrix
    /// ################################################################################################
    template<unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void equals_row_of_a_add_b(const Matrix<A,N>& a, const size_t row_of_a, const Matrix<1,N>& b) noexcept
    {
        static_assert(M==1,"M is not 1");
        for(size_t j=0;j<N;j++)
            (*this)[0][j]=a[row_of_a][j]+b[0][j];
    }

    template<unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void equals_row_of_a_sub_b(const Matrix<A,N>& a, const size_t row_of_a, const Matrix<1,N>& b) noexcept
    {
        static_assert(M==1,"M is not 1");
        for(size_t j=0;j<N;j++)
            (*this)[0][j]=a[row_of_a][j]-b[0][j];
    }

    template<unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void add_row_of_a_add_b(const Matrix<A,N>& a, const size_t row_of_a, const Matrix<1,N>& b) noexcept
    {
        static_assert(M==1,"M is not 1");
        for(size_t j=0;j<N;j++)
            (*this)[0][j]+=a[row_of_a][j]+b[0][j];
    }

    template<unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void add_row_of_a_sub_b(const Matrix<A,N>& a, const size_t row_of_a, const Matrix<1,N>& b) noexcept
    {
        static_assert(M==1,"M is not 1");
        for(size_t j=0;j<N;j++)
            (*this)[0][j]+=a[row_of_a][j]-b[0][j];
    }

    template<unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void sub_row_of_a_add_b(const Matrix<A,N>& a, const size_t row_of_a, const Matrix<1,N>& b) noexcept
    {
        static_assert(M==1,"M is not 1");
        for(size_t j=0;j<N;j++)
            (*this)[0][j]-=a[row_of_a][j]+b[0][j];
    }

    template<unsigned long A>//user should make sure that this!=&a and this!=&b
    inline void sub_row_of_a_sub_b(const Matrix<A,N>& a, const size_t row_of_a, const Matrix<1,N>& b) noexcept
    {
        static_assert(M==1,"M is not 1");
        for(size_t j=0;j<N;j++)
            (*this)[0][j]-=a[row_of_a][j]-b[0][j];
    }


    /// Several special methods, used for neural networks
    /// ################################################################################################
    inline void apply_sigmoid() noexcept
    {
        for(auto &row:*this)
            for(auto &element:row)
                element=1.0/(1.0+std::exp(-element));
    }

    inline void apply_tanh() noexcept
    {
        for(auto &row:*this)
            for(auto &element:row)
                element=tanh(element);
    }

    inline void apply_softmax() noexcept
    {
        double sum=0.0;
        for(auto &row:*this) for(auto &element:row)
        {
            element=exp(element);
            sum+=element;
        }
        for(auto &row:*this)
            for(auto &element:row)
                element/=sum;
    }

    inline void apply_rectifier() noexcept
    {
        for(auto &row:*this)
            for(auto &element:row)
                if(element<0) element=0;
                // element=element<0?0:element;
    }

    inline void mult_after_func01(const Matrix &a) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]*=(a[i][j])*(1.0-a[i][j]);
    }

    inline void mult_after_func02(const Matrix &a) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]*=1-(a[i][j])*(a[i][j]);
                // (*this)[i][j]*=2*(a[i][j])*((a[i][j])*(a[i][j]) - 1);
    }

    inline void mult_after_func03(const Matrix &a) noexcept //Same as 01, but it will be used for softmax
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]*=(a[i][j])*(1.0-a[i][j]);
    }

    inline void mult_after_func04(const Matrix &a) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                if(a[i][j]<=0) (*this)[i][j]=0;
                // (*this)[i][j]*=a[i][j];
    }

    inline void randomize_for_nn(std::normal_distribution<double>::result_type scal) noexcept
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dst(0,1.0/(sqrt(scal)));
        for(auto &row:*this)
            for(auto &element:row)
                element=dst(gen);
    }

    inline void randomize_for_nn() noexcept
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dst(0,1.0/(sqrt(M)));
        for(auto &row:*this)
            for(auto &element:row)
                element=dst(gen);
    }

    inline void randomize_for_autoencoder() noexcept //same as for relu :)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        double a=4.*sqrt(6./(M+N));
        std::uniform_real_distribution<double> dst(-a,a);
        for(auto &row:*this)
            for(auto &element:row)
                element=dst(gen);
    }

    inline void randomize_for_relu_nn() noexcept
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        double a=sqrt(12.0/(M+N));
        std::uniform_real_distribution<double> dst(-a,a);
        for(auto &row:*this)
            for(auto &element:row)
                element=dst(gen);
    }

    inline void equals_a_mul_b_add_c_mul_d(const Matrix<M,N>& a, const Matrix<M,N>& b, const Matrix<M,N>& c, const Matrix<M,N>& d) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=a[i][j]*b[i][j]+c[i][j]*d[i][j];
    }

    inline void equals_a_mul_b(const Matrix<M,N>& a, const Matrix<M,N>& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]=a[i][j]*b[i][j];
    }

    inline double sum() noexcept
    {
        double sum=0.0;
        for(auto &row:*this)
            for(auto &element:row)
                sum+=element;
        return sum;
    }

    inline double sum_of_squares() noexcept
    {
        double sum=0.0;
        for(auto &row:*this)
            for(auto &element:row)
                sum+=element*element;
        return sum;
    }

    template<unsigned long A, unsigned long B, unsigned long C, unsigned long D>
    inline void set_from_four(const Matrix<1, A> &a, const Matrix<1, B> &b, const Matrix<1, C> &c, const Matrix<1, D> &d) noexcept
    {
        static_assert(A+B+C+D==N, "Sizes do not add up.");
        static_assert(M==1, "M is not 1");

        size_t i=0;
        for(const auto e:a[0])
        {
            (*this)[0][i]=e;
            i++;
        }
        for(const auto e:b[0])
        {
            (*this)[0][i]=e;
            i++;
        }
        for(const auto e:c[0])
        {
            (*this)[0][i]=e;
            i++;
        }
        for(const auto e:d[0])
        {
            (*this)[0][i]=e;
            i++;
        }
    }

    template<unsigned long A, unsigned long B, unsigned long C>
    inline void set_from_three(const Matrix<1, A> &a, const Matrix<1, B> &b, const Matrix<1, C> &c) noexcept
    {
        static_assert(A+B+C==N, "Sizes do not add up.");
        static_assert(M==1, "M is not 1");

        size_t i=0;
        for(const auto e:a[0])
        {
            (*this)[0][i]=e;
            i++;
        }
        for(const auto e:b[0])
        {
            (*this)[0][i]=e;
            i++;
        }
        for(const auto e:c[0])
        {
            (*this)[0][i]=e;
            i++;
        }
    }

    template<unsigned long A>
    inline void set_from_part(const Matrix<1, A> &a, size_t index_from)
    {
        assert(N<=A-index_from);

        size_t j=index_from;
        for(auto &e:(*this)[0])
        {
            e=a[0][j];
            j++;
        }
    }

    template<unsigned long A>
    inline void add_from_part(const Matrix<1, A> &a, size_t index_from)
    {
        assert(N<=A-index_from);

        size_t j=index_from;
        for(auto &e:(*this)[0])
        {
            e+=a[0][j];
            j++;
        }
    }

    // inline void add_factor_mul_a_mul_a(const double factor, const Matrix& a) noexcept
    // {
    //     for(size_t i=0;i<M;i++)
    //         for(size_t j=0;j<N;j++)
    //             (*this)[i][j]+=factor*a[i][j]*a[i][j];
    // }

    inline void add_a_mul_rate_div_sqrt_b(const Matrix& a, const double rate, const Matrix& b) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]+=a[i][j]*(rate/sqrt(b[i][j]+1e-8));
    }

    inline void add_factor_mul_a(const double factor, const Matrix& a) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]+=factor*a[i][j];
    }

    inline void add_factor_mul_a_squared(const double factor, const Matrix& a) noexcept
    {
        for(size_t i=0;i<M;i++)
            for(size_t j=0;j<N;j++)
                (*this)[i][j]+=factor*a[i][j]*a[i][j];
    }

    // template<unsigned long L>//user should make sure that this!=&a and this!=&b
    // inline void add_at_dot_bt(const Matrix<L,M>& a, const Matrix<N,L>& b) noexcept
    // {
    //     for(size_t i=0;i<M;i++)
    //         for(size_t j=0;j<N;j++)
    //         {
    //             double acc=0.0;
    //             for(size_t k=0;k<L;k++) acc+=a[k][i]*b[j][k];
    //             (*this)[i][j]+=acc;
    //         }
    // }

    /// File IO stuff
    /// ################################################################################################
    inline void to_file(std::ofstream &out) const
    {
        assert(out.good());
        out << M << "\t" << N << std::endl;
        for(const auto &row:*this)
            for(const auto &element:row)
                out << element << std::endl;
        assert(not out.fail());
    }

    inline void from_file(std::ifstream &in)
    {
        assert(in.good());
        unsigned long m, n;
        in >> m >> n;
        assert(m==M and n==N);
        for(auto &row:*this)
            for(auto &element:row)
                in >> element;
        assert(not in.fail());
    }

    inline void to_bin_file(std::ofstream &out) const
    {
        assert(out.good());
        unsigned long MM=M, NN=N;
        out.write((const char*) &MM, sizeof(unsigned long));
        out.write((const char*) &NN, sizeof(unsigned long));
        for(const auto &row:*this)
            for(const auto &element:row)
                out.write((const char*)&element, sizeof(double));
        assert(not out.fail());
    }

    inline void from_bin_file(std::ifstream &in)
    {
        assert(in.good());
        unsigned long m, n;
        in.read((char*)&m, sizeof(unsigned long));
        in.read((char*)&n, sizeof(unsigned long));
        assert(m==M and n==N);
        for(auto &row:*this)
            for(auto &element:row)
                in.read((char*)&element, sizeof(double));
        assert(not in.fail());
    }
};

template<unsigned long M, unsigned long N>
inline void update_weight_with_ms(Matrix<M,N> &weights, const Matrix<M,N> &ms, const Matrix<M,N> &gradient, const double learning_rate) noexcept
{
    for(size_t i=0;i<M;i++)
        for(size_t j=0;j<N;j++)
        {
            weights[i][j]+=(gradient[i][j]*learning_rate)/sqrt(ms[i][j]+1e-8);//+1e-8
        }
}

template<unsigned long M, unsigned long N>
inline void update_weight_with_adam(Matrix<M,N> &weights, const Matrix<M,N> &ms, const Matrix<M,N> &mns, const double learning_rate, const double decay1, const double decay2) noexcept
{
    for(size_t i=0;i<M;i++)
        for(size_t j=0;j<N;j++)
        {
            double v=ms[i][j]/(1-decay1);
            double m=mns[i][j]/(1-decay2);
            // weights[i][j]+=(learning_rate/sqrt(v+1e-8))*m;
            weights[i][j]+=(learning_rate*m)/sqrt(v+1e-8);
        }
}

template<unsigned long M, unsigned long N>
inline void update_weight_momentum(Matrix<M,N> &weights, Matrix<M,N> &momentums, const Matrix<M,N> &ms, const Matrix<M,N> &gradient, const double learning_rate) noexcept
{
    for(size_t i=0;i<M;i++)
        for(size_t j=0;j<N;j++)
        {
            double weight_change=(gradient[i][j]*learning_rate)/sqrt(ms[i][j]+1e-8);//+1e-8
            weights[i][j]+=weight_change;
            momentums[i][j]+=weight_change;
        }
}

template<unsigned long M, unsigned long N>
inline void update_weight_momentum(Matrix<M,N> &weights, Matrix<M,N> &momentums, const Matrix<M,N> &gradient, const double learning_rate) noexcept
{
    for(size_t i=0;i<M;i++)
        for(size_t j=0;j<N;j++)
        {
            double weight_change=(gradient[i][j]*learning_rate);
            weights[i][j]+=weight_change;
            momentums[i][j]+=weight_change;
        }
}

void test_matrix()
{
    //Constructor
    assert((Matrix<2,3>{{2.0,2.0,2.0},{2.0,2.0,2.0}}==Matrix<2,3>(2.0)));

    //Operator overload
    assert((Matrix<2,3>(1.0)==Matrix<2,3>(1.0) && Matrix<2,3>(1.0)!=Matrix<2,3>(3.0)));

    //......
}

template<unsigned long mat_size>
class OneHot
{
private:
    size_t hot_index;
    Matrix<1, mat_size> X;
public:
    OneHot()noexcept:hot_index(0), X(0.0)
    {
    }

    inline void set(size_t index)
    {
        assert(index<mat_size);
        X[0][hot_index]=0.0;
        hot_index=index;
        X[0][hot_index]=1.0;
    }

    inline void reset() noexcept
    {
        X[0][hot_index]=0.0;
    }

    inline const Matrix<1, mat_size>& get() const noexcept
    {
        return X;
    }

    // inline const Matrix<1,mat_size> get_noisy(double sigma_squared) const noexcept
    // {
    //     Matrix<1,mat_size> ret;
    //     ret.set(X);
    //     static std::random_device rd;
    //     static std::mt19937 gen(rd());
    //     static std::normal_distribution<double> dst(0,sigma_squared);
    //     for(auto &row:ret)
    //         for(auto &element:row)
    //             {
    //                 element+=dst(gen);
    //                 if(element<0.0) element=0.0;
    //                 if(element>1.0) element=1.0;
    //             }
    //     return ret;
    // }
};

#endif