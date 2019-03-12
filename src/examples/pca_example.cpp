#include <iostream>
#include <decomposition/pca.hpp>

#define NUM_COMPONENTS 2

using namespace aegean;
using namespace decomposition;

struct Params {
    struct PCA : public defaults::PCA {
        static constexpr bool scale = true;
    };
};

int main()
{

    /*
        using matlabs hald dataset (e.g., load hald)
        for comparison
    */
    Eigen::MatrixXd ingredients(13, 4);
    ingredients << 7, 26, 6, 60,
        1, 29, 15, 52,
        11, 56, 8, 20,
        11, 31, 8, 47,
        7, 52, 6, 33,
        11, 55, 9, 22,
        3, 71, 17, 6,
        1, 31, 22, 44,
        2, 54, 18, 22,
        21, 47, 4, 26,
        1, 40, 23, 34,
        11, 66, 9, 12,
        10, 68, 8, 12;

    Eigen::MatrixXd target_coefficients(4, 4);
    target_coefficients << -0.0678, -0.6460, 0.5673, 0.5062,
        -0.6785, -0.0200, -0.5440, 0.4933,
        0.0290, 0.7553, 0.4036, 0.5156,
        0.7309, -0.1085, -0.4684, 0.4844;

    PCA<Params> pca;
    Eigen::MatrixXd coef = pca(ingredients);

    std::cout << "Resulting coefficients" << std::endl;
    std::cout << coef << std::endl
              << std::endl;

    std::cout << "MSE from target" << std::endl;
    std::cout << (target_coefficients - coef).colwise().norm() << std::endl
              << std::endl;

    std::cout << "Variance explained per eigenvector" << std::endl;
    std::cout << pca.variance_explained().transpose() << std::endl
              << std::endl;

    Eigen::MatrixXd transformed = pca.transformed(NUM_COMPONENTS);
    std::cout << "Data projected only on the first " << NUM_COMPONENTS << " PCs" << std::endl;
    std::cout << transformed << std::endl
              << std::endl;

    std::cout << "Centered data" << std::endl;
    std::cout << pca.centered_data() << std::endl
              << std::endl;

    Eigen::MatrixXd approx = transformed * pca.eigenvectors(NUM_COMPONENTS).transpose();
    std::cout << "Centered data approximation" << std::endl;
    std::cout << approx << std::endl
              << std::endl;

    std::cout << "Original data approximation" << std::endl;
    std::cout << pca.to_original_scale(approx) << std::endl
              << std::endl;

    return 0;
}