#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>
#include <vector>
#include <queue>

using Point = Eigen::Vector2d;

int main()
{
    Point p1(0.0, 0.0);
    std::tuple<Point, char> foo(p1, 'x');
    auto bar = std::make_tuple("test", 3.1, 14, 'y');

    // std::get<2>(bar) = 100; // access element

    // int myint;
    // char mychar;

    // std::tie(myint, mychar) = foo; // unpack elements
    // std::tie(std::ignore, std::ignore, myint, mychar) = bar; // unpack (with ignore)

    // mychar = std::get<3>(bar);

    // std::get<0>(foo) = std::get<2>(bar);
    // std::get<1>(foo) = mychar;

    std::cout << "foo contains: ";
    std::cout << std::get<0>(foo) << ' ';
    std::cout << std::get<1>(foo) << '\n';

    return 0;
}

// int main()
// {
//     std::vector<std::tuple<Point, Point>> edge;
//     Point p1(0.0, 0.0);
//     Point p2(0.0, 1.0);
//     edge.push_back(std::make_pair(p1, p2));
//     // edge.push_back(std::make_pair(, 3));
//     std::cout << p1 << std::endl;
//     std::cout << edge[0] << std::endl;
//     std::cout << edge[1] << std::endl;
//     return 0;
// }

/*
class point{
    public:
        Eigen::VectorXd x;
        double cost;
};
}


*/