#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>
#include <vector>
#include <queue>

// TODO: Install OPENCV
// #include <opencv2/opencv.hpp>

class Node
{

private:
    void f_hat_cal()
    {
        this->f_hat = this->g_hat + this->h_hat;
    }
    void g_hat_cal()
    {
        this->g_hat = sqrt(pow(this->x - this->start->x, 2) + pow(this->y - this->start->y, 2));
    }
    void h_hat_cal()
    {
        this->h_hat = sqrt(pow(this->x - this->goal->x, 2) + pow(this->y - this->goal->y, 2));
    }

public:
    double x, y;
    double f_hat, g_hat, h_hat; // Estimated costs
    double f, g, h;             // Actual costs
    Node *parent;
    std::vector<Node *> children;
    bool is_expanded; // We might use this
    Node *start;
    Node *goal;
    void f_hat_cal();
    void g_hat_cal();
    void h_hat_cal();
    Node()
    {
        this->x = 0.0;
        this->y = 0.0;
        this->f_hat = 0.0;
        this->g_hat = 0.0;
        this->h_hat = 0.0;
        this->f = 0.0;
        this->g = 0.0;
        this->h = 0.0;
        this->parent = NULL;
        this->is_expanded = false;
    }
    Node(double x, double y)
    {
        this->x = x;
        this->y = y;
    }
    Node(double x, double y, Node *start, Node *goal)
    {
        this->x = x;
        this->y = y;
        this->start = start;
        this->goal = goal;
    }
    Node(double x, double y, double f_hat, double g_hat, double h_hat)
    {
        this->x = x;
        this->y = y;
        this->f_hat = f_hat;
        this->g_hat = g_hat;
        this->h_hat = h_hat;
    }
    Node(double x, double y, double f_hat, double g_hat, double h_hat, double f, double g, double h)
    {
        this->x = x;
        this->y = y;
        this->f_hat = f_hat;
        this->g_hat = g_hat;
        this->h_hat = h_hat;
        this->f = f;
        this->g = g;
        this->h = h;
    }
    Node(double x, double y, double f_hat, double g_hat, double h_hat, double f, double g, double h, Node *parent)
    {
        this->x = x;
        this->y = y;
        this->f_hat = f_hat;
        this->g_hat = g_hat;
        this->h_hat = h_hat;
        this->f = f;
        this->g = g;
        this->h = h;
        this->parent = parent;
    }
    Node(double x, double y, double f_hat, double g_hat, double h_hat, double f, double g, double h, Node *parent, Node *child, Node *start, Node *goal)
    {
        this->x = x;
        this->y = y;
        this->f_hat = f_hat;
        this->g_hat = g_hat;
        this->h_hat = h_hat;
        this->f = f;
        this->g = g;
        this->h = h;
        this->parent = parent;
        this->children.push_back(child);
        this->start = start;
        this->goal = goal;
    }
    void g_hat_cal();
    void h_hat_cal();
    void f_hat_cal();
};

class Bit_star
{
private:
    Node start, goal;
    std::priority_queue<Node> vertex_q;
    std::priority_queue<Node> edge_q;
    double Rbit = 100.0;
    std::vector<Node> unexp_vertex;
    std::vector<Node> x_new;
    std::vector<Node> x_reuse;
    int no_samples = 20;
    std::vector<Node> vert;
    std::vector<Node, Node> edge;
    int dimension = 2;
    std::vector<Node> vsol;
    double cur_cost = std::numeric_limits<double>::infinity();
    double old_cost = cur_cost;
    Eigen::Matrix2Xi map;
    int map_width = map.rows();
    int map_height = map.cols();
    std::vector<Node> x_phs;

public:
    Bit_star(Node start, Node goal, std::string map_path);
    double gt(Node node);
    void generate_phs(){};
    void
};

Bit_star::Bit_star(Node start, Node goal, std::string map_path)
{
    this->start = start;
    this->goal = goal;
    // this->map = cv::imread(map_path, cv::IMREAD_GRAYSCALE);

    this->vert.push_back(start);
    this->unexp_vertex.push_back(start);
    // TODO: Read map from file

    generate_phs();
}

double Bit_star::gt(Node node)
{
    double length = 0.0;

    if (node == this->start)
    {
        return 0.0;
    }
    if (std::find(this->vert.begin(), this->vert.end(), node) == this->vert.end())
    {
        return std::numeric_limits<double>::infinity();
    }
}
