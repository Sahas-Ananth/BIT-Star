#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>
#include <vector>
#include <queue>
#include <thread>

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
    void c_hat_cal()
    {
        this->c_hat = this->g_hat + this->h_hat;
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
    
    double Rbit = 100.0;
    std::vector<Node> unexp_vertex;
    
    
    struct Edge {
        Node source;
        Node target;
        double weight;
    };
    
    int dimension = 2;
    std::vector<Node> vsol;
    double cur_cost = std::numeric_limits<double>::infinity();
    double old_cost = cur_cost;
    Eigen::Matrix2Xi map;
    int map_width = map.rows();
    int map_height = map.cols();
    std::vector<Node> x_phs;
    int dim = 2;
    double ci = std::numeric_limits<double>::infinity();
    double old_ci = 0;

    



public:
    Bit_star::Bit_star(Node start, Node goal, Eigen::MatrixXd map)
    {
    this->start = start;
    this->goal = goal;
    // this->map = cv::imread(map_path, cv::IMREAD_GRAYSCALE);

    this->vert.push_back(start);
    this->unexp_vertex.push_back(start);
    // TODO: Read map from file
    generate_phs();

    // For samplePHS
    
    std::vector<double, double> center = {(start.x + goal.x) /2, (start.y + goal.y) /2};
    std::vector<double, double> a1 = {(goal.x - start.x)/cmin, (goal.y - start.y)/cmin};

    }
    // Bit_star::Bit_star(Node start, Node goal, std::string map_path)
    // {
    // this->start = start;
    // this->goal = goal;
    // // this->map = cv::imread(map_path, cv::IMREAD_GRAYSCALE);

    // this->vert.push_back(start);
    // this->unexp_vertex.push_back(start);
    // // TODO: Read map from file
    // generate_phs();

    // // For samplePHS
    
    // std::vector<double, double> center = {(start.x + goal.x) /2, (start.y + goal.y) /2};
    // std::vector<double, double> a1 = {(goal.x - start.x)/cmin, (goal.y - start.y)/cmin};

    // }


    // variables
    std::priority_queue<Node> vertex_q;
    std::priority_queue<Node> edge_q;
    std::vector<Node> x_reuse;
    std::vector<std::tuple<int, int>> intersection;
    Eigen::MatrixXd f_hat_map;
    std::vector<Eigen::Vector2d> xphs;
    int no_samples = 20;
    std::vector<Node> x_new;
    std::vector<Node> vert;
    std::vector<Edge> edges;

    
    
    
    
    
    
    
    
    
    
    // functions
    double gt(Node node);
    void generate_phs();
    double c_hat(Node node1, Node node2);
    std::vector<Node> near(Node node, std::vector<Node> search_list);
    std::vector<double> sample_unit_ball(int d);
    std::vector<Node> samplePHS();
    void get_PHS();
    void get_f_hat_map();
    bool intersection_check(Eigen::Vector2d node);
    Node sample();
    std::vector<Node> prune();
    std::vector<Eigen::Vector2d> final_solution();
};
