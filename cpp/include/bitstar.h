#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>
#include <vector>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <set>

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
    // actual costs
    double gt;
    double f, g, h;  
    double vertex_weight;           // Actual costs
    Node *parent;
    std::vector<Node *> children;
    bool is_expanded; // We might use this
    Node *start;
    Node *goal;
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
        this->children = {};
    }
    Node(double x, double y)
    {
        this->x = x;
        this->y = y;
        this->f_hat = 0.0;
        this->g_hat = 0.0;
        this->h_hat = 0.0;
        this->f = 0.0;
        this->g = 0.0;
        this->h = 0.0;
        this->parent = NULL;
        this->children = {};

    }
    Node(double x, double y, Node *start, Node *goal)
    {
        this->x = x;
        this->y = y;
        this->start = start;
        this->goal = goal;
    }

    Node(int x, int y, Node *start, Node *goal)
    {
        this->x = static_cast<double>(x);
        this->y = static_cast<double>(y);
        this->start = start;
        this->goal = goal;
    }

    Node(double x, double y, Node *start, Node *goal, double gt,  bool self_calculate)
    {
        this->x = x;
        this->y = y;
        this->start = start;
        this->goal = goal;
        this->gt = gt;
        if(self_calculate)
        {
            this->g_hat_cal();
            this->h_hat_cal();
            this->f_hat_cal();
            this->vertex_weight = this->gt  + this->h_hat;
        }


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

   
    bool operator==(const Node& other) const {
        return (this->x == other.x) && (this->y == other.y);
    }
    bool operator<(const Node& other) const {
        // check: how to compare 2 nodes - needed for sets
        return this->g_hat< other.g_hat;
    }
};


class Edge {
    public:
        Node from_node;
        Node to_node;
        double c_hat;
        double edge_weight;


        Edge(Node from_node, Node to_node, double c_hat){
            this->from_node = from_node;
            this->to_node = to_node;
            this->c_hat = c_hat;
            this->edge_weight = from_node.gt + c_hat  + to_node.h_hat;
        }

        bool operator<(const Edge& other) const {
            return edge_weight > other.edge_weight;
        }

        bool operator==(const Edge& other) const {
            return ((this->from_node == other.to_node) && (this->to_node == other.from_node) || (this->from_node == other.from_node) && (this->to_node == other.to_node));
        }
};

struct NodeComparator {
    bool operator() (const Node& node1, const Node& node2) {

        // check: node wiht higher cost si pushed to the back

        return node1.gt + node1.h_hat > node2.gt + node2.h_hat;
        }
    };

struct NodeComparatorSort {
    bool operator() (const Node& node1, const Node& node2) {


        return node1.gt + node1.h_hat < node2.gt + node2.h_hat;
        }
    };  
    

class Bit_star
{
private:
    
    Node start;
    Node goal;

    double Rbit = 100.0;
    
    
    int dimension = 2;
    
    double cur_cost = std::numeric_limits<double>::infinity();
    double old_cost = cur_cost;
    
    std::vector<Node> x_phs;
    int dim = 2;
    
    double cmin;

    
    



public:
    Bit_star(Node start_node, Node goal_node, Eigen::MatrixXd map)
    {
    start = start_node;
    goal = goal_node;
    this->map = map;
    
    // add node  = vert. and self.V = unconnected_vertex
    this->vert.push_back(start);
    this->vert.push_back(goal);
    this->unexp_vertex.push_back(start);

    // goal is not connected to the graph and hence unconnected
    this->unconnected_vertex.push_back(goal);
    this->x_new = this->unconnected_vertex;
    // TODO: Read map from file
    // Assuming map is a 2D matrix of 10 x 10 for now

    // For samplePHS
    cmin = sqrt(pow(goal.x - start.x, 2) + pow(goal.y - start.y, 2));
    center = { (start.x + goal.x) / 2, (start.y + goal.y) / 2 };
    a1 = { (goal.x - start.x) / cmin, (goal.y - start.y) / cmin };
    one_1 = {{1, 0}};

    map_width = map.rows();
    map_height = map.cols();
    std::cout << "map_width: " << map_width << std::endl;
    std::cout << "map_height: " << map_height << std::endl;
    f_hat_map = Eigen::MatrixXd::Zero(map_width, map_height);
    free_nodes_map();

   
    }
    

    // variables
    // vertex queue , cost = gt + h_hat of the node.
    // std::priority_queue<Node> vertex_q;
    std::priority_queue<Node, std::vector<Node>, NodeComparator> vertex_q;
    // std::priority_queue<Node, std::vector<Node>, decltype(&Node::operator())> vertex_q(&Node::operator());
    // edge queue, cost = gt + c_hat + h_hat 
    std::priority_queue<Edge> edge_q;

    std::vector<Node> x_reuse;
    std::set<Node> intersection;
    Eigen::MatrixXd f_hat_map;
    // std::vector<Eigen::Vector2d> xphs;
    int no_samples = 20;
    std::vector<Node> x_new;
    std::vector<Node> vert;
    std::vector<Edge> edges;
    std::vector<Node> unexp_vertex;
    std::vector<Node> unconnected_vertex;
    std::vector<Node> free_nodes;   
    // hash table for unconnected vertex
    // std::unordered_map<Node, bool> unconnected_map;
    std::vector<Node> connected_vertex;
    double ci = std::numeric_limits<double>::infinity();
    double old_ci = 0;
    std::vector<Node> vsol;
    std::vector<Node> xphs;
    Eigen::Vector2d center;
    Eigen::Vector2d  a1;
    std::vector<std::pair<double, double>> one_1;
    int map_width;
    int map_height;
    Eigen::MatrixXd map;
    
    // functions
    double gt(Node node);
    void generate_phs();
    double c_hat(Node node1, Node node2);
    std::vector<Node> near(Node node, std::vector<Node> search_list);
    std::vector<double> sample_unit_ball(int d);
    void get_PHS();
    void get_f_hat_map();
    Node sample();
    std::vector<Node> prune();
    std::vector<Eigen::Vector2d> final_solution();
    void expand_next_vertex();
    std::vector<Node> near(std::vector<Node> search_list, Node node);
    void remove_node(Node *node);
    bool nodeEqual(const Node& n1, const Node& n2);
    double c(Node node1, Node node2);
    Node samplePHS();
    Node sample_map();
    void f_hat_map_data();
    void free_nodes_map();
    bool intersection_check(Eigen::Vector2d node);
};
