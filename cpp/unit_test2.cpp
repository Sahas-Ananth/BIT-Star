#include <bitstar.h>

// TODO: add parent and children to the nodes whenever we are adding an edge
// TODO: 


std::vector<double> Bit_star::sample_unit_ball(int d)
{
    
    // uniform random sample from unit ball
    std::vector<double> u(d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);



    for (int i = 0; i < d; i++) {
        u[i] = dis(gen);
    }

    double norm = 0.0;
    for (int i = 0; i < d; i++) {
        norm += u[i] * u[i];
    }
    norm = std::sqrt(norm);

    std::uniform_real_distribution<double> dis_r(0.0, 1.0);
    double r = dis_r(gen);

    std::vector<double> x(d);
    for (int i = 0; i < d; i++) {
        x[i] = r * u[i] / norm;
    }

    return x;

}


bool Bit_star::intersection_check(Eigen::Vector2d node){

    Node node1(node(0), node(1));

    if(std::find(intersection.begin(), intersection.end(), node1) != intersection.end()){
        return true;
    }
    else{
        return false;
    }
}


Node Bit_star::samplePHS(){

    // SVD
    Eigen::Matrix2d U, Vt;
    Eigen::Vector2d one_1(1.0, 0.0);
    Eigen::Matrix2d a1_outer_one_1 = a1 * one_1.transpose();


    Eigen::JacobiSVD<Eigen::Matrix2d> svd(a1_outer_one_1, Eigen::ComputeFullU | Eigen::ComputeFullV);

    U = svd.matrixU();
    Eigen::Vector2d S  = svd.singularValues();
    Vt = svd.matrixV().transpose();

    Eigen::Matrix2d Sigma = S.asDiagonal();
    Eigen::Matrix2d lam = Eigen::Matrix2d::Identity(Sigma.rows(), Sigma.cols());
    lam(lam.rows() - 1, lam.cols() - 1) = U.determinant() * Vt.transpose().determinant();
    Eigen::Matrix2d cwe = U * lam * Vt;

    std::vector<double> rn(dim - 1, std::sqrt(std::pow(ci, 2) - std::pow(cmin, 2)) / 2);
    Eigen::MatrixXd r(dim, dim);
    r.setZero();
    r(0, 0) = ci / 2;
    for (int i = 1; i < dim; i++)
    {
        r(i, i) = rn[i - 1];
    }

    Eigen::Vector2d output;

    while(true){

        std::vector<double> xball = sample_unit_ball(dim);
        // convert xball to Eigen::Vector2d
        Eigen::Vector2d xball_eigen(xball[0], xball[1]);
        output = center + (cwe * r) * xball_eigen;

        //  self.intersection - can be checked in another way?
        Eigen::Vector2d int_output{ int(output(0)), int(output(1)) };

        // check for intersection in intersect function
        if(intersection_check(int_output)){
            break;

        }

   }

    Node xrand = Node(output(0), output(1), &start, &goal);
  
    return xrand;


}

Node Bit_star::sample_map(){

    while(true){

        double x = static_cast<double>(rand()) / RAND_MAX * map_width;
        double y = static_cast<double>(rand()) / RAND_MAX * map_height;

        if (map(static_cast<int>(x), static_cast<int>(y)) != 0){

            Node node = Node(x, y, &start, &goal);
            return node;

        }

    }



}


void Bit_star::free_nodes_map(){

    for(int i=0; i<map.rows(); i++){
        for(int j=0; j<map.cols(); j++){
            if(map(i,j) > 0){
                free_nodes.push_back(Node(i,j,&start,&goal));
            }
        }
    }

}


void Bit_star::f_hat_map_data(){

    
    for (int x = 0; x < map.rows(); x++) {
        for (int y = 0; y < map.cols(); y++) {
        double f_hat = std::sqrt(std::pow(x - goal.x, 2) + std::pow(y - goal.y, 2)) 
                       + std::sqrt(std::pow(x - start.x, 2) + std::pow(y - start.y, 2));
        f_hat_map(x, y) = f_hat;
        }
    }        

}

void Bit_star::get_PHS(){

    for(int i=0; i<f_hat_map.rows(); i++){
        for(int j=0; j<f_hat_map.cols(); j++){
            if(f_hat_map(i,j) < ci){
                xphs.push_back(Node(i,j,&start,&goal));
            }
        }
    }

    old_ci = ci;

    
    std::set<Node> set_xphs(xphs.begin(), xphs.end());
    std::set<Node> set_free_nodes(free_nodes.begin(), free_nodes.end());
    intersection.clear();
    std::set_intersection(set_xphs.begin(), set_xphs.end(),
                        set_free_nodes.begin(), set_free_nodes.end(),
                        std::inserter(intersection, intersection.begin()));



}


Node Bit_star::sample(){


    Node xrand;
    if(old_ci!=ci){
        get_PHS();
    }

    if(xphs.size() < map.rows()*map.cols()){
        
        xrand = samplePHS();

    }
    else{

        xrand = sample_map();

    }

    return xrand;

}




std::vector<Node> Bit_star::near(std::vector<Node> search_list, Node node){

    std::vector<Node> near_list;
    for (int i = 0; i < search_list.size(); i++)
    {
        if ((c_hat(search_list[i], node) <= Rbit) && (search_list[i].x != node.x && search_list[i].y != node.y))
        {
            near_list.push_back(search_list[i]);
        }
    }

    return near_list;

}

void Bit_star::remove_node(Node *node){


    // remove node from edges
    for (int i = 0; i < this->edges.size(); i++)
    {
        if (this->edges[i].to_node == *node || this->edges[i].from_node == *node)
        {
            this->edges.erase(this->edges.begin() + i);
        }
    }

    // remove node from unconnected_vertex
    if (std::find(this->unconnected_vertex.begin(), this->unconnected_vertex.end(), *node) != this->unconnected_vertex.end())
    {
        this->unconnected_vertex.erase(std::remove(this->unconnected_vertex.begin(), this->unconnected_vertex.end(), *node), this->unconnected_vertex.end());

    }

    // remove node from unexp_vertex
    else 
    {
    // remove node from connected_vertex
    if (std::find(this->connected_vertex.begin(), this->connected_vertex.end(), *node) != this->connected_vertex.end())
    {
        // remove the children of the node
        for(auto node : node->children){
            this->remove_node(node);
        }

        this->connected_vertex.erase(std::remove(this->connected_vertex.begin(), this->connected_vertex.end(), *node), this->connected_vertex.end());
    }
    }
    
    

}

std::vector<Node> Bit_star::prune(){

    std::vector<Node> x_reuse;
    for(auto node : this->unconnected_vertex){
        if(node.f_hat >= ci){
            // x_reuse.push_back(node);
            this->remove_node(&node);
        }
    }


    std::sort(connected_vertex.begin(), connected_vertex.end(), [](const Node& a, const Node& b) {
        return a.gt < b.gt;
    });


    for(auto node : this->connected_vertex){
        if( !(node== this->start) && !(node== goal)){

            if(node.f_hat >= ci || (node.gt + node.h_hat > ci)){
                // x_reuse.push_back(node);
                this->remove_node(&node);

            }
            // if node is in unexp_vertex, remove from unexp_vertex
            if (std::find(this->unexp_vertex.begin(), this->unexp_vertex.end(), node) != this->unexp_vertex.end())
            {
                this->unexp_vertex.erase(std::remove(this->unexp_vertex.begin(), this->unexp_vertex.end(), node), this->unexp_vertex.end());
            }

            if(node.f_hat < ci){

                x_reuse.push_back(node);

            }


        }
           
    }
    return x_reuse;

}

void Bit_star::expand_next_vertex()
{

    Node vmin = vertex_q.top();
    vertex_q.pop();
    std::vector<Node> near_list;
    // check if vmin is in unexp_vertex
    if (std::find(this->unexp_vertex.begin(), this->unexp_vertex.end(), vmin) != this->unexp_vertex.end())
    {
        near_list = near(this->unexp_vertex, vmin);
    }
    else
    {   
        
        std::vector<Node> intersect;
        for (const auto& node : x_new) {
            for(const auto& node2 : this->unexp_vertex){
                if(node.x == node2.x && node.y == node2.y){
                    intersect.push_back(node);
                }
            }
        }

        near_list = near(intersect, vmin);

    }

    for(auto near : near_list){
        
        if(vmin.g_hat + c_hat(vmin, near) + near.h_hat  < ci) {

            Edge edge = Edge(vmin, near, c_hat(vmin, near));
            this->edge_q.push(edge);
        }
    }

    // check if vmin is in unexp_vertex
    if (std::find(this->unexp_vertex.begin(), this->unexp_vertex.end(), vmin) != this->unexp_vertex.end())
    {
        std::vector<Node> v_near = near(this->connected_vertex, vmin);
        for(auto near : v_near){

            // check if the edge exist in edges
            if (std::find(this->edges.begin(), this->edges.end(), Edge(vmin, near, c_hat(vmin, near))) == this->edges.end())
            {
                if((vmin.g_hat + c_hat(vmin, near) + near.h_hat  < ci ) && (near.g_hat + c_hat(vmin, near) < near.gt)){

                    Edge edge = Edge(vmin, near, c_hat(vmin, near));
                    this->edge_q.push(edge);
                }

                
            }
            
        }
        auto new_end = std::remove(this->unexp_vertex.begin(), this->unexp_vertex.end(), vmin);
        this->unexp_vertex.erase(new_end, this->unexp_vertex.end());

    }
}

double Bit_star::c_hat(Node node1, Node node2)
{
    return sqrt(pow(node1.x - node2.x, 2) + pow(node1.y - node2.y, 2));
}

double Bit_star::c(Node node1, Node node2)
{
    
    double x1 = node1.x;
    double y1 = node1.y;
    double x2 = node2.x;
    double y2 = node2.y;

    int n_divs = int(10 * sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)));

    for (double lam = 0; lam <= 1; lam += n_divs){

        int x = int(x1 + lam * (x2 - x1));
        int y = int(y1 + lam * (y2 - y1));

        if (x < 0 || x >= map.rows() || y < 0 || y >= map.cols() || map(x, y) != 0) {
            return std::numeric_limits<double>::infinity();
        }

    }
    return c_hat(node1, node2); 

}

double Bit_star::gt(Node node)
{

    if (node.x == start.x && node.y == start.y)
    {
        return 0.0;
    }
    //check:  assumed all the nodes in vert are conected?
    if (std::find(this->vert.begin(), this->vert.end(), node) == this->vert.end())
    {
        return std::numeric_limits<double>::infinity();
    }
    double length = 0.0;
    Node *current = &node;
    Node *parent = current->parent;
    while(parent->x != start.x && parent->y != start.y)
    {
        // what is weight here - c_hat between current and parent?
        length = length + c_hat(*current, *parent);
        current = parent;
        parent = current->parent;
    }
    return length;

}

bool Bit_star::nodeEqual(const Node& n1, const Node& n2) {
    if(n1.x == n2.x && n1.y == n2.y){
        return true;
    }
    return false;

}

int main()
{
    
    Node* start = new Node(0.0, 0.0);
    Node* goal = new Node(9.0, 9.0);
    Eigen::MatrixXd map = Eigen::MatrixXd::Ones(10, 10);

    Bit_star *tree = new Bit_star(*start, *goal, map);
    
    int iteration = 0;
    std::cout << "start" << std::endl;

    while(true){
        
        std::cout << "iteration: " << iteration << std::endl;
        iteration++;

        if(tree->edge_q.empty() && tree->vertex_q.empty()){
            
            std::cout << "edge_q and vertex_q are empty" << std::endl;
            tree->x_reuse = tree->prune();
           
            std::vector<Node> x_sampling;

            while(x_sampling.size() < tree->no_samples){
                Node node = tree->sample();
                // check if node is in x_Sampling
                if (std::find(x_sampling.begin(), x_sampling.end(), node) == x_sampling.end())
                {
                    x_sampling.push_back(node);
                }

            }

            for(auto node : x_sampling){
                tree->x_new.push_back(node);
            }
            for(auto node : tree->x_reuse){
                tree->x_new.push_back(node);
            }
            // remove duplicates
            auto newEnd = std::unique(tree->x_new.begin(), tree->x_new.end(),
                    [tree](const Node& n1, const Node& n2) {
                        return tree->nodeEqual(n1, n2);
                    });
            tree->x_new.erase(newEnd, tree->x_new.end());
                        
            for(auto node : x_sampling){
                Node newNode = Node(node.x, node.y, start, goal, tree->gt(node), true);
                tree->vert.push_back(newNode);
                tree->unconnected_vertex.push_back(newNode);
            }

            for(auto node: tree->connected_vertex){

                tree->vertex_q.push(node);

            }



       }

        while(true){

            if(tree->vertex_q.empty()){
                std::cout << "vertex_q is empty" << std::endl;
                break;
            }
            tree->expand_next_vertex();

            if(tree->edge_q.empty()){
                continue;
            }

            if(tree->vertex_q.empty() || (tree->vertex_q.top().vertex_weight) <= tree->edge_q.top().edge_weight){
                break;
           }

        }

        if(!(tree->edge_q.empty())){
            
            Edge edge = tree->edge_q.top();
            tree->edge_q.pop();

            if(edge.edge_weight < tree->ci){
                if(edge.from_node.gt + tree->c_hat(edge.from_node, edge.to_node) < edge.to_node.gt){
                   
                    double cedge = tree->c(edge.from_node, edge.to_node);
                    if(edge.from_node.gt + cedge + edge.to_node.h_hat < tree->ci){

                        if(edge.from_node.gt + cedge < edge.to_node.gt){
                        //    check if to_node exists in connected_vertex
                            if (std::find(tree->connected_vertex.begin(), tree->connected_vertex.end(), edge.to_node) != tree->connected_vertex.end())
                            {
                                // remove edge from connected_vertex and edges
                                for(auto node : tree->connected_vertex){
                                    if((node.x == edge.to_node.x && node.y == edge.to_node.y) && (node.parent->x == edge.from_node.x && edge.from_node.y == edge.from_node.y) ){
                                        tree->connected_vertex.erase(std::remove(tree->connected_vertex.begin(), tree->connected_vertex.end(), node), tree->connected_vertex.end());
                                        tree->edges.erase(std::remove(tree->edges.begin(), tree->edges.end(), edge), tree->edges.end());
                                    }
                                }
                                Edge e = Edge(edge.from_node, edge.to_node, edge.edge_weight);
                                tree->edges.push_back(e);

                                // check if edge.from node exists in connected_vertex
                                if (std::find(tree->connected_vertex.begin(), tree->connected_vertex.end(), edge.from_node) == tree->connected_vertex.end())
                                {
                                    tree->connected_vertex.push_back(edge.from_node);
                                }
                                if (std::find(tree->connected_vertex.begin(), tree->connected_vertex.end(), edge.to_node) == tree->connected_vertex.end())
                                {
                                    tree->connected_vertex.push_back(edge.to_node);
                                }

                            }

                            else{

                                tree->vert.push_back(edge.to_node);
                                Edge e = Edge(edge.from_node, edge.to_node, edge.edge_weight);
                                tree->edges.push_back(e);
                                if (std::find(tree->connected_vertex.begin(), tree->connected_vertex.end(), edge.from_node) == tree->connected_vertex.end())
                                {
                                    tree->connected_vertex.push_back(edge.from_node);
                                }
                                if (std::find(tree->connected_vertex.begin(), tree->connected_vertex.end(), edge.to_node) == tree->connected_vertex.end())
                                {
                                    tree->connected_vertex.push_back(edge.to_node);
                                }

                                tree->vertex_q.push(edge.to_node);
                                tree->unexp_vertex.push_back(edge.to_node);

                                if(edge.to_node.x == goal->x && edge.to_node.y == goal->y){
                                    tree->vsol.push_back(edge.to_node);
                                }
                            }


                            tree->ci = goal->gt;
                            if(edge.to_node.x == goal->x && edge.to_node.y == goal->y){

                                    std::cout << "goal found" << std::endl;




                            }


                            
                        }


                    }

                }



            }
            else{


                // empty the edge queue and vertex queue
                while(!tree->edge_q.empty()){
                    tree->edge_q.pop();
                }
                while(!tree->vertex_q.empty()){
                    tree->vertex_q.pop();
                }

            }




        }
        else{
            
            // empty the edge queue and vertex queue
            while(!tree->edge_q.empty()){
                tree->edge_q.pop();
            }
            while(!tree->vertex_q.empty()){
                tree->vertex_q.pop();
            }
        }

    }

    return 0;

}