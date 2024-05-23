#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/block_solver.h>

using namespace std; 

typedef Eigen::Matrix<double, 3, 3> Matrix3d;
typedef Eigen::Matrix<double, 3, 1> Vector3d;
typedef Eigen::Matrix<double, 4, 4> Matrix4d;
typedef Eigen::Matrix<double, 8, 1> Vector8d;

struct Edge {
    int id0, id1;
    Vector8d vec;
};

struct Node {
    int id;
    Vector8d vec;
};


std::vector<Edge> readEdges(const std::string& filename) {
    std::vector<Edge> edges;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return edges;
    }

    int edge_nums;
    file >> edge_nums;
    edges.resize(edge_nums);

    for (int i = 0; i < edge_nums; ++i) {
        file >> edges[i].id0 >> edges[i].id1;
        Vector8d vec;
        file >> vec(0) >> vec(1) >> vec(2) >> vec(3) >> vec(4) >> vec(5) >> vec(6) >> vec(7);
        edges[i].vec = vec;
    }

    file.close();
    return edges;
}

std::vector<Node> readNodes(const std::string& filename) {
    std::vector<Node> nodes;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return nodes;
    }

    int node_nums;
    file >> node_nums;
    nodes.resize(node_nums);

    for (int i = 0; i < node_nums; ++i) {
        file >> nodes[i].id;
        Vector8d vec;
        file >> vec(0) >> vec(1) >> vec(2) >> vec(3) >> vec(4) >> vec(5) >> vec(6) >> vec(7);
        nodes[i].vec = vec;
    }

    file.close();
    return nodes;
}

Matrix4d v2t(Vector8d v) {
    Eigen::Vector3d translation = v.head<3>(); 
    Eigen::Quaterniond quaternion(v.segment<4>(3)); 
    double scale = v(7); 

    Eigen::Matrix3d Rotation = quaternion.normalized().toRotationMatrix();

    Matrix4d T = Matrix4d::Zero();
    T.topLeftCorner<3, 3>() = scale * Rotation;
    T.topRightCorner<3, 1>() = translation;
    T(3, 3) = 1;

    return T;
}

// transform T into a vector structure like this: x y z qx qy qz qw s
Vector8d t2v(Matrix4d T){
    Vector8d result;
    double scale = T.topLeftCorner<3, 3>().row(0).norm();
    result.head<3>() = T.topRightCorner<3, 1>(); // translation
    Eigen::Quaterniond rotation(T.topLeftCorner<3, 3>() / scale);

    // keep w positive
    if (rotation.w() < 0) {
        rotation.coeffs() *= -1;
    }

    result.segment<3>(3) = rotation.vec(); // quaternion
    result(6) = rotation.w();
    result(7) = scale; // scale
    return result;
}

Matrix4d accumulateTranslations(const std::vector<Edge>& edges, int id, int next_id) {
    Matrix4d accumulated_transform = Matrix4d::Identity(); // 初始变换矩阵为单位矩阵

    // 遍历边，叠加位移部分
    for (size_t i = id; i < next_id && i < edges.size(); ++i) {
        const Edge& edge = edges[i]; // 使用索引访问边
        Matrix4d pose = v2t(edge.vec); // 将边的向量转换为变换矩阵
        accumulated_transform = pose * accumulated_transform; // 叠加变换矩阵
    }

    // 将叠加后的变换矩阵转换回向量，并提取其中的位移部分
    return accumulated_transform;
}

// 定义顶点类型
class VertexPointXYZ : public g2o::BaseVertex<8, Vector8d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() {
        _estimate.setZero();
    }

    virtual void oplusImpl(const double* update) {
        _estimate += Vector8d(update);
    }

    virtual bool read(std::istream& /*is*/) { return false; }
    virtual bool write(std::ostream& /*os*/) const { return false; }
};

// 定义边类型
class EdgeXYZ : public g2o::BaseBinaryEdge<8, Vector8d, VertexPointXYZ, VertexPointXYZ> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void computeError() {
        const VertexPointXYZ* v1 = static_cast<const VertexPointXYZ*>(_vertices[0]);
        const VertexPointXYZ* v2 = static_cast<const VertexPointXYZ*>(_vertices[1]);

        const Vector8d& V1 = v1->estimate();
        const Vector8d& V2 = v2->estimate();

        Matrix4d T1 = v2t(V1);
        Matrix4d T2 = v2t(V2);
        Matrix4d T12 = T2 * T1.inverse();
  
        Vector8d v;
        v = t2v(T12) - _measurement;
        if(_measurement(7) == -1){
            Eigen::Matrix<double, 8, 8> matrix = Eigen::Matrix<double, 8, 8>::Zero();
            matrix.block<7, 7>(0, 0) = Eigen::Matrix<double, 7, 7>::Identity();
            _error = (v.transpose() * matrix).transpose();
        }else{
            _error = v;
        }
    }


    virtual bool read(std::istream& /*is*/) { return false; }
    virtual bool write(std::ostream& /*os*/) const { return false; }
};

int main() {
    // read from data files
    std::string edges_filename = "../../data/scale_jump_circle3/edges.txt";
    std::string nodes_filename = "../../data/scale_jump_circle3/nodes.txt";

    std::vector<Edge> edges = readEdges(edges_filename);
    std::vector<Node> nodes = readNodes(nodes_filename);

    // setting up optimizer
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<8, 8>> Block;

    auto linearSolver = std::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>();
    auto solver_ptr = std::make_unique<Block>(std::move(linearSolver));

    g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg(std::move(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    std::vector<VertexPointXYZ*> critical_vertices;
    std::vector<Eigen::Vector3d> bar_container;

    // Adding vertices
    std::vector<VertexPointXYZ*> vertices;
    for (const auto& node : nodes) {
        VertexPointXYZ* v = new VertexPointXYZ();
        v->setId(node.id);
        v->setEstimate(node.vec);
        optimizer.addVertex(v);
        vertices.push_back(v);
    }
    vertices[0]->setFixed(true); // fix first node

    // Adding edges
    for (size_t i = 0; i < edges.size(); ++i) {
        EdgeXYZ* e = new EdgeXYZ();
        e->setVertex(0, vertices[edges[i].id0]);
        e->setVertex(1, vertices[edges[i].id1]);
        e->setMeasurement(edges[i].vec);
        e->setInformation(Eigen::Matrix<double, 8, 8>::Identity());
        optimizer.addEdge(e);

        if(edges[i].vec(7) == -1){
            critical_vertices.push_back(vertices[edges[i].id1]);
        }
    }

    // construct critical graph
    if(critical_vertices.size() > 1){
        for (size_t i = 0; i < critical_vertices.size() - 1; ++i) {
            VertexPointXYZ* vertex = critical_vertices[i];
            VertexPointXYZ* next_vertex = critical_vertices[i+1];
            int id = vertex->id();
            int next_id = next_vertex->id();
            
            Matrix4d T = accumulateTranslations(edges, id, next_id);
            Vector3d v = T.topRightCorner<3, 1>();
            bar_container.push_back(v);
        }

        int last_id = critical_vertices.back()->id();
        int first_id = critical_vertices[0] -> id();
        Matrix4d T_1 = accumulateTranslations(edges, last_id, edges.size());
        Matrix4d T_2 = accumulateTranslations(edges, 0, first_id);
        Vector3d v_last = (T_2 * T_1).topRightCorner<3, 1>();
        bar_container.push_back(v_last);
    }

    int N = critical_vertices.size();
    int B = bar_container.size();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3 * B + 3, 3 * N + B);
    Eigen::MatrixXd x = Eigen::MatrixXd::Zero(N + B, 3);

    A.block(3 * B, 0, 3, 3).setIdentity();

    for (int k = 0; k < B-1; ++k) {
        int sk = k;
        int ek = k + 1; 

        // 设置第 sk 对应的矩阵为 -I
        A(3*k, 3*sk) = -1;
        A(3*k+1, 3*sk+1) = -1;
        A(3*k+2, 3*sk+2) = -1;

        // 设置第 ek 对应的矩阵为 I
        A(3*k, 3*ek) = 1;
        A(3*k+1, 3*ek+1) = 1;
        A(3*k+2, 3*ek+2) = 1;
    }

    for (int k = 0; k < B; ++k) {
        A(3*k, 3*N+k) = bar_container[k](0);
        A(3*k + 1, 3*N+k) = bar_container[k](1);
        A(3*k + 2, 3*N+k) = bar_container[k](2);
    }

    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(A);
    int rank = lu_decomp.rank();
    int numRows = A.rows();
    int numCols = A.cols();
    int rankDeficiency = std::min(numRows, numCols) - rank;

    std::cout << "The rank of A is: " << rank << std::endl;
    std::cout << "The rank deficiency of A is: " << rankDeficiency << std::endl;


    // optimize and record time
    auto start = std::chrono::high_resolution_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "running duration: " << duration.count() << " seconds" << std::endl;

    // save optimization result
    std::string output_filename = "../../output.txt";
    std::ofstream output_file(output_filename);
    if (output_file.is_open()) {
        for (const auto& v : vertices) {
            output_file << v->id() << " ";
            for (int i = 0; i < 8; ++i) {
                output_file << v->estimate()[i] << " ";
            }
            output_file << std::endl;
        }
        output_file.close();
        std::cout << "Vertices information has been written to " << output_filename << std::endl;
    } else {
        std::cerr << "Failed to open " << output_filename << " for writing." << std::endl;
    }

    return 0;
}
