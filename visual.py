import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_edges_file(filename):
    edge_data = []
    with open(filename, 'r') as file:
        next(file)  # skip header
        for line in file:
            data = line.strip().split(' ')
            edge_data.append(data)
    return edge_data

def read_nodes_file(filename):
    node_data = []
    with open(filename, 'r') as file:
        next(file)  # skip header
        for line in file:
            data = line.strip().split(' ')
            node_data.append(data)
    return node_data

def plot_edges(edges):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for edge in edges:
        id0, id1, x, y, z, qx, qy, qz, qw, s = edge
        ax.plot([float(x), float(x)+float(edge[2])], [float(y), float(y)+float(edge[3])], [float(z), float(z)+float(edge[4])], marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Edge Visualisation')
    plt.show()

def plot_nodes(nodes):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for node in nodes:
        id, x, y, z, qx, qy, qz, qw, s = node
        ax.scatter(float(x), float(y), float(z), color='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Node Visualisation')
    plt.show()

def plot_nodes_with_gt(nodes, gt_nodes):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot nodes
    for node in nodes:
        id, x, y, z, qx, qy, qz, qw, s = node
        ax.scatter(float(x), float(y), float(z), color='r', marker='o', label='Nodes')
    
    # Plot GT nodes
    for node in gt_nodes:
        id, x, y, z, qx, qy, qz, qw, s = node
        ax.scatter(float(x), float(y), float(z), color='b', marker='x', label='Ground Truth Nodes')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Node Visualisation with Ground Truth\nRed: Nodes, Blue: Ground Truth Nodes')
    plt.show()

# File paths
# nodes_file = 'data/scale_drift_circle/nodes.txt'

# edges_file = 'data/scale_drift_circle/edges.txt'
# nodes_file = 'output.txt'
# gt_file = 'data/scale_drift_circle/GT.txt'

# nodes_file = 'data/scale_jump_circle3/nodes.txt'

edges_file = 'data/scale_jump_circle3/edges.txt'
nodes_file = 'output.txt'
gt_file = 'data/scale_jump_circle3/GT.txt'

# Read data from files
edges_data = read_edges_file(edges_file)
nodes_data = read_nodes_file(nodes_file)
gt_data = read_nodes_file(gt_file)

# plot_nodes(nodes_data)



# Plot edges and nodes
plot_nodes_with_gt(nodes_data, gt_data)
