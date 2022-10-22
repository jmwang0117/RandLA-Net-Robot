import os
import zmq
import time
import pickle
import numpy as np
import open3d as o3d
from os.path import join, exists
from sklearn.neighbors import KDTree
from test_SemanticKITTI import Tester
from utils.data_process import DataProcessing as DP
from utils.config import ConfigSemanticKITTI as cfg
output_path = "/home/RandLA-Net-pytorch/robot/100/"
#output_path = "/dev/shm/robot/100/"
pc_path_out = join(output_path, 'velodyne')
KDTree_path_out = join(output_path, 'KDTree')
proj_path = join(output_path, 'proj')
os.makedirs(proj_path) if not exists(proj_path) else None
os.makedirs(pc_path_out) if not exists(pc_path_out) else None
os.makedirs(KDTree_path_out) if not exists(KDTree_path_out) else None

def data_prepare_semanticKITTI():
        pc_path = "/home/RandLA-Net-pytorch/data/sequence/100/velodyne/"
        #pc_path = "/dev/shm/data/sequence/100/velodyne/"
        scan_list = sorted(os.listdir(pc_path))
        #for id in range(len(scan_list)):
        for id in range(1):
                print(scan_list[id])
                file_name = pc_path + str(scan_list[id])
                points = np.fromfile(file_name, dtype=np.float).reshape((-1, 6))
                #points = np.float32(message[:, :3])
                points = np.float32(points[:, :3])
                #print("points:" + str(points))
                # sub_points = DP.grid_sub_sampling(points, grid_size=0.06)
                sub_points = points
                search_tree = KDTree(sub_points)
                proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
                proj_inds = proj_inds.astype(np.int32)
                KDTree_save = join(KDTree_path_out, str(scan_list[id][:-4]) + '.pkl')
                proj_save = join(proj_path, str(scan_list[id][:-4]) + '_proj.pkl')
                np.save(join(pc_path_out, str(scan_list[id][:-4])), sub_points)
                os.remove(file_name)
                with open(KDTree_save, 'wb') as f:
                    pickle.dump(search_tree, f)
                    #print("search_tree:" + str(sub_points))
                with open(proj_save, 'wb') as f:
                    pickle.dump([proj_inds], f)
                    #print("[proj_inds]:" + str([proj_inds]))

def get_voxels(point_cloud):
        """voxelization."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
        ret = []
        size = 0.5
        pcd_vox, _ , iddx = pcd.voxel_down_sample_and_trace(size, pcd.get_min_bound(), pcd.get_max_bound(), False)
        points_xyz = np.asarray(pcd_vox.points)
        
        for i in range(len(iddx)):
                vox_class = []
                for j in range(len(iddx[i])):
                        vox_class.append(point_cloud[iddx[i][j]][3])
                mid_vox = (*points_xyz[i], size, max(set(vox_class),key=vox_class.count))
                ret.append(mid_vox)
        return ret

def classify_points():
        """Return class id of points."""
        points_prediction_path = "/home/RandLA-Net-pytorch/result/100/predictions/"
        #points_prediction_path = "/dev/shm/result/100/predictions/"
        predict_list = sorted(os.listdir(points_prediction_path))
        predict_np = points_prediction_path + predict_list[0]
        print(predict_np)
        points_predictions_xyzl = np.load(predict_np)
        os.remove(predict_np)
        vox_label = get_voxels(points_predictions_xyzl)        #return [(0.1, -0.3, 0.2, 1), ...]
        return vox_label

def safe_float(input_item):
        floats = []
        for element in input_item:
            floats.append(float(element))
        return floats

def begin_seg(message):
        mid_list = [list(i) for i in message] # tuple to list
        mid_list = [safe_float(i) for i in mid_list] # dtype to float
        mid_list =np.array(mid_list) 
        #print(mid_list)
        mid_list.tofile("/home/RandLA-Net-pytorch/data/sequence/100/velodyne/000000.bin")     
        # data preprocessing
        data_prepare_semanticKITTI()
        # load dataset
        run_point.get_dataset()
        # point cloud semantic segmentation
        run_point.test()
        # point cloud to voxel
        vox_label = classify_points()
        #print(vox_label)
        return vox_label

if __name__ == '__main__':
        # Load the point cloud semantic segmentation model
        run_point = Tester()
        begin_seg([(*p, 128, 128, 128) for p in np.linspace((-5, -5, -5), (5, 5, 5), 50000)])
        context = zmq.Context()
        socket = context.socket(zmq.REP)  # Set the type of socket, zmq.REP reply
        socket.bind("tcp://*:34567")  # Bind the IP and port of the server
        print('start serving...')
        while True:
                message = socket.recv()  # Receive messages from clients
                message = pickle.loads(message)
                # print(message) #[(x,y,z,r,g,b),(x,y,z,r,g,b)]
                if len(message) < cfg.num_points:
                        socket.send(pickle.dumps(None)) # send voxel lable
                        continue       
                vox_label = begin_seg(message)
                # [(x,y,z,size,id),(x,y,z,size,id)]
                socket.send(pickle.dumps(vox_label)) # send voxel lable

