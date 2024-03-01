import sys
import time

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import hashlib
import heapq
from phe import paillier


class VO:
    def __init__(self, q):
        self.point = []
        self.q = q
        self.hash = []
        self.sig = []


class NearestNeighborResult:
    def __init__(self, point, distance, node):
        self.point = point
        self.distance = distance
        self.node = node


class kNNSearch:
    def __init__(self, k):
        self.k = k
        self.SSED = 0
        self.compare = 0
        self.SSED2 = 0
        self.compare2 = 0

    def encrypt_tree(self, pk, node):

        node.mbr = [pk.encrypt(int(x)) for x in node.mbr]

        if node.is_leaf:
            node.data = [(pk.encrypt(int(x[0])), pk.encrypt(int(x[1]))) for x in node.data]
            #node.ids
            V = []
            for v in node.V:
                tem = []
                for j in v:
                    tem.append((pk.encrypt(float(j[0])), pk.encrypt(float(j[1]))))
                V.append(tem)
            node.V = V

            #node.NVC
            #node.D
            #node.Sig

        else:
            for child in node.children:
                self.encrypt_tree(pk, child)

    def SED(self, R, p, sk):
        sigma = [x-p for x in R]
        dec = [sk.decrypt(x) for x in sigma]
        for i in dec:
            if i == 0:
                return 1
        return 0


    def encrypt_R_vo(self, pk, R, vo):
        tem_R = []
        for r in R:
            tem_R.append([pk.encrypt(int(r[0])), pk.encrypt(int(r[1]))])
        point = []
        for nvc in vo.point:
            tem = []
            for p in nvc:
                tem.append([pk.encrypt(int(p[0])), pk.encrypt(int(p[1]))])
            point.append(tem)
        vo.point = point
        return tem_R, vo

    def calculate_tree_size(self, node):
        mbr_size = sys.getsizeof(node.mbr[0])*4
        data_size = 0
        ids_size = 0
        V_size = 0
        NVC_size = 0
        D_size = 0
        Sig_size = 0
        if node.is_leaf:
            data_size = sys.getsizeof(node.data[0])*len(node.data)
            ids_size = sys.getsizeof(node.ids[0])*len(node.ids)
            for v in node.V:
                V_size += sys.getsizeof(v[0])*len(v)
            for nvc in node.NVC:
                NVC_size += sys.getsizeof(nvc[0])*len(nvc)
            D_size = sys.getsizeof(node.D[0])*len(node.D)
            Sig_size = sys.getsizeof(node.Sig[0])*len(node.Sig)

        inc = mbr_size + data_size + ids_size + V_size + NVC_size + D_size + Sig_size + sys.getsizeof(node)

        if node.is_leaf is False:
            for child in node.children:
                inc += self.calculate_tree_size(child)

        return inc

    def calculate_vo_size(self, vo):
        size = 0
        for nvc in vo.point:
            size += sys.getsizeof(nvc[0])*len(nvc)
        size += sys.getsizeof(vo.q) + sys.getsizeof(vo.hash[0])*len(vo.hash) + sys.getsizeof(vo.sig[0])*len(vo.sig)
        return size

    def vo_generate(self, query_point, R, points, root, NVC):

        vo = VO(query_point)

        R_ids = []
        for data in R:
            R_ids.append(points.index(data))

        R_distances = [np.linalg.norm(np.array(r) - np.array(query_point)) for r in R]
        self.SSED2 += len(R)
        sorted_indices = np.argsort(R_distances)
        self.compare2 += len(R)*len(R)/2
        sorted_R = [R[i] for i in sorted_indices]
        sorted_R_ids = [R_ids[i] for i in sorted_indices]

        for id in sorted_R_ids:
            nvc = []
            for i in NVC[id]:
                nvc.append(points[i])
            vo.point.append(nvc)

        result = []
        d = []
        sig = []
        root.search_points(root, sorted_R, result, d, sig)

        for r in sorted_R:
            for i, re in enumerate(result):
                if r == re:
                    vo.hash.append(d[i])
                    vo.sig.append(sig[i])
                    break

        return sorted_R, vo

    def verify(self, sk, R, vo):
        #decrypt
        tem_R = []
        for r in R:
            tem_R.append([sk.decrypt(r[0]), sk.decrypt(r[1])])
        R = tem_R
        tem = []
        for nvc in vo.point:
            tem_point = []
            for p in nvc:
                tem_point.append([sk.decrypt(p[0]), sk.decrypt(p[1])])
            tem.append(tem_point)
        vo.point = tem

        check_sig = True
        check_dist_equal = True
        check_dist_compare = True

        for i, sig in enumerate(vo.sig):

            data_hash = ''
            data_hash += hashlib.sha256((str(R[i][0]) + str(R[i][1])).encode()).hexdigest()
            data_hash += hashlib.sha256(vo.hash[i].encode()).hexdigest()
            data_hash = hashlib.sha256(data_hash.encode()).hexdigest()
            if sig != data_hash:
                check_sig = False
                break
            else:
                continue

        for i, point in enumerate(R):
            if i == 0:
                continue
            dist1 = np.linalg.norm(np.array(point) - np.array(vo.q))
            flag = False
            discard = []
            sub_R = R[0:i]
            nvc = []

            for j in range(i):
                for p in vo.point[j]:
                    if np.linalg.norm(np.array(p) - np.array(vo.q)) == dist1:
                        flag = True
                        discard = p
                        break
            if flag == False:
                check_dist_equal = False
                break

            for j in range(i):
                for p in vo.point[j]:
                    if p not in nvc and p not in sub_R:
                        nvc.append(p)

            nvc.remove(discard)
            nvc_distances = [np.linalg.norm(np.array(n) - np.array(vo.q)) for n in nvc]
            id = np.argmin(nvc_distances)
            Dmin = nvc_distances[id]
            if dist1 >= Dmin:
                check_dist_compare = False
                break

        return check_sig and check_dist_equal and check_dist_compare

    def NN(self, query_point, root):
        best_result = [None]

        def search_recursive(node):
            if node.is_leaf:
                # Leaf node, check each data point
                for i, data_point in enumerate(node.data):
                    distance = np.linalg.norm(np.array(query_point) - np.array(data_point))
                    self.SSED += 1
                    update_best_result(data_point, distance, node)
            else:
                # Non-leaf node, recursively search child nodes
                for child_node in node.children:
                    #if node.mbr_overlap(query_point, best_result[0].distance):
                    search_recursive(child_node)

        def update_best_result(data_point, distance, node):
            self.compare += 1
            if best_result[0] is None or distance < best_result[0].distance:
                best_result[0] = NearestNeighborResult(data_point, distance, node)

        search_recursive(root)
        return best_result[0]

    def kNN(self, points, query_point, nearestneighbor, NVC):
        visited = set()
        #heap = [(0, query_point)]
        heap = [(nearestneighbor.distance, tuple(nearestneighbor.point))]

        while heap and len(visited) < self.k:
            distance, current_point = heapq.heappop(heap)
            self.compare += 1
            if current_point in visited:
                continue

            visited.add(tuple(current_point))

            # Get Voronoi neighbors of the current point
            id = points.index(list(current_point))
            for index in NVC[id]:
                neighbor_point = points[index]
                neighbor_distance = np.linalg.norm(np.array(neighbor_point) - np.array(query_point))
                self.SSED += 1
                heapq.heappush(heap, (neighbor_distance, tuple(neighbor_point)))

        visited = list(visited)
        v_ids = []
        for data in visited:
            for i, point in enumerate(points):
                if tuple(point) == data:
                    index = i
                    break
            v_ids.append(index)
        result = []
        for v in v_ids:
            result.append(points[v])
        return result

    def query_maintenance(self, current_point, R, points, NVC, root):
        R_ids = []
        for data in R:
            R_ids.append(points.index(data))

        IS_ids = []
        IS = []
        for i in R_ids:
            nvc = NVC[i]
            for n in nvc:
                self.compare += 1
                if n not in IS_ids and n not in R_ids:
                    IS_ids.append(n)
        for i in IS_ids:
            IS.append(points[i])

        #R_distances = np.linalg.norm(np.array(R) - np.array(current_point))
        #IS_distances = np.linalg.norm(np.array(IS) - np.array(current_point))

        R_distances = [np.linalg.norm(np.array(r) - np.array(current_point)) for r in R]
        self.SSED += len(R)
        IS_distances = [np.linalg.norm(np.array(i) - np.array(current_point)) for i in IS]
        self.SSED += len(IS)

        delete = np.argmax(R_distances)
        self.compare += len(R)-1
        Dmax = R_distances[delete]
        candidate = np.argmin(IS_distances)
        self.compare += len(IS) - 1
        Dmin = IS_distances[candidate]

        self.compare += 1
        if Dmin < Dmax:
            R = self.update(current_point, R, IS, delete, candidate, R_ids, IS_ids, NVC, points, root)

        return R

    def update(self, current_point, R, IS, delete, candidate, R_ids, IS_ids, NVC, points, root):
        R_new = R
        delete_point = R_new.pop(delete)

        R_new.append(IS[candidate])
        candidate_NVC = NVC[IS_ids[candidate]]

        #添加元素
        IS_new = IS
        if delete_point not in IS:
            IS_new.append(delete_point)

        for i in candidate_NVC:
            self.compare += 1
            if points[i] not in IS_new:
                IS_new.append(points[i])

        #移除相同的元素
        for data in R_new:
            self.compare += 1
            if data in IS_new:
                IS_new.remove(data)

        #R_distances = np.linalg.norm(np.array(R_new) - np.array(current_point))
        #IS_distances = np.linalg.norm(np.array(IS_new) - np.array(current_point))

        R_distances = [np.linalg.norm(np.array(r) - np.array(current_point)) for r in R_new]
        self.SSED += len(R_new)
        IS_distances = [np.linalg.norm(np.array(i) - np.array(current_point)) for i in IS_new]
        self.SSED += len(IS_new)

        delete_new = np.argmax(R_distances)
        self.compare += len(R_new)-1
        Dmax = R_distances[delete_new]
        candidate_new = np.argmin(IS_distances)
        self.compare += len(IS_new)-1
        Dmin = IS_distances[candidate_new]

        self.compare += 1
        if Dmax >= Dmin:
            print("Need recomputation!")
            nearestneighbor = self.NN(current_point, root)
            R_new = self.kNN(points, current_point, nearestneighbor, NVC)

        return R_new

    def point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False

        # 射线起点
        p1x, p1y = polygon[0]

        for i in range(n + 1):
            # 射线终点
            p2x, p2y = polygon[i % n]

            # 检查射线是否与边界相交
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside

            # 移动到下一个顶点
            p1x, p1y = p2x, p2y

        return inside

class Node:
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.children = []
        self.mbr = None
        self.data = []
        self.ids = []
        self.V = [] #Voronoi单元的顶点坐标
        self.NVC = [] #Voronoi邻居点
        self.D = [] #某个点的所有邻居单元格的哈希值
        self.Sig = [] #签名

    def build_rtree(self, points, ids, V, NVC, max_children=4):
        root = self.build_rtree_recursive(points, ids, V, NVC, max_children)
        return root

    def build_rtree_recursive(self, points, ids, V, NVC, max_children):
        #if len(points) <= max_children:
        if len(ids) <= max_children:
            # Create a leaf node
            leaf_node = Node(is_leaf=True)
            leaf_node.data = [points[i] for i in ids]
            #leaf_node.ids = list(range(len(points)))
            leaf_node.ids = ids
            leaf_node.V = [V[i] for i in ids]
            leaf_node.NVC = [NVC[i] for i in ids]
            D = []
            for i in ids:
                points_id = NVC[i]
                data_hash = ''
                for j in points_id:
                    data_hash += hashlib.sha256((str(points[j][0])+str(points[j][1])).encode()).hexdigest()
                data_hash = hashlib.sha256(data_hash.encode()).hexdigest()
                D.append(data_hash)
            leaf_node.D = D
            Sig = []
            for ind, i in enumerate(ids):
                data_hash = ''
                data_hash += hashlib.sha256((str(points[i][0])+str(points[i][1])).encode()).hexdigest()
                data_hash += hashlib.sha256((leaf_node.D[ind]).encode()).hexdigest()
                data_hash = hashlib.sha256(data_hash.encode()).hexdigest()
                Sig.append(data_hash)
            leaf_node.Sig = Sig
            self.update_mbr(leaf_node)
            return leaf_node

        # Create a non-leaf node
        non_leaf_node = Node(is_leaf=False)
        non_leaf_node.children = self.split_points(points, ids, V, NVC, max_children)
        self.update_mbr(non_leaf_node)
        return non_leaf_node

    def split_points(self, points, ids, V, NVC, max_children):
        # Perform a simple split algorithm (linear split)
        num_splits = int(np.ceil(len(points) / max_children))
        split_size = int(np.ceil(len(points) / num_splits))

        children = []
        for i in range(0, len(points), split_size):
            child_ids = ids[i:i+split_size]
            child_node = self.build_rtree_recursive(points, child_ids, V, NVC, max_children)
            children.append(child_node)

        return children

    def update_mbr(self, node):
        if node.is_leaf:
            # For leaf nodes, update MBR based on data points
            if len(node.data) > 0:
                min_x = min(point[0] for point in node.data)
                max_x = max(point[0] for point in node.data)
                min_y = min(point[1] for point in node.data)
                max_y = max(point[1] for point in node.data)
                node.mbr = [min_x, min_y, max_x, max_y]
        else:
            # For non-leaf nodes, update MBR based on child nodes
            if len(node.children) > 0:
                min_x = min(child.mbr[0] for child in node.children)
                max_x = max(child.mbr[2] for child in node.children)
                min_y = min(child.mbr[1] for child in node.children)
                max_y = max(child.mbr[3] for child in node.children)
                node.mbr = [min_x, min_y, max_x, max_y]

    def search_rtree(self, node, query_mbr):
        results = []
        self.search_recursive(node, query_mbr, results)
        return results

    def search_recursive(self, node, query_mbr, results):
        if not self.mbr_overlap(node, query_mbr):
            return

        if node.is_leaf:
            for i, point in enumerate(node.data):
                if self.point_inside_mbr(point, query_mbr):
                    results.append((node.ids[i], point, node.V[i], node.NVC[i], node.D[i], node.Sig[i]))
        else:
            for child in node.children:
                self.search_recursive(child, query_mbr, results)

    def search_points(self, node, R, result, d, sig):
        if node.is_leaf:
            for i, point in enumerate(node.data):
                if point in R:
                    result.append(point)
                    d.append(node.D[i])
                    sig.append(node.Sig[i])
        else:
            for child in node.children:
                self.search_points(child, R, result, d, sig)

    def mbr_overlap(self, node, query_mbr):
        # Check if the MBR of the node overlaps with the query MBR
        return (
            node.mbr[0] <= query_mbr[2] and
            node.mbr[2] >= query_mbr[0] and
            node.mbr[1] <= query_mbr[3] and
            node.mbr[3] >= query_mbr[1]
        )

    def point_inside_mbr(self, point, mbr):
        return mbr[0] <= point[0] <= mbr[2] and mbr[1] <= point[1] <= mbr[3]

def get_vertices_neighbors(vor, points):

    points_region = vor.point_region

    #region对应的顶点索引
    region = []
    for i in points_region:
        region.append(vor.regions[i])

    vertices = []
    for i in range(len(points)):
        r = region[i]
        v = []
        for j in r:
            v.append(vor.vertices[j].tolist())
        vertices.append(v)

    neighbors = []
    # 打印每个输入点的邻居点序号
    for i, point in enumerate(points):
        neighbor = set()
        for ridge in vor.ridge_points:
            if i in ridge:
                neighbor.update(ridge)
        neighbor.discard(i)  # 移除自身
        neighbors.append(list(neighbor))

    return vertices, neighbors

#paillier加解密时间，增大6，7倍
paillier_dict = {512: {"enc":0.003, "dec":0.001},
                 1024: {"enc":0.02, "dec":0.007},
                 2048: {"enc":0.14, "dec":0.04}
                 }

#SSED指的是DPSSED, 一次SSED需要5次加密和1次解密
#密文下计算的相应开销，这里的是512位密钥的数据
enc_time_dict = {
    "compare": {"communication": 0.0001528, "compute": 0.004},
    "SSED": {"communication": 0.0001860, "compute": 0.016},
    # 可以继续添加其他键
}

if __name__ == "__main__":

    np.random.seed(0)

    #参数
    query_point = [100, 100]
    k = 1
    kappa = 512
    n = 2000
    M = 3
    s = 4

    #参数列表
    n_list = [2000,4000,6000,8000,10000]
    kappa_list = [512,1024,2048]
    k_list = [1,5,10,15,20]
    M_list = [3,5,7,9,11]
    s_list = [1,2,4,8,16,32]
    sed_list = [1,5,10,15,20]

    '''
    for sed in sed_list:
        pk, sk = paillier.generate_paillier_keypair(n_length=kappa)
        R = [pk.encrypt(int(x)) for x in range(1,sed+1)]
        p = pk.encrypt(int(2))
        search = kNNSearch(k)
        start = time.time()
        res = search.SED(R,p,sk)
        sed_time = time.time() - start
        print(f"sed_time:{sed_time}")
    '''

    for M in M_list:
        points = np.random.randint(1, 10000, size=(n, 2)).tolist()
        vor = Voronoi(points)
        pk, sk = paillier.generate_paillier_keypair(n_length=kappa)

        V, NVC = get_vertices_neighbors(vor, points)
        ids = [x for x in range(len(points))]

        #start = time.time()
        rtree = Node()
        root = rtree.build_rtree(points,ids, V, NVC, max_children=M)

        search = kNNSearch(k)
        #search.encrypt_tree(pk, root)
        #tree_time = time.time() - start
        #tree_size = search.calculate_tree_size(root)/(1024*1024)

        #输出
        print(f"参数：{M}")
        #print(f"time:{tree_time}")
        #print(f"size:{tree_size}")

        #start = time.time()
        nearest_neighbor = search.NN(query_point, root)
        #used1 = time.time() - start
        #nn_time = used1 + enc_time_dict['compare']['compute']*search.compare + enc_time_dict['SSED']['compute']*search.SSED
        #print(f"nn_time:{nn_time}")

        result = search.kNN(points, query_point, nearest_neighbor, NVC)

        compare = search.compare
        SSED = search.SSED

        #used1 = time.time() - start
        #knn_time = used1 + enc_time_dict['compare']['compute']*search.compare + enc_time_dict['SSED']['compute']*search.SSED
        #print(f"knn_time:{knn_time}")

        #start = time.time()
        #sorted_R, vo = search.vo_generate(query_point, result, points, root, NVC)
        #vo_time = time.time() - start + enc_time_dict['compare']['compute']*search.compare2 + enc_time_dict['SSED']['compute']*search.SSED2
        #vo_size = search.calculate_vo_size(vo)/1024
        #sorted_R, vo = search.encrypt_R_vo(pk, sorted_R, vo)
        #start2 = time.time()
        #check = search.verify(sk, sorted_R, vo)
        #verify_time = time.time() - start2
        #print(f"vo_time:{vo_time}")
        #print(f"vo_size:{vo_size}")
        #print(f"verify_time:{verify_time}")

        current_point = [100+s,100]
        start = time.time()
        result_new = search.query_maintenance(current_point, result, points, NVC, root)
        used1 = time.time() - start
        maintenance_time = used1+ enc_time_dict['compare']['compute'] * (search.compare-compare) + enc_time_dict['SSED']['compute'] * (search.SSED-SSED)
        print(f"maintenance_time:{maintenance_time}")
