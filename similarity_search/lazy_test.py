import numpy as np
import time



def test_method(method, k = 10, tests=100):
    from sklearn.neighbors import NearestNeighbors
    t0 = time.time()
    nn = NearestNeighbors(leaf_size=data.shape[0]).fit(data)

    score = 0.0
    t_nn = 0.0
    t_meth = 0.0
    np.random.seed(0)

    for i in range(tests):
        d = data[np.random.randint(data.shape[0])]

        t0 = time.time()
        method_res = method(d, k)
        t_meth += time.time()-t0

        t0 = time.time()
        nn_res = nn.kneighbors(d, n_neighbors=k, return_distance=False)
        t_nn += time.time()-t0

        score += np.mean(np.in1d(nn_res, method_res))

    t_nn /= tests
    t_meth /= tests

    r1 = 'NN time: %1.10f method time: %1.10f speedup: %1.10f' % (t_nn, t_meth, t_nn/t_meth)

    r2 = '%1.2f%% overlap' % ((score/tests) * 100)
    return r1 + '\n' + r2

def test_save_and_load(data, init_nn=3, init_index=3, init_search=3):
    import nmslib
    reload(nmslib)
    n = data.shape[0]
    space_type = 'l2'
    space_param = []
    method_name = 'small_world_rand'
    method_param = ['NN=%d'%init_nn,
                    'initIndexAttempts=%d'%init_index,
                    'initSearchAttempts=%d'%init_search,
                    'indexThreadQty=1',
                    'graphFileName=savedGraph.txt',
                    'saveGraphFile=1',
                    'loadGraphFile=0']
    index = nmslib.initIndex(n,
                             space_type,
                             space_param,
                             method_name,
                             method_param,
                             nmslib.DataType.VECTOR,
                             nmslib.DistType.FLOAT)
    t0 = time.time()
    for pos, d in enumerate(data):
        nmslib.setData(index, pos, d.tolist())

    nmslib.buildIndex(index)
    print 'Building %i dataset took %1.4f' % (data.shape[0], time.time()-t0)

    def query(q, k=10, m=3):
        return nmslib.knnQuery(index, k, q.tolist())

    print 'building score: '
    print test_method(query)

    nmslib.freeIndex(index)

    method_param = ['NN=%d'%init_nn,
                    'initIndexAttempts=%d'%init_index,
                    'initSearchAttempts=%d'%init_search,
                    'indexThreadQty=1',
                    'graphFileName=savedGraph.txt',
                    'saveGraphFile=0',
                    'loadGraphFile=1']
    index2 = nmslib.initIndex(n,
                             space_type,
                             space_param,
                             method_name,
                             method_param,
                             nmslib.DataType.VECTOR,
                             nmslib.DistType.FLOAT)
    t0 = time.time()
    for pos, d in enumerate(data):
        nmslib.setData(index2, pos, d.tolist())

    nmslib.buildIndex(index2)
    print 'Building %i dataset took %1.4f' % (data.shape[0], time.time()-t0)

    def query2(q, k=10, m=3):
        return nmslib.knnQuery(index2, k, q.tolist())

    print 'loading score: '
    print test_method(query2)
    nmslib.freeIndex(index2)

if __name__ == '__main__':
    import sys

    np.random.seed(0)
    data = np.random.rand(50000,100)

    test_save_and_load(data, init_nn=20, init_search=3, init_index=3)
